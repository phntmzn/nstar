import Cocoa

struct DependencyStatus: Decodable {
    let available: Bool
    let version: String?
    let error: String?
}

struct BackendDependencies: Decodable {
    let torch: DependencyStatus
    let midiutil: DependencyStatus
}

struct ModelRuntime: Decodable {
    let name: String
    let seed: Int
    let hiddenSize: Int
    let embeddingSize: Int
    let epochs: Int
    let learningRate: Double
    let checkpointPath: String
    let checkpointExists: Bool
}

struct BackendStatusResponse: Decodable {
    let ok: Bool
    let bridgeVersion: Int
    let pythonExecutable: String
    let pythonVersion: String
    let platform: String
    let defaultModelName: String
    let defaultOutputDir: String
    let generationAvailable: Bool
    let dependencies: BackendDependencies
    let models: [ModelRuntime]
}

struct GenerationResponse: Decodable {
    let ok: Bool
    let savedCount: Int?
    let savedFiles: [String]?
    let outputDir: String?
    let modelName: String?
    let error: String?
}

struct GenerationRequest {
    let modelName: String
    let target: Int
    let outputDirectory: String
    let producers: Int
    let namePrefix: String
    let epochs: Int
    let hiddenSize: Int
    let embeddingSize: Int
    let learningRate: Double
    let trainLogEvery: Int
    let retrainModel: Bool

    func bridgeArguments() -> [String] {
        var arguments = [
            "generate",
            "--model-name", modelName,
            "--target", String(target),
            "--output-dir", outputDirectory,
            "--producers", String(producers),
            "--name-prefix", namePrefix,
            "--epochs", String(epochs),
            "--hidden-size", String(hiddenSize),
            "--embedding-size", String(embeddingSize),
            "--learning-rate", String(learningRate),
            "--train-log-every", String(trainLogEvery),
        ]

        if retrainModel {
            arguments.append("--retrain-model")
        }

        return arguments
    }
}

enum BackendBridgeError: LocalizedError {
    case missingBackendScript
    case invalidRuntimePath(String)
    case invalidResponse(String)
    case executionFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingBackendScript:
            return "Unable to locate a bundled backend or gui_bridge.py in the app bundle or source tree."
        case let .invalidRuntimePath(path):
            return "Runtime executable is not valid: \(path)"
        case let .invalidResponse(message):
            return "Backend returned an invalid response: \(message)"
        case let .executionFailed(message):
            return message
        }
    }
}

final class BackendBridge {
    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    let scriptURL: URL?
    let bundledBackendURL: URL?

    init(
        scriptURL: URL? = BackendBridge.locateScript(),
        bundledBackendURL: URL? = BackendBridge.locateBundledBackend()
    ) {
        self.scriptURL = scriptURL
        self.bundledBackendURL = bundledBackendURL
    }

    static func locateScript() -> URL? {
        let fileManager = FileManager.default
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidates = [
            Bundle.main.resourceURL?.appendingPathComponent("backend/nstar/gui_bridge.py"),
            sourceRoot.appendingPathComponent("nstar/gui_bridge.py"),
            URL(fileURLWithPath: fileManager.currentDirectoryPath).appendingPathComponent("nstar/gui_bridge.py"),
        ].compactMap { $0 }

        return candidates.first(where: { fileManager.fileExists(atPath: $0.path) })
    }

    static func locateBundledBackend() -> URL? {
        let fileManager = FileManager.default
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidates = [
            Bundle.main.resourceURL?.appendingPathComponent("backend/nstar-backend"),
            Bundle.main.resourceURL?.appendingPathComponent("backend/nstar-backend/nstar-backend"),
            sourceRoot.appendingPathComponent("dist-swift/NStar.app/Contents/Resources/backend/nstar-backend"),
            sourceRoot.appendingPathComponent("dist-swift/NStar.app/Contents/Resources/backend/nstar-backend/nstar-backend"),
        ].compactMap { $0 }

        return candidates.first(where: { fileManager.isExecutableFile(atPath: $0.path) })
    }

    func defaultPythonPath() -> String {
        let fileManager = FileManager.default
        var candidates: [String] = []
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        if let bundledBackendURL {
            candidates.append(bundledBackendURL.path)
        }

        if let scriptURL {
            let backendRoot = scriptURL.deletingLastPathComponent().deletingLastPathComponent()
            candidates.append(backendRoot.appendingPathComponent(".venv/bin/python3").path)
            candidates.append(backendRoot.appendingPathComponent(".venv/bin/python").path)
        }

        let bundleProjectRoot = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        candidates.append(bundleProjectRoot.appendingPathComponent(".venv/bin/python3").path)
        candidates.append(bundleProjectRoot.appendingPathComponent(".venv/bin/python").path)
        candidates.append(sourceRoot.appendingPathComponent(".venv/bin/python3").path)
        candidates.append(sourceRoot.appendingPathComponent(".venv/bin/python").path)

        if let resourceURL = Bundle.main.resourceURL {
            candidates.append(resourceURL.appendingPathComponent("python/bin/python3").path)
            candidates.append(resourceURL.appendingPathComponent("python/bin/python").path)
        }

        candidates.append(contentsOf: [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ])

        if let persistedPath = UserDefaults.standard.string(forKey: MainWindowController.pythonDefaultsKey),
           !persistedPath.isEmpty {
            candidates.append(persistedPath)
        }

        if let resolvedPath = candidates.first(where: { fileManager.isExecutableFile(atPath: $0) }) {
            return resolvedPath
        }

        return "/usr/bin/python3"
    }

    @discardableResult
    func fetchStatus(
        pythonPath: String,
        logHandler: @escaping (String) -> Void,
        completion: @escaping (Result<BackendStatusResponse, Error>) -> Void
    ) -> Process? {
        runBridge(
            pythonPath: pythonPath,
            arguments: ["status"],
            logHandler: logHandler,
            completion: completion
        )
    }

    @discardableResult
    func generate(
        pythonPath: String,
        request: GenerationRequest,
        logHandler: @escaping (String) -> Void,
        completion: @escaping (Result<GenerationResponse, Error>) -> Void
    ) -> Process? {
        runBridge(
            pythonPath: pythonPath,
            arguments: request.bridgeArguments(),
            logHandler: logHandler,
            completion: completion
        )
    }

    private func runBridge<Response: Decodable>(
        pythonPath: String,
        arguments: [String],
        logHandler: @escaping (String) -> Void,
        completion: @escaping (Result<Response, Error>) -> Void
    ) -> Process? {
        let fileManager = FileManager.default
        guard fileManager.isExecutableFile(atPath: pythonPath) else {
            DispatchQueue.main.async {
                completion(.failure(BackendBridgeError.invalidRuntimePath(pythonPath)))
            }
            return nil
        }

        let process = Process()
        let runtimeURL = URL(fileURLWithPath: pythonPath)
        if shouldUseBundledBackend(runtimeURL: runtimeURL) {
            process.executableURL = runtimeURL
            process.arguments = arguments
            process.currentDirectoryURL = runtimeURL.deletingLastPathComponent()
        } else {
            guard let scriptURL else {
                DispatchQueue.main.async {
                    completion(.failure(BackendBridgeError.missingBackendScript))
                }
                return nil
            }

            process.executableURL = runtimeURL
            process.arguments = [scriptURL.path] + arguments
            process.currentDirectoryURL = scriptURL.deletingLastPathComponent()
        }

        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let ioQueue = DispatchQueue(label: "NStarGUI.BackendBridge.io")
        var stderrData = Data()

        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }

            ioQueue.sync {
                stderrData.append(data)
            }

            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async {
                logHandler(text)
            }
        }

        process.terminationHandler = { process in
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            let stderrRemainder = stderrPipe.fileHandleForReading.readDataToEndOfFile()

            let stderrSnapshot = ioQueue.sync { () -> Data in
                stderrData.append(stderrRemainder)
                return stderrData
            }

            if !stderrRemainder.isEmpty {
                let text = String(decoding: stderrRemainder, as: UTF8.self)
                DispatchQueue.main.async {
                    logHandler(text)
                }
            }

            DispatchQueue.main.async {
                if stdoutData.isEmpty {
                    let stderrText = String(decoding: stderrSnapshot, as: UTF8.self)
                    completion(.failure(BackendBridgeError.executionFailed(stderrText.isEmpty ? "Backend exited without output." : stderrText)))
                    return
                }

                do {
                    let response = try self.decoder.decode(Response.self, from: stdoutData)
                    completion(.success(response))
                } catch {
                    let responseText = String(decoding: stdoutData, as: UTF8.self)
                    completion(.failure(BackendBridgeError.invalidResponse(responseText)))
                }
            }
        }

        do {
            try process.run()
            return process
        } catch {
            DispatchQueue.main.async {
                completion(.failure(error))
            }
            return nil
        }
    }

    private func shouldUseBundledBackend(runtimeURL: URL) -> Bool {
        if let bundledBackendURL, bundledBackendURL.path == runtimeURL.path {
            return true
        }

        let lastPathComponent = runtimeURL.lastPathComponent.lowercased()
        if lastPathComponent.hasPrefix("nstar-backend") {
            return true
        }

        return scriptURL == nil
    }
}
