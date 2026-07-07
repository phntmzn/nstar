import Cocoa

final class MainWindowController: NSWindowController {
    static let pythonDefaultsKey = "nstar.pythonPath"
    private static let outputDefaultsKey = "nstar.outputDir"
    private static let modelDefaultsKey = "nstar.modelName"
    private static let targetDefaultsKey = "nstar.target"
    private static let producersDefaultsKey = "nstar.producers"
    private static let prefixDefaultsKey = "nstar.namePrefix"
    private static let epochsDefaultsKey = "nstar.epochs"
    private static let hiddenDefaultsKey = "nstar.hiddenSize"
    private static let embeddingDefaultsKey = "nstar.embeddingSize"
    private static let learningRateDefaultsKey = "nstar.learningRate"
    private static let trainLogDefaultsKey = "nstar.trainLogEvery"
    private static let retrainDefaultsKey = "nstar.retrainModel"
    private static let trainAllDefaultsKey = "nstar.trainAllModels"

    private let bridge = BackendBridge()

    private let pythonPathField = NSTextField()
    private let modelPopUpButton = NSPopUpButton()
    private let retrainCheckbox = NSButton()
    private let trainAllCheckbox = NSButton()
    private let retrainButton = NSButton()
    private let outputDirectoryField = NSTextField()
    private let targetField = NSTextField()
    private let producersField = NSTextField()
    private let namePrefixField = NSTextField()
    private let epochsField = NSTextField()
    private let hiddenSizeField = NSTextField()
    private let embeddingSizeField = NSTextField()
    private let learningRateField = NSTextField()
    private let trainLogEveryField = NSTextField()
    private let runtimeField = NSTextField(labelWithString: "Runtime not checked.")
    private let statusField = NSTextField(labelWithString: "Select a bundled backend or Python runtime, then refresh.")
    private let logTextView = NSTextView()
    private let generateButton = NSButton()
    private let stopButton = NSButton()
    private let progressIndicator = NSProgressIndicator()

    private var activeProcess: Process?
    private var lastStatus: BackendStatusResponse?
    private var lastSuggestedPrefix = ""
    private var attemptedAutomaticRuntimeFallback = false

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1040, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "NStar"
        window.minSize = NSSize(width: 920, height: 680)

        super.init(window: window)
        configureInterface()
        loadDefaults()
        applyBackendDefaultsIfNeeded()
        refreshRuntime(nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    @objc private func browsePython(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Select Runtime"
        if panel.runModal() == .OK, let url = panel.url {
            pythonPathField.stringValue = url.path
            saveDefaults()
            refreshRuntime(nil)
        }
    }

    @objc private func chooseOutputDirectory(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select Output Folder"
        if panel.runModal() == .OK, let url = panel.url {
            outputDirectoryField.stringValue = url.path
            saveDefaults()
        }
    }

    @objc private func openOutputDirectory(_ sender: Any?) {
        let outputPath = outputDirectoryField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !outputPath.isEmpty else {
            showAlert(message: "Output directory is empty.")
            return
        }

        let url = URL(fileURLWithPath: outputPath)
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)
            NSWorkspace.shared.open(url)
        } catch {
            showAlert(message: "Unable to open output directory.", informativeText: error.localizedDescription)
        }
    }

    @objc private func modelSelectionChanged(_ sender: Any?) {
        let selectedModel = selectedModelName()
        if namePrefixField.stringValue.isEmpty || namePrefixField.stringValue == lastSuggestedPrefix {
            namePrefixField.stringValue = selectedModel
        }
        lastSuggestedPrefix = selectedModel
        saveDefaults()
    }

    @objc private func refreshRuntime(_ sender: Any?) {
        saveDefaults()
        let pythonPath = pythonPathField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pythonPath.isEmpty else {
            setStatus("Select a Python interpreter first.", color: NSColor.systemRed)
            return
        }

        setStatus("Checking Python runtime...", color: NSColor.controlTextColor)
        runtimeField.stringValue = "Inspecting \(pythonPath)"

        _ = bridge.fetchStatus(
            pythonPath: pythonPath,
            logHandler: { [weak self] text in
                self?.appendLog(text)
            },
            completion: { [weak self] result in
                self?.handleRuntimeStatus(result)
            }
        )
    }

    @objc private func generateSongs(_ sender: Any?) {
        guard activeProcess == nil else {
            return
        }

        do {
            let request = try buildRequest()
            saveDefaults()
            appendLog("\n[gui] Starting generation for model \(request.modelName)\n")
            setRunning(true)
            setStatus("Generating MIDI files...", color: NSColor.controlTextColor)
            let pythonPath = pythonPathField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)

            activeProcess = bridge.generate(
                pythonPath: pythonPath,
                request: request,
                logHandler: { [weak self] text in
                    self?.appendLog(text)
                },
                completion: { [weak self] result in
                    self?.activeProcess = nil
                    self?.setRunning(false)
                    self?.handleGenerationResult(result)
                }
            )

            if activeProcess == nil {
                setRunning(false)
            }
        } catch {
            showAlert(message: "Invalid generation settings.", informativeText: error.localizedDescription)
        }
    }

    @objc private func stopGeneration(_ sender: Any?) {
        guard let activeProcess else {
            return
        }

        setStatus("Stopping generation...", color: NSColor.controlTextColor)
        appendLog("[gui] Stop requested.\n")
        stopButton.isEnabled = false
        activeProcess.terminate()
    }

    @objc private func retrainModel(_ sender: Any?) {
        guard activeProcess == nil else {
            return
        }

        do {
            let request = try buildTrainRequest()
            saveDefaults()
            let target = request.trainAllModels ? "all models" : request.modelName
            appendLog("\n[gui] Starting training for \(target)\n")
            setRunning(true)
            setStatus("Training \(target)...", color: NSColor.controlTextColor)
            let pythonPath = pythonPathField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)

            activeProcess = bridge.train(
                pythonPath: pythonPath,
                request: request,
                logHandler: { [weak self] text in
                    self?.appendLog(text)
                },
                completion: { [weak self] result in
                    self?.activeProcess = nil
                    self?.setRunning(false)
                    self?.handleTrainResult(result)
                }
            )

            if activeProcess == nil {
                setRunning(false)
            }
        } catch {
            showAlert(message: "Invalid training settings.", informativeText: error.localizedDescription)
        }
    }

    private func configureInterface() {
        retrainCheckbox.setButtonType(.switch)
        retrainCheckbox.title = "Retrain checkpoint"
        retrainCheckbox.target = self
        retrainCheckbox.action = #selector(saveDefaultsFromControl(_:))

        trainAllCheckbox.setButtonType(.switch)
        trainAllCheckbox.title = "Train all models"
        trainAllCheckbox.target = self
        trainAllCheckbox.action = #selector(saveDefaultsFromControl(_:))

        retrainButton.title = "Retrain Model"
        retrainButton.bezelStyle = .rounded
        retrainButton.target = self
        retrainButton.action = #selector(retrainModel(_:))

        modelPopUpButton.target = self
        modelPopUpButton.action = #selector(modelSelectionChanged(_:))

        generateButton.title = "Generate"
        generateButton.bezelStyle = .rounded
        generateButton.target = self
        generateButton.action = #selector(generateSongs(_:))

        stopButton.title = "Stop"
        stopButton.bezelStyle = .rounded
        stopButton.target = self
        stopButton.action = #selector(stopGeneration(_:))
        stopButton.isEnabled = false

        progressIndicator.style = .spinning
        progressIndicator.controlSize = .small
        progressIndicator.isDisplayedWhenStopped = false

        runtimeField.font = NSFont.systemFont(ofSize: NSFont.smallSystemFontSize)
        statusField.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        statusField.textColor = NSColor.systemGreen
        outputDirectoryField.textColor = NSColor.systemGreen
        outputDirectoryField.usesSingleLineMode = true

        [pythonPathField, outputDirectoryField, namePrefixField].forEach { field in
            field.translatesAutoresizingMaskIntoConstraints = false
        }

        configureNumberField(targetField, value: "1")
        configureNumberField(producersField, value: "1")
        configureNumberField(epochsField, value: "600")
        configureNumberField(hiddenSizeField, value: "32")
        configureNumberField(embeddingSizeField, value: "16")
        configureNumberField(trainLogEveryField, value: "100")
        learningRateField.stringValue = "0.01"

        pythonPathField.placeholderString = "/path/to/python3"
        outputDirectoryField.placeholderString = "~/Documents/mid"
        outputDirectoryField.drawsBackground = false
        namePrefixField.placeholderString = "phntmzn"

        let headerLabel = NSTextField(labelWithString: "NStar")
        headerLabel.font = NSFont.boldSystemFont(ofSize: 24)
        let subtitleLabel = NSTextField(labelWithString: "Swift AppKit GUI with the existing Python and PyTorch backend.")
        subtitleLabel.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)

        let headerStack = NSStackView(views: [headerLabel, subtitleLabel])
        headerStack.orientation = .vertical
        headerStack.spacing = 4

        let pythonRow = makeRow(
            title: "Runtime",
            views: [
                pythonPathField,
                makeButton(title: "Browse", action: #selector(browsePython(_:))),
                makeButton(title: "Refresh", action: #selector(refreshRuntime(_:))),
            ]
        )
        let modelRow = makeRow(
            title: "Model",
            views: [
                modelPopUpButton,
                retrainCheckbox,
            ]
        )
        let outputRow = makeRow(
            title: "Output",
            views: [
                outputDirectoryField,
                makeButton(title: "Choose", action: #selector(chooseOutputDirectory(_:))),
                makeButton(title: "Open", action: #selector(openOutputDirectory(_:))),
            ]
        )
        let batchRow = makeRow(
            title: "Batch",
            views: [
                inlineField(title: "Count", field: targetField, width: 64),
                inlineField(title: "Producers", field: producersField, width: 64),
                inlineField(title: "Prefix", field: namePrefixField, width: 160),
            ]
        )
        let trainingRow = makeRow(
            title: "Training",
            views: [
                inlineField(title: "Epochs", field: epochsField, width: 70),
                inlineField(title: "Hidden", field: hiddenSizeField, width: 64),
                inlineField(title: "Embed", field: embeddingSizeField, width: 64),
                inlineField(title: "LR", field: learningRateField, width: 80),
                inlineField(title: "Log", field: trainLogEveryField, width: 64),
            ]
        )

        let actionRow = NSStackView(views: [generateButton, retrainButton, trainAllCheckbox, stopButton, progressIndicator])
        actionRow.orientation = .horizontal
        actionRow.alignment = .centerY
        actionRow.spacing = 8

        let logScrollView = NSScrollView()
        logScrollView.translatesAutoresizingMaskIntoConstraints = false
        logScrollView.hasVerticalScroller = true
        logScrollView.borderType = .bezelBorder
        logScrollView.documentView = logTextView

        logTextView.isEditable = false
        logTextView.isRichText = false
        logTextView.font = NSFont.userFixedPitchFont(ofSize: 12)
        logTextView.textContainerInset = NSSize(width: 8, height: 8)
        logTextView.autoresizingMask = [.width]
        logTextView.minSize = NSSize(width: 0, height: 420)
        logTextView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        logTextView.isHorizontallyResizable = false
        logTextView.isVerticallyResizable = true
        logTextView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        logTextView.textContainer?.widthTracksTextView = true

        let rootStack = NSStackView(views: [
            headerStack,
            pythonRow,
            modelRow,
            outputRow,
            batchRow,
            trainingRow,
            runtimeField,
            statusField,
            actionRow,
            logScrollView,
        ])
        rootStack.orientation = .vertical
        rootStack.spacing = 12
        rootStack.edgeInsets = NSEdgeInsets(top: 20, left: 20, bottom: 20, right: 20)
        rootStack.translatesAutoresizingMaskIntoConstraints = false

        let contentView = NSView()
        contentView.addSubview(rootStack)
        NSLayoutConstraint.activate([
            rootStack.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            rootStack.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            rootStack.topAnchor.constraint(equalTo: contentView.topAnchor),
            rootStack.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            logScrollView.heightAnchor.constraint(greaterThanOrEqualToConstant: 360),
        ])

        let viewController = NSViewController()
        viewController.view = contentView
        window?.contentViewController = viewController
    }

    private func configureNumberField(_ field: NSTextField, value: String) {
        field.stringValue = value
        field.alignment = .right
        field.target = self
        field.action = #selector(saveDefaultsFromControl(_:))
    }

    @objc private func saveDefaultsFromControl(_ sender: Any?) {
        saveDefaults()
    }

    private func makeButton(title: String, action: Selector) -> NSButton {
        let button = NSButton(title: title, target: self, action: action)
        button.bezelStyle = .rounded
        return button
    }

    private func inlineField(title: String, field: NSTextField, width: CGFloat) -> NSView {
        field.target = self
        field.action = #selector(saveDefaultsFromControl(_:))
        field.widthAnchor.constraint(equalToConstant: width).isActive = true

        let label = NSTextField(labelWithString: title)
        let stack = NSStackView(views: [label, field])
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 6
        return stack
    }

    private func makeRow(title: String, views: [NSView]) -> NSView {
        let label = NSTextField(labelWithString: title)
        label.alignment = .right
        label.font = NSFont.boldSystemFont(ofSize: NSFont.systemFontSize)
        label.widthAnchor.constraint(equalToConstant: 70).isActive = true
        label.setContentHuggingPriority(.required, for: .horizontal)

        views.forEach {
            $0.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        }

        let row = NSStackView(views: [label] + views)
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 8
        return row
    }

    private func loadDefaults() {
        let defaults = UserDefaults.standard

        pythonPathField.stringValue = defaults.string(forKey: Self.pythonDefaultsKey) ?? ""
        outputDirectoryField.stringValue = defaults.string(forKey: Self.outputDefaultsKey) ?? ""
        targetField.stringValue = defaults.string(forKey: Self.targetDefaultsKey) ?? targetField.stringValue
        producersField.stringValue = defaults.string(forKey: Self.producersDefaultsKey) ?? producersField.stringValue
        namePrefixField.stringValue = defaults.string(forKey: Self.prefixDefaultsKey) ?? ""
        epochsField.stringValue = defaults.string(forKey: Self.epochsDefaultsKey) ?? epochsField.stringValue
        hiddenSizeField.stringValue = defaults.string(forKey: Self.hiddenDefaultsKey) ?? hiddenSizeField.stringValue
        embeddingSizeField.stringValue = defaults.string(forKey: Self.embeddingDefaultsKey) ?? embeddingSizeField.stringValue
        learningRateField.stringValue = defaults.string(forKey: Self.learningRateDefaultsKey) ?? learningRateField.stringValue
        trainLogEveryField.stringValue = defaults.string(forKey: Self.trainLogDefaultsKey) ?? trainLogEveryField.stringValue
        retrainCheckbox.state = defaults.bool(forKey: Self.retrainDefaultsKey) ? .on : .off
        trainAllCheckbox.state = defaults.bool(forKey: Self.trainAllDefaultsKey) ? .on : .off
    }

    private func applyBackendDefaultsIfNeeded() {
        if pythonPathField.stringValue.isEmpty {
            pythonPathField.stringValue = bridge.defaultPythonPath()
        }

        if outputDirectoryField.stringValue.isEmpty {
            outputDirectoryField.stringValue = NSString(string: "~/Documents/mid").expandingTildeInPath
        }

        if bridge.scriptURL == nil {
            appendLog("[gui] gui_bridge.py was not found. Build the app bundle with backend resources or run from the repo root.\n")
            setStatus("Backend script is missing.", color: NSColor.systemRed)
        }
    }

    private func saveDefaults() {
        let defaults = UserDefaults.standard
        defaults.set(pythonPathField.stringValue, forKey: Self.pythonDefaultsKey)
        defaults.set(outputDirectoryField.stringValue, forKey: Self.outputDefaultsKey)
        defaults.set(selectedModelName(), forKey: Self.modelDefaultsKey)
        defaults.set(targetField.stringValue, forKey: Self.targetDefaultsKey)
        defaults.set(producersField.stringValue, forKey: Self.producersDefaultsKey)
        defaults.set(namePrefixField.stringValue, forKey: Self.prefixDefaultsKey)
        defaults.set(epochsField.stringValue, forKey: Self.epochsDefaultsKey)
        defaults.set(hiddenSizeField.stringValue, forKey: Self.hiddenDefaultsKey)
        defaults.set(embeddingSizeField.stringValue, forKey: Self.embeddingDefaultsKey)
        defaults.set(learningRateField.stringValue, forKey: Self.learningRateDefaultsKey)
        defaults.set(trainLogEveryField.stringValue, forKey: Self.trainLogDefaultsKey)
        defaults.set(retrainCheckbox.state == .on, forKey: Self.retrainDefaultsKey)
        defaults.set(trainAllCheckbox.state == .on, forKey: Self.trainAllDefaultsKey)
    }

    private func handleRuntimeStatus(_ result: Result<BackendStatusResponse, Error>) {
        switch result {
        case let .success(response):
            let currentPythonPath = pythonPathField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            let suggestedPythonPath = bridge.defaultPythonPath()

            if !response.generationAvailable,
               !attemptedAutomaticRuntimeFallback,
               !suggestedPythonPath.isEmpty,
               suggestedPythonPath != currentPythonPath {
                attemptedAutomaticRuntimeFallback = true
                pythonPathField.stringValue = suggestedPythonPath
                appendLog("[gui] Current runtime is missing dependencies. Retrying with \(suggestedPythonPath).\n")
                refreshRuntime(nil)
                return
            }

            attemptedAutomaticRuntimeFallback = false
            lastStatus = response
            populateModels(response.models, defaultModelName: response.defaultModelName)
            if outputDirectoryField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                outputDirectoryField.stringValue = response.defaultOutputDir
            }

            let torchSummary = response.dependencies.torch.available
                ? "torch \(response.dependencies.torch.version ?? "unknown")"
                : "torch missing"
            let midiSummary = response.dependencies.midiutil.available
                ? "midiutil \(response.dependencies.midiutil.version ?? "unknown")"
                : "midiutil missing"

            runtimeField.stringValue = "\(response.pythonExecutable) | Python \(response.pythonVersion) | \(torchSummary) | \(midiSummary)"

            if response.generationAvailable {
                let availableCheckpoints = response.models.filter(\.checkpointExists).count
                setStatus("Runtime ready. \(availableCheckpoints)/\(response.models.count) checkpoints already exist.", color: NSColor.systemGreen)
            } else {
                let missingParts = [
                    response.dependencies.torch.available ? nil : "PyTorch",
                    response.dependencies.midiutil.available ? nil : "midiutil",
                ].compactMap { $0 }.joined(separator: " and ")
                setStatus("Runtime incomplete: \(missingParts).", color: NSColor.systemRed)
            }

            updateControlState()

        case let .failure(error):
            attemptedAutomaticRuntimeFallback = false
            runtimeField.stringValue = "Runtime check failed."
            setStatus(error.localizedDescription, color: NSColor.systemRed)
            appendLog("[gui] Runtime check failed: \(error.localizedDescription)\n")
            updateControlState()
        }
    }

    private func populateModels(_ models: [ModelRuntime], defaultModelName: String) {
        let previousSelection = UserDefaults.standard.string(forKey: Self.modelDefaultsKey) ?? selectedModelName()
        modelPopUpButton.removeAllItems()
        modelPopUpButton.addItems(withTitles: models.map(\.name))

        let selectedName = models.contains(where: { $0.name == previousSelection }) ? previousSelection : defaultModelName
        if let index = models.firstIndex(where: { $0.name == selectedName }) {
            modelPopUpButton.selectItem(at: index)
        }

        modelSelectionChanged(nil)
    }

    private func handleGenerationResult(_ result: Result<GenerationResponse, Error>) {
        switch result {
        case let .success(response):
            if response.ok {
                let count = response.savedCount ?? 0
                let outputDirectory = response.outputDir ?? outputDirectoryField.stringValue
                setStatus("Saved \(count) MIDI file(s) to \(outputDirectory).", color: NSColor.systemGreen)
                appendLog("[gui] Generation finished. Saved \(count) MIDI file(s).\n")
            } else {
                let message = response.error ?? "Generation failed."
                setStatus(message, color: NSColor.systemRed)
                appendLog("[gui] Generation failed: \(message)\n")
            }

        case let .failure(error):
            setStatus(error.localizedDescription, color: NSColor.systemRed)
            appendLog("[gui] Backend error: \(error.localizedDescription)\n")
        }

        updateControlState()
    }

    private func handleTrainResult(_ result: Result<TrainResponse, Error>) {
        switch result {
        case let .success(response):
            let modelResults = response.models ?? []
            for modelResult in modelResults {
                if modelResult.ok {
                    let lossText = modelResult.finalLoss.map { String(format: "%.6f", $0) } ?? "unknown"
                    appendLog("[gui] Trained \(modelResult.modelName) (loss=\(lossText)) -> \(modelResult.checkpointPath ?? "unknown path")\n")
                } else {
                    appendLog("[gui] Training failed for \(modelResult.modelName): \(modelResult.error ?? "unknown error")\n")
                }
            }

            if response.ok {
                let count = modelResults.count
                setStatus("Training complete for \(count) model\(count == 1 ? "" : "s").", color: NSColor.systemGreen)
            } else {
                let message = response.error ?? "Training failed."
                setStatus(message, color: NSColor.systemRed)
            }

            // Checkpoints on disk may have changed; refresh runtime status so the
            // model list reflects the freshly trained checkpoints.
            refreshRuntime(nil)

        case let .failure(error):
            setStatus(error.localizedDescription, color: NSColor.systemRed)
            appendLog("[gui] Backend error: \(error.localizedDescription)\n")
        }

        updateControlState()
    }

    private func buildTrainRequest() throws -> TrainRequest {
        let epochs = try parsePositiveInteger(epochsField.stringValue, name: "Epochs")
        let hiddenSize = try parsePositiveInteger(hiddenSizeField.stringValue, name: "Hidden size")
        let embeddingSize = try parsePositiveInteger(embeddingSizeField.stringValue, name: "Embedding size")
        let trainLogEvery = try parseNonNegativeInteger(trainLogEveryField.stringValue, name: "Log frequency")

        guard let learningRate = Double(learningRateField.stringValue), learningRate > 0 else {
            throw ValidationError("Learning rate must be a positive number.")
        }

        return TrainRequest(
            modelName: selectedModelName(),
            epochs: epochs,
            hiddenSize: hiddenSize,
            embeddingSize: embeddingSize,
            learningRate: learningRate,
            trainLogEvery: trainLogEvery,
            trainAllModels: trainAllCheckbox.state == .on
        )
    }

    private func buildRequest() throws -> GenerationRequest {
        let outputPath = outputDirectoryField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !outputPath.isEmpty else {
            throw ValidationError("Output directory is required.")
        }

        let target = try parsePositiveInteger(targetField.stringValue, name: "Count")
        let producers = try parsePositiveInteger(producersField.stringValue, name: "Producers")
        let epochs = try parsePositiveInteger(epochsField.stringValue, name: "Epochs")
        let hiddenSize = try parsePositiveInteger(hiddenSizeField.stringValue, name: "Hidden size")
        let embeddingSize = try parsePositiveInteger(embeddingSizeField.stringValue, name: "Embedding size")
        let trainLogEvery = try parseNonNegativeInteger(trainLogEveryField.stringValue, name: "Log frequency")

        guard let learningRate = Double(learningRateField.stringValue), learningRate > 0 else {
            throw ValidationError("Learning rate must be a positive number.")
        }

        return GenerationRequest(
            modelName: selectedModelName(),
            target: target,
            outputDirectory: NSString(string: outputPath).expandingTildeInPath,
            producers: producers,
            namePrefix: namePrefixField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines),
            epochs: epochs,
            hiddenSize: hiddenSize,
            embeddingSize: embeddingSize,
            learningRate: learningRate,
            trainLogEvery: trainLogEvery,
            retrainModel: retrainCheckbox.state == .on
        )
    }

    private func selectedModelName() -> String {
        modelPopUpButton.selectedItem?.title ?? lastStatus?.defaultModelName ?? "phntmzn"
    }

    private func parsePositiveInteger(_ rawValue: String, name: String) throws -> Int {
        guard let value = Int(rawValue.trimmingCharacters(in: .whitespacesAndNewlines)), value > 0 else {
            throw ValidationError("\(name) must be a whole number greater than zero.")
        }
        return value
    }

    private func parseNonNegativeInteger(_ rawValue: String, name: String) throws -> Int {
        guard let value = Int(rawValue.trimmingCharacters(in: .whitespacesAndNewlines)), value >= 0 else {
            throw ValidationError("\(name) must be zero or greater.")
        }
        return value
    }

    private func setRunning(_ isRunning: Bool) {
        if isRunning {
            progressIndicator.startAnimation(nil)
        } else {
            progressIndicator.stopAnimation(nil)
        }

        updateControlState(isRunning: isRunning)
    }

    private func updateControlState(isRunning: Bool? = nil) {
        let running = isRunning ?? (activeProcess != nil)
        let runtimeReady = lastStatus?.generationAvailable ?? false
        generateButton.isEnabled = runtimeReady && !running
        retrainButton.isEnabled = runtimeReady && !running
        trainAllCheckbox.isEnabled = !running
        stopButton.isEnabled = running
        modelPopUpButton.isEnabled = !running
        retrainCheckbox.isEnabled = !running
    }

    private func setStatus(_ message: String, color: NSColor) {
        statusField.stringValue = message
        statusField.textColor = color
        statusField.textColor = NSColor.systemGreen
    }

    private func appendLog(_ text: String) {
        guard !text.isEmpty else {
            return
        }

        if Thread.isMainThread {
            logTextView.textStorage?.append(NSAttributedString(string: text))
            logTextView.scrollToEndOfDocument(nil)
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.appendLog(text)
            }
        }
    }

    private func showAlert(message: String, informativeText: String = "") {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = informativeText
        alert.runModal()
    }
}

private struct ValidationError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? {
        message
    }
}
