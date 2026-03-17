// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NStarGUI",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .executable(
            name: "nstar-gui",
            targets: ["NStarGUI"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "NStarGUI",
            path: "Sources/NStarGUI"
        ),
    ]
)
