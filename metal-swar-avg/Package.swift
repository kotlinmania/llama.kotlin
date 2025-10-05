// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalSwarAvg",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MetalSwarAvg", targets: ["MetalSwarAvg"]) 
    ],
    targets: [
        .executableTarget(
            name: "MetalSwarAvg",
            dependencies: [],
            path: "Sources"
        )
    ]
)
