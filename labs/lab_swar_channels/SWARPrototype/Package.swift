// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "SWARPrototype",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "swar-prototype", targets: ["SWARPrototype"])
    ],
    targets: [
        .executableTarget(
            name: "SWARPrototype",
            linkerSettings: [
                .linkedFramework("Metal")
            ]
        )
    ]
)
