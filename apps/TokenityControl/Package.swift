// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "TokenityControl",
    platforms: [.macOS("26.2")],
    products: [
        .executable(name: "TokenityControl", targets: ["TokenityControl"])
    ],
    targets: [
        .executableTarget(
            name: "TokenityControl",
            resources: [.process("Resources")]
        ),
        .testTarget(name: "TokenityControlTests", dependencies: ["TokenityControl"])
    ]
)
