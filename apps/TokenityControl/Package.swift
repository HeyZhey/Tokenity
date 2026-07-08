// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "TokenityControl",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "TokenityControl", targets: ["TokenityControl"])
    ],
    targets: [
        .executableTarget(name: "TokenityControl"),
        .testTarget(name: "TokenityControlTests", dependencies: ["TokenityControl"])
    ]
)
