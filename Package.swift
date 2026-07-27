// swift-tools-version: 6.2
import PackageDescription

// mlx-drunet-swift — DRUNet (DPIR) denoising with a continuous strength dial.
// Upstream: cszn/DPIR (MIT); weights first-party from the cszn/KAIR v1.0 release (MIT).
let package = Package(
    name: "mlx-drunet-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "DRUNetMLXCore", targets: ["DRUNetMLXCore"]),
        .library(name: "MLXDRUNet", targets: ["MLXDRUNet"]),
        .executable(name: "drunet-gate", targets: ["DRUNetGate"]),
        .executable(name: "drunet-validate", targets: ["DRUNetValidate"]),
    ],
    dependencies: [
        // 0.39.0 = contract 1.30.0, which added the optional `strength` on imageRestore FOR this
        // package — DRUNet is the case the capability could not previously express.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.39.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(name: "DRUNetMLXCore", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .target(name: "MLXDRUNet", dependencies: [
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            "DRUNetMLXCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "Hub", package: "swift-transformers"),
            .product(name: "MLXProfiling", package: "mlx-profiling"),
        ], swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(name: "DRUNetMLXTests", dependencies: [
            "DRUNetMLXCore", "MLXDRUNet",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
        ], resources: [.copy("Resources/goldens")]),
        .executableTarget(name: "DRUNetValidate", dependencies: [
            "MLXDRUNet",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            .product(name: "MLXEngineTestKit", package: "mlx-engine-swift"),
        ], path: "Sources/Validate", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "DRUNetGate", dependencies: [
            "DRUNetMLXCore", .product(name: "MLX", package: "mlx-swift"),
        ], path: "Sources/Gate", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
