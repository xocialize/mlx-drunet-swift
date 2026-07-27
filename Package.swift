// swift-tools-version: 6.2
import PackageDescription

// mlx-drunet-swift — DRUNet (DPIR) denoising with a continuous strength dial.
// Upstream: cszn/DPIR (MIT); weights first-party from the cszn/KAIR v1.0 release (MIT).
let package = Package(
    name: "mlx-drunet-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "DRUNetMLXCore", targets: ["DRUNetMLXCore"]),
        .executable(name: "drunet-gate", targets: ["DRUNetGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        .target(name: "DRUNetMLXCore", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .executableTarget(name: "DRUNetGate", dependencies: [
            "DRUNetMLXCore", .product(name: "MLX", package: "mlx-swift"),
        ], path: "Sources/Gate", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
