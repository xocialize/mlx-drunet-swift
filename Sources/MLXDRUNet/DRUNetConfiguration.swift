import Foundation
import MLXToolKit
import DRUNetMLXCore

/// A DRUNet checkpoint.
public enum DRUNetVariant: String, Codable, Sendable, CaseIterable {
    /// Colour denoising — the default. `drunet_color.pth`, 4-channel input (RGB + σ plane).
    case color
    /// Greyscale denoising. `drunet_gray.pth`, 2-channel input (Y + σ plane).
    case gray

    public var repo: String {
        switch self {
        case .color: return "mlx-community/DRUNet-color-fp32"
        case .gray: return "mlx-community/DRUNet-gray-fp32"
        }
    }

    var coreConfiguration: DRUNet.Configuration {
        switch self {
        case .color: return DRUNet.Configuration()
        case .gray: return .grayscale
        }
    }

    /// 32.6 M params at fp32 = 130.6 MB. Small enough that a lower dtype buys little, and denoise
    /// is precision-sensitive at the low end where the work happens.
    public var quant: Quant { .fp32 }
}

/// Init-time configuration for `DRUNetRestorePackage` (C9).
public struct DRUNetConfiguration: PackageConfiguration, ModelStorable {
    public var variant: DRUNetVariant

    /// The σ (0…255 scale) that `strength == 1.0` maps to.
    ///
    /// The request's `strength` is a normalized 0…1 dial; the model wants a noise level on the
    /// conventional 0…255 scale. 50 is the paper's top reported level, so `strength: 1.0` means
    /// "as strong as the authors characterised" rather than an extrapolation into territory the
    /// checkpoint was never evaluated at.
    public var maxSigma: Float

    /// σ used when a request omits `strength`.
    ///
    /// 15 is the paper's mildest reported level — a conservative default for "just clean it up"
    /// that will not smear detail on an already-clean image.
    public var defaultSigma: Float

    public var modelsRootDirectory: URL?
    public var weightsURL: URL?

    public init(variant: DRUNetVariant = .color,
                maxSigma: Float = 50,
                defaultSigma: Float = 15,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.variant = variant
        self.maxSigma = maxSigma
        self.defaultSigma = defaultSigma
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }

    /// Maps a request's normalized `strength` onto the model's σ input.
    ///
    /// ⚠️ **This mapping is a product decision, not a calibration.** σ nominally means "the AWGN
    /// standard deviation of the input", and the honest way to pick it is to *measure* the noise —
    /// which needs a per-ISO characterisation from corpus **C5** (dark frames). Until that exists,
    /// treating σ as a user-facing strength dial is the defensible interim: the user sees the effect
    /// and decides, rather than the package guessing a number and being wrong silently.
    func sigma(for strength: Float?) -> Float {
        guard let strength else { return defaultSigma }
        return min(max(strength, 0), 1) * maxSigma
    }

    private enum CodingKeys: String, CodingKey {
        case variant, maxSigma, defaultSigma
    }
}

extension DRUNetConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

extension DRUNetConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: variant.repo, revision: nil,
                      matching: ["model.safetensors"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}
