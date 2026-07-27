import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
import Hub
import DRUNetMLXCore

/// Errors at the DRUNet package boundary.
public enum DRUNetPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsMissing(String)
}

/// An MLXEngine `imageRestore` package over **DRUNet** — denoising with a **continuous strength
/// dial**, the first backer on this capability to honour `ImageRestoreRequest.strength`.
///
/// NAFNet, FFTformer and Restormer each bake their level into the checkpoint: you pick a model, not
/// a level. DRUNet takes the noise level as a genuine *input* — a constant 4th channel of `σ/255` —
/// so one set of weights spans a continuous range. Contract 1.30.0 grew the optional `strength`
/// parameter precisely for this, and this package reports `appliedStrength` so a caller can tell a
/// dial that worked from a backer that has none.
///
/// ⚠️ **Positioning, from the port queue: this is a UX play, not a quality play.** DRUNet is
/// Gaussian-only in its paper and does not report SIDD — it is not a NAFNet replacement on quality.
/// It is the slider.
@InferenceActor
public final class DRUNetRestorePackage: ModelPackage {
    public typealias Configuration = DRUNetConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // cszn/DPIR is MIT; the weights are published first-party by the same author in the
            // cszn/KAIR v1.0 release, also MIT. (The third-party deepinv mirror's bsd-3-clause tag
            // is the DeepInverse library's licence, not this model's — not used here.)
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "cszn/DPIR", revision: "master", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14) — ✅ MEASURED through the REAL `MLXServeEngine`
                // via `MLXEngineTestKit.ValidationHarness` (`swift run drunet-validate`), process
                // `phys_footprint`, floor read post-load/pre-run:
                //
                //   [drunet-color] SPLIT floor=0.16GB peak=6.71GB act=6.56GB retain=2.49GB
                //                  engine=0.20GB reserve=6.50GB load=0.0s run=1.6s   @1920x1080
                //
                // Declared with margin: resident 200 MB (floor 155.5 MB), activation 7.0 GB
                // (measured 6.56 GB). The estimate of 6.5 GB was very slightly UNDER — nudged up,
                // since under-declaring is the direction that falsely admits on tight Macs.
                //
                // ⚠️ Runs the FULL FRAME — no tiling. A plain conv U-Net has no channel blow-up, so
                // this is viable at 1080p where FFTformer's 6x expansion was not. But activation is
                // therefore LINEAR in resolution, unlike the tiled siblings whose peak is flat: a 4K
                // frame will cost roughly 4x this. Tile before claiming 4K support.
                //
                // NOTE `retain=2.49GB` — large, like CIDNet's (also untiled). The live model holds
                // full-frame intermediates; it frees on evict, so transient not resident.
                footprints: [
                    QuantFootprint(quant: .fp32,
                                   residentBytes: 200_000_000,
                                   peakActivationBytes: 7_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageRestoreContract.descriptor(
                    name: "drunet-denoise",
                    summary: "DRUNet denoising with a continuous strength dial: the noise level "
                        + "is a model input, so one checkpoint spans the range.",
                    supportsStrength: true
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: DRUNet?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }

        let url: URL
        if let explicit = configuration.weightsURL {
            guard FileManager.default.fileExists(atPath: explicit.path) else {
                throw DRUNetPackageError.weightsMissing(explicit.path)
            }
            url = explicit
        } else {
            // Since contract 1.24 the engine materializes declared `weightSources` before load().
            // This snapshot is the defensive path — it finds everything already present in the
            // normal flow, and still works for a standalone (engine-less) consumer of the package.
            let repo = configuration.variant.repo
            let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
            let dir = try await hub.snapshot(from: Hub.Repo(id: repo),
                                             matching: ["model.safetensors"]) { progress, speed in
                WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
            }
            url = dir.appendingPathComponent("model.safetensors")
        }

        // The variant picks the LayerNorm kind, which determines the KEY SET — the denoising
        // checkpoints are BiasFree and carry no bias vectors at all (406 tensors vs 494). A
        // mismatch fails the strict load with 88 missing keys rather than loading something wrong.
        let net = DRUNet(configuration.variant.coreConfiguration)
        try net.loadWeights(from: url)
        model = net
    }

    public func unload() async {
        model = nil
        MLX.Memory.clearCache()   // drop the retained MLX pool so eviction frees RSS, not just refs
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the FIRST act of run(), before notLoaded validation.
        // Mid-run cadence: like NAFNet this is ONE monolithic full-frame eval with no iterative
        // loop, so the honest seams are pre-forward (post-decode) and post-forward (pre-encode).
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .imageRestore,
              let req = request as? ImageRestoreRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let pb = try Self.decodeToPixelBuffer(req.image)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let x = rgbNHWC(from: ensureBGRA(pb), width: w, height: h) else {
            throw DRUNetPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h))")
        }

        // The strength dial. `nil` means the caller expressed no preference, so the package's own
        // default applies — deliberately the paper's mildest level, so "just clean it up" cannot
        // smear detail on an already-clean image.
        let sigma = configuration.sigma(for: req.strength)
        let applied = req.strength.map { min(max($0, 0), 1) } ?? (sigma / configuration.maxSigma)

        // Pre-forward checkpoint: last seam before committing to the eval.
        try Task.checkCancellation()
        let prof = MLXProfiler.shared
        prof.beginRun("drunet imageRestore \(configuration.variant.rawValue) \(w)x\(h) sigma=\(sigma)")
        // No tiling: DRUNet is a plain conv U-Net with no channel blow-up at full resolution, so
        // unlike FFTformer and Restormer it runs the frame whole. Confirm against the measured
        // footprint before assuming that holds at 4K.
        let restoredNHWC = prof.region("denoise", "forward") {
            model.denoise(x, sigma: sigma)
        }
        prof.endRun(denominators: ["image": 1])
        let outPB = pixelBuffer(fromRGBNHWC: restoredNHWC, width: w, height: h)
        guard let outPB else { throw DRUNetPackageError.imageEncodeFailed }

        // Post-forward checkpoint: between materialization and output encode.
        try Task.checkCancellation()
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw DRUNetPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw DRUNetPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageRestoreResponse(image: outImage, appliedStrength: applied)
    }

    // MARK: - Image codec
    //
    // Same shape as the sibling image packages. Duplicated rather than shared: each `-swift`
    // package stays independently buildable, and the codec is the package's own boundary.

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DRUNetPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw DRUNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw DRUNetPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no decode.
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw DRUNetPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw DRUNetPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw DRUNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw DRUNetPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes.
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension DRUNetRestorePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(DRUNetRestorePackage.self)
    }
}
