//
//  DRUNet.swift
//  mlx-drunet-swift / DRUNetMLXCore
//
//  Role: MLX-Swift port of DRUNet (DPIR) — denoising with a **continuous strength dial**.
//
//  Upstream: https://github.com/cszn/DPIR — **MIT**. Weights are published first-party by the same
//            author in the `cszn/KAIR` v1.0 release, also MIT.
//  Paper:    Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", TPAMI 2021.
//            32,640,960 parameters.
//
//  🔑 Why this model earns its place next to NAFNet: it takes the **noise level as an input**, as a
//  constant 4th channel of `σ/255`. One set of weights therefore gives a *continuous* denoise-strength
//  dial, which NAFNet architecturally cannot offer — you would need a separate checkpoint per level.
//  Verified against the released weights: sweeping σ 0→15→25→50 moves the output monotonically
//  (mean |Δ| 0.0055 → 0.018 → 0.036 → 0.130), so the conditioning is real, not decorative.
//
//  ⚠️ Honest positioning (from the port queue): DRUNet is **Gaussian-only in its paper** and does not
//  report SIDD. It is not a NAFNet replacement on quality. It is the slider.
//
//  Conventions: NHWC; module keys mirror the upstream state dict exactly.
//

import Foundation
import MLX
import MLXNN

/// `conv → ReLU → conv`, then add the input. Upstream `ResBlock(mode: 'CRC')`, `bias=False`.
///
/// The nested `res` property matches the upstream key path (`m_down1.0.res.0.weight`), where index 0
/// and index 2 are the two convs — index 1 is the parameter-free ReLU.
public final class ResBlock: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "res") public var res: [Module]     // [Conv2d, ReLU, Conv2d]

    public init(_ channels: Int) {
        self._res.wrappedValue = [
            Conv2d(inputChannels: channels, outputChannels: channels,
                   kernelSize: 3, padding: 1, bias: false),
            Identity(),                                   // the ReLU — no parameters, holds the index
            Conv2d(inputChannels: channels, outputChannels: channels,
                   kernelSize: 3, padding: 1, bias: false),
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let c0 = res[0] as? Conv2d, let c2 = res[2] as? Conv2d else { return x }
        return x + c2(relu(c0(x)))
    }
}

public final class DRUNet: Module, @unchecked Sendable {

    public struct Configuration: Sendable {
        /// 3 image channels **+ 1 noise-level plane**. The `+1` is the point of the model.
        public var inChannels = 4
        public var outChannels = 3
        public var nc = [64, 128, 256, 512]
        /// Residual blocks per scale.
        public var nb = 4
        /// The released `drunet_color.pth` uses every default (main_dpir_denoising.py:93).
        public init() {}

        /// Greyscale variant (`drunet_gray.pth`): 1 image channel + the same noise plane.
        public static var grayscale: Configuration {
            var c = Configuration(); c.inChannels = 2; c.outChannels = 1; return c
        }
    }

    @ModuleInfo(key: "m_head") public var head: Conv2d
    @ModuleInfo(key: "m_down1") public var down1: [Module]
    @ModuleInfo(key: "m_down2") public var down2: [Module]
    @ModuleInfo(key: "m_down3") public var down3: [Module]
    @ModuleInfo(key: "m_body") public var body: [ResBlock]
    @ModuleInfo(key: "m_up3") public var up3: [Module]
    @ModuleInfo(key: "m_up2") public var up2: [Module]
    @ModuleInfo(key: "m_up1") public var up1: [Module]
    @ModuleInfo(key: "m_tail") public var tail: Conv2d

    private let nb: Int

    public init(_ cfg: Configuration = Configuration()) {
        self.nb = cfg.nb
        let nc = cfg.nc

        // Each m_down is `nb × ResBlock` then a **stride-2 conv** (k=2, s=2, p=0) — upstream
        // `downsample_strideconv`, not a pool.
        func downStage(_ inCh: Int, _ outCh: Int) -> [Module] {
            (0 ..< cfg.nb).map { _ in ResBlock(inCh) as Module }
                + [Conv2d(inputChannels: inCh, outputChannels: outCh,
                          kernelSize: 2, stride: 2, padding: 0, bias: false)]
        }
        // Each m_up is a **transposed conv** (k=2, s=2, p=0) then `nb × ResBlock`.
        func upStage(_ inCh: Int, _ outCh: Int) -> [Module] {
            [ConvTransposed2d(inputChannels: inCh, outputChannels: outCh,
                              kernelSize: 2, stride: 2, padding: 0, bias: false) as Module]
                + (0 ..< cfg.nb).map { _ in ResBlock(outCh) as Module }
        }

        self._head.wrappedValue = Conv2d(inputChannels: cfg.inChannels, outputChannels: nc[0],
                                         kernelSize: 3, padding: 1, bias: false)
        self._down1.wrappedValue = downStage(nc[0], nc[1])
        self._down2.wrappedValue = downStage(nc[1], nc[2])
        self._down3.wrappedValue = downStage(nc[2], nc[3])
        self._body.wrappedValue = (0 ..< cfg.nb).map { _ in ResBlock(nc[3]) }
        self._up3.wrappedValue = upStage(nc[3], nc[2])
        self._up2.wrappedValue = upStage(nc[2], nc[1])
        self._up1.wrappedValue = upStage(nc[1], nc[0])
        self._tail.wrappedValue = Conv2d(inputChannels: nc[0], outputChannels: cfg.outChannels,
                                         kernelSize: 3, padding: 1, bias: false)
    }

    private func runStage(_ stage: [Module], _ x: MLXArray) -> MLXArray {
        stage.reduce(x) { acc, m in
            if let r = m as? ResBlock { return r(acc) }
            if let c = m as? Conv2d { return c(acc) }
            if let t = m as? ConvTransposed2d { return t(acc) }
            return acc
        }
    }

    /// Forward on an NHWC tensor of `inChannels` (RGB + the σ plane), H and W divisible by 8.
    ///
    /// Skips are **additive**, not concatenated — unusual, and simpler than the sibling ports. There
    /// is also no global residual on the input, which matters because the input has 4 channels and
    /// the output has 3.
    public func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let x1 = head(x0)
        let x2 = runStage(down1, x1)
        let x3 = runStage(down2, x2)
        let x4 = runStage(down3, x3)
        var x = body.reduce(x4) { $1($0) }
        x = runStage(up3, x + x4)
        x = runStage(up2, x + x3)
        x = runStage(up1, x + x2)
        return tail(x + x1)
    }

    /// Denoise an NHWC RGB image in [0,1] at a given noise level.
    ///
    /// - Parameter sigma: noise level on the conventional **0…255** scale (15 / 25 / 50 are the
    ///   levels the paper reports). Internally becomes a constant plane of `sigma/255` — that plane
    ///   IS the strength dial, so exposing `sigma` is exposing the slider.
    public func denoise(_ image: MLXArray, sigma: Float) -> MLXArray {
        let (h, w) = (image.dim(1), image.dim(2))
        let plane = MLXArray.full([image.dim(0), h, w, 1], values: MLXArray(sigma / 255.0))
        let (padded, size) = Self.padToMultiple(concatenated([image, plane], axis: -1), 8)
        let out = self(padded)
        return clip(out[0..., 0 ..< size.0, 0 ..< size.1, 0...], min: 0, max: 1)
    }

    /// Reflect-pad to a multiple of 8 (three stride-2 stages), bottom/right only.
    static func padToMultiple(_ x: MLXArray, _ m: Int) -> (MLXArray, (Int, Int)) {
        let (h, w) = (x.dim(1), x.dim(2))
        let ph = (m - h % m) % m, pw = (m - w % m) % m
        if ph == 0 && pw == 0 { return (x, (h, w)) }
        var out = x
        if ph > 0 {
            out = concatenated([out, out.take(MLXArray((1...ph).map { Int32(h - 1 - $0) }), axis: 1)], axis: 1)
        }
        if pw > 0 {
            out = concatenated([out, out.take(MLXArray((1...pw).map { Int32(w - 1 - $0) }), axis: 2)], axis: 2)
        }
        return (out, (h, w))
    }

    /// Loads converted safetensors weights under the strict verifier.
    public func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}
