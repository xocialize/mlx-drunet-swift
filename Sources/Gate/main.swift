//
//  main.swift — DRUNet parity gates against the PyTorch oracle.
//

import Foundation
import DRUNetMLXCore
import MLX
import MLXNN

private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()
func fail(_ m: String) -> Never { _ = _unbuffered; print("❌ \(m)"); exit(1) }

func loadedModel(_ p: String) -> DRUNet {
    let m = DRUNet()
    do { try m.loadWeights(from: URL(fileURLWithPath: p)) } catch { fail("weight load: \(error)") }
    return m
}
func g(_ d: String, _ n: String) -> MLXArray {
    do { return try loadNPY("\(d)/\(n).npy") } catch { fail("golden \(n): \(error)") }
}

func gateS0(_ w: String) {
    _ = _unbuffered
    print("=== S0 · key contract ===\n")
    let model = DRUNet()
    var keys: [String: [Int]] = [:]; var total = 0
    for (k, v) in model.parameters().flattened() { keys[k] = v.shape; total += v.size }
    print("Swift : \(keys.count) tensors, \(total) params")
    guard let loaded = try? MLX.loadArrays(url: URL(fileURLWithPath: w)) else { fail("load \(w)") }
    print("Ckpt  : \(loaded.count) tensors, \(loaded.values.reduce(0) { $0 + $1.size }) params\n")
    let sk = Set(keys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted(), unused = ck.subtracting(sk).sorted()
    if !missing.isEmpty { print("MISSING (\(missing.count)):"); missing.prefix(10).forEach { print("   \($0) \(keys[$0]!)") } }
    if !unused.isEmpty { print("UNUSED (\(unused.count)):"); unused.prefix(10).forEach { print("   \($0) \(loaded[$0]!.shape)") } }
    var mis: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) where keys[k]! != loaded[k]!.shape { mis.append((k, keys[k]!, loaded[k]!.shape)) }
    if !mis.isEmpty {
        print("SHAPE MISMATCH (\(mis.count)):")
        for (k, a, b) in mis.prefix(10) { print("   \(k)\n     swift \(a) vs ckpt \(b)") }
    }
    guard missing.isEmpty, unused.isEmpty, mis.isEmpty else { fail("S0 FAILED") }
    do { try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all) }
    catch { fail("S0 FAILED at update: \(error)") }
    print("✅ S0 PASSED — \(keys.count) tensors, \(total) params, strict update clean.")
}

func gateS1(_ d: String, _ w: String) -> Bool {
    print("=== S1 · primitives ===\n")
    let r = GateReport("S1")
    let m = loadedModel(w)
    guard let rb = m.down1[0] as? ResBlock,
          let dn = m.down1[4] as? Conv2d,
          let up = m.up3[0] as? ConvTransposed2d else { fail("stage layout unexpected") }

    // Tolerance 2e-6 across the primitives. Measured spread on this model is 1.6e-07 (head) to
    // 1.0e-06 (tail) — 1e-6 sits ON the fp32 noise floor for convs accumulating over 64+ channels,
    // so it produces coin-flip failures rather than signal. The gate loses nothing by moving to
    // 2e-6, because the hazard it actually guards is not rounding: it is the transposed-conv weight
    // layout below, and that lands at EXACTLY 0.00e+00. Discrimination on the real risk is total.
    r.check("resblock", toNCHW(rb(toNHWC(g(d, "resblock_in")))), g(d, "resblock_out"), tol: 2e-6)
    r.check("strideconv_down", toNCHW(dn(toNHWC(g(d, "down_in")))), g(d, "down_out"), tol: 2e-6)
    // The transposed conv is the one that needs a DIFFERENT weight transpose from every other
    // conv — PyTorch stores it (I,O,k,k). `m_down3.4` and `m_up3.0` are both (512,256,2,2) with
    // opposite meanings, so a swap would load clean and be silently wrong. This gate is what
    // catches that.
    r.check("convtranspose_up", toNCHW(up(toNHWC(g(d, "up_in")))), g(d, "up_out"), tol: 2e-6)
    r.check("head", toNCHW(m.head(toNHWC(g(d, "head_in")))), g(d, "head_out"), tol: 2e-6)
    r.check("tail", toNCHW(m.tail(toNHWC(g(d, "tail_in")))), g(d, "tail_out"), tol: 2e-6)
    return r.summarize()
}

/// The σ sweep — the row's premise. A port that ignored the 4th channel would still denoise
/// plausibly, so each level is gated separately AND the monotonic response is asserted.
func gateS2(_ d: String, _ w: String) -> Bool {
    print("=== S2 · full model across the noise dial ===\n")
    let r = GateReport("S2")
    let m = loadedModel(w)
    var deltas: [(Int, Float)] = []
    let rgb = toNHWC(g(d, "full_rgb_in"))
    for sigma in [0, 15, 25, 50] {
        let x = toNHWC(g(d, "full_sigma\(sigma)_in"))
        let out = m(x); eval(out)
        r.check("sigma\(sigma)", toNCHW(out), g(d, "full_sigma\(sigma)_out"), tol: 1e-4)
        deltas.append((sigma, MLX.mean(MLX.abs(out - rgb)).item(Float.self)))
    }
    print("\n  response to the dial (mean |out − in|):")
    for (s, v) in deltas { print(String(format: "     sigma=%3d  %.6f", s, v)) }
    let monotonic = zip(deltas, deltas.dropFirst()).allSatisfy { $0.1 < $1.1 }
    print(monotonic
          ? "  ✅ monotonic — the model is genuinely conditioned on the noise plane"
          : "  ❌ NOT monotonic — the 4th channel may be mis-wired")
    return r.summarize() && monotonic
}

let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else {
    print("usage: drunet-gate --s0 <weights> | --s1|--s2|--all <goldens> <weights>"); exit(2)
}
Device.setDefault(device: .cpu)
switch mode {
case "--s0":
    guard args.count >= 2 else { fail("--s0 needs weights") }
    gateS0(args[1])
case "--s1", "--s2", "--all":
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    let (d, w) = (args[1], args[2])
    var ok = true
    if mode == "--s1" || mode == "--all" { ok = gateS1(d, w) && ok; print("") }
    if mode == "--s2" || mode == "--all" { ok = gateS2(d, w) && ok }
    if !ok { exit(1) }
default: fail("unknown mode \(mode)")
}
