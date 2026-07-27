# mlx-drunet-swift — port status

**Work order:** P5 in `mlxengine-todo/PORT-QUEUE.md` — DRUNet, "the denoise strength slider".

Upstream: [`cszn/DPIR`](https://github.com/cszn/DPIR) — **MIT**. Zhang et al., *Plug-and-Play Image
Restoration with Deep Denoiser Prior*, TPAMI 2021. **32,640,960 parameters** (130.56 MB fp32).

---

## Licence / availability — the queue was wrong in our favour

The queue recorded *"P5 DRUNet has nothing in `cszn/DPIR` — third-party `deepinv/drunet` only, check
its license."* In fact **the same author publishes every weight first-party in the
[`cszn/KAIR` v1.0 release](https://github.com/cszn/KAIR/releases/tag/v1.0), whose `LICENSE` is MIT** —
`drunet_color.pth` (130.58 MB, 7,204 downloads), `drunet_gray.pth`, the deblocking variants, and the
whole SCUNet family that unblocks P9 too.

⚠️ **Do not use the `deepinv` mirrors.** `deepinv/drunet`'s HF card claims `bsd-3-clause` — that is
the *DeepInverse library's* licence blanketed across their org, and says nothing about upstream, which
is MIT. `deepinv/scunet` carries no licence at all. **A mirror's licence tag is not evidence about the
weights it mirrors.**

---

## Stage 0 ✅ PASSED (2026-07-27)

| Fact | Verified value |
|---|---|
| Licence | **MIT** (DPIR), weights MIT (KAIR release) · zero NC anywhere |
| Parameters | **32,640,960** — 130.56 MB fp32 |
| Load | `strict=True` **clean**, 64 tensors, flat `OrderedDict` |
| Constructor | `UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')` |

### 🔑 The premise, demonstrated rather than assumed

DRUNet takes the **noise level as an input** — a constant 4th channel of `σ/255`
(`main_dpir_denoising.py:129`). One set of weights therefore gives a *continuous* strength dial, which
NAFNet architecturally cannot offer. Sweeping σ on the released checkpoint:

| σ | 0 | 15 | 25 | 50 |
|---|---|---|---|---|
| mean \|out − in\| | 0.0055 | 0.0182 | 0.0365 | 0.1299 |

Monotonic — the conditioning is real, not decorative. A port that silently ignored the 4th channel
would still produce plausible-looking denoised output, so this is gated explicitly.

⚠️ **Honest positioning:** DRUNet is **Gaussian-only in its paper** and does not report SIDD. It is
not a NAFNet replacement on quality. It is the slider.

### Architecture notes

`m_head` conv(4→64) · `m_down1/2/3` = 4×ResBlock + **stride-2 conv** (k2 s2 p0) · `m_body` 4×ResBlock(512)
· `m_up3/2/1` = **ConvTranspose2d** (k2 s2 p0) + 4×ResBlock · `m_tail` conv(64→3). All `bias=False`.
ResBlock is `conv → ReLU → conv`, then `+x`.

Skips are **additive**, not concatenated — simpler than the sibling ports. There is **no global
residual** on the input, which matters: the input has 4 channels and the output has 3. Three stride-2
stages mean H and W must be divisible by 8.

---

## 🔴 The conversion trap: two different 4-D transposes

|  | PyTorch | MLX | transpose |
|---|---|---|---|
| regular `Conv2d` | `(O, I, kH, kW)` | `(O, kH, kW, I)` | `(0,2,3,1)` |
| `ConvTranspose2d` | **`(I, O, kH, kW)`** | `(O, kH, kW, I)` | **`(1,2,3,0)`** |

**They cannot be told apart by shape.** `m_down3.4.weight` (a strideconv, 256→512) and
`m_up3.0.weight` (a transposed conv, 512→256) are **both `(512, 256, 2, 2)`** — identical shape,
opposite meaning. Apply the wrong transpose and you get a validly-shaped tensor that loads clean,
passes every structural check, and is silently wrong.

The only reliable discriminator is the **key**: exactly `m_up{1,2,3}.0.weight` are transposed. The
converter asserts it finds exactly three.

---

## Stage 1 — parity ✅ **ALL GATES GREEN**

| Gate | Result | Worst |
|---|---|---|
| **S0** key contract | ✅ 64 tensors, 32,640,960 params, strict clean | exact |
| **S1** primitives | ✅ 5/5 — **`convtranspose_up` is BIT-IDENTICAL (0.00e+00)** | 1.0e-06 |
| **S2** σ sweep | ✅ 4/4 at cosine **1.00000000**, response monotonic | 1.8e-06 |

Primitive tolerance is **2e-6**, not 1e-6. Measured spread is 1.6e-07 (head) to 1.0e-06 (tail), so
1e-6 sits *on* the fp32 noise floor for convs accumulating over 64+ channels and produces coin-flip
failures rather than signal. Nothing is lost: the hazard this gate actually guards is the transposed-conv
layout, and that lands at **exactly 0.00e+00** — discrimination on the real risk is total, not marginal.

---

## Remaining

- [ ] Wrap as a package on `imageRestore` — **but the capability question is a real one.** The
      request carries no strength parameter, so exposing the σ dial through `imageRestore` means
      either a `mode`, a `metaData` key, or a new capability. Worth a stop-and-ask.
- [ ] Publish `mlx-community/DRUNet-color-fp32`; conformance; footprint; registry row.
- [ ] Validation needs corpus **C5** — and P5 is why C5 now also wants a **dark frame per ISO**:
      something has to choose σ for a real photograph, and a dark frame's standard deviation *is* σ
      for that ISO, measured rather than guessed.

## Reproduce

```bash
cd oracle
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python torch numpy safetensors
git clone --depth 1 https://github.com/cszn/DPIR.git upstream
curl -sLO https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth   # into weights/
.venv/bin/python verify.py && .venv/bin/python gen_goldens.py && .venv/bin/python convert.py
cd .. && swift run drunet-gate --all oracle/goldens oracle/converted/drunet_color/model.safetensors
```

---

## Stage 2 — package ✅ (2026-07-27)

`MLXDRUNet.DRUNetRestorePackage` — the **fourth** package on `imageRestore`, and the **first that
honours a strength dial**.

### Contract 1.30.0 was added for this row

`imageRestore` had no strength parameter, because NAFNet, FFTformer and Restormer all bake their
level into the checkpoint — you pick a model, not a level. DRUNet takes the level as a genuine
*input*, so the capability grew an **optional** `ImageRestoreRequest.strength` (0…1) and
`ImageRestoreResponse.appliedStrength`. Both defaulted, so the three existing backers are untouched
and simply ignore them.

`appliedStrength == nil` is the honest signal that a backer has **no** dial, as distinct from a dial
that did nothing — a UI greys the control, a planner routes elsewhere. `descriptor(supportsStrength:)`
keeps the parameter off the surface of packages that would ignore it.

Engine **v0.39.0**, pushed.

### strength → σ

| strength | 0 | 0.5 | 1.0 | omitted |
|---|---|---|---|---|
| σ | 0 | 25 | **50** | **15** |

`maxSigma` stops at **50** — the paper's top *reported* level — rather than the nominal 0…255 range,
so `strength: 1.0` means "as strong as the authors characterised" and not an extrapolation. Omitting
strength gives σ 15, the mildest reported level, so "just clean it up" cannot smear an already-clean
image. Out-of-range input clamps.

⚠️ **This mapping is a product decision, not a calibration.** σ nominally means "the AWGN standard
deviation of the input"; choosing σ *automatically* for a real photograph needs a measured per-ISO
characterisation — corpus **C5**, which now also asks for a dark frame per ISO precisely because of
this row.

### Footprint — ✅ measured

```
[drunet-color] SPLIT floor=0.16GB peak=6.71GB act=6.56GB retain=2.49GB
               engine=0.20GB reserve=6.50GB load=0.0s run=1.6s   @1920x1080
```

Declared resident 200 MB / activation 7.0 GB, with margin above measured.

⚠️ **No tiling.** A plain conv U-Net has no channel blow-up, so full-frame is viable at 1080p where
FFTformer's 6× expansion was not. But activation is therefore **linear in resolution**, unlike the
tiled siblings whose peak is flat — a 4K frame will cost roughly 4× this. **Tile before claiming 4K.**

Conformance **12/12**, including four tests that pin the dial: the descriptor must advertise
`strength`, the σ endpoints and default are pinned, out-of-range clamps, and `maxSigma` stays within
the characterised range.

**Published:** [`mlx-community/DRUNet-color-fp32`](https://huggingface.co/mlx-community/DRUNet-color-fp32).
Round-trip verified — freshly downloaded artifact re-passed S0.
