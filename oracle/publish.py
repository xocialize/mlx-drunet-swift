"""Publish the converted DRUNet weights to mlx-community."""
import os, sys
from huggingface_hub import HfApi
API = HfApi(); HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "mlx-community/DRUNet-color-fp32"
W = os.path.join(HERE, "converted", "drunet_color", "model.safetensors")

CARD = """---
library_name: mlx
license: mit
license_link: https://github.com/cszn/DPIR/blob/master/LICENSE
base_model: cszn/DPIR
pipeline_tag: image-to-image
tags:
  - mlx
  - denoising
  - image-restoration
  - drunet
  - dpir
---

# mlx-community/DRUNet-color-fp32

[DRUNet](https://github.com/cszn/DPIR) colour denoising, converted to **Apple MLX** for
Apple-Silicon inference via [`mlx-drunet-swift`](https://github.com/xocialize/mlx-drunet-swift).

Zhang et al., *Plug-and-Play Image Restoration with Deep Denoiser Prior*, **IEEE TPAMI 2021**.
**32,640,960 parameters** (130.6 MB).

## 🔑 The point of this model: the noise level is an INPUT

DRUNet takes σ as a **constant 4th channel of `σ/255`**, so **one set of weights spans a continuous
range** — a denoise-strength dial rather than a fixed level baked into the checkpoint. Verified on
these weights:

| σ | 0 | 15 | 25 | 50 |
|---|---|---|---|---|
| mean \\|out − in\\| | 0.0055 | 0.018 | 0.036 | 0.130 |

Monotonic — the conditioning is real. A port that silently dropped the 4th channel would still
produce plausible-looking denoised output, which is why it is worth checking rather than assuming.

```swift
import DRUNetMLXCore

let model = DRUNet()
try model.loadWeights(from: weightsURL)
let clean = model.denoise(imageNHWC, sigma: 25)   // NHWC RGB in [0,1]; sigma on the 0…255 scale
```

Or as an MLXEngine `imageRestore` ModelPackage (`MLXDRUNet.DRUNetRestorePackage`), which honours
`ImageRestoreRequest.strength` — the optional parameter contract 1.30.0 added for exactly this case.

## ⚠️ Honest positioning: a UX play, not a quality play

DRUNet is **Gaussian-only in its paper** and does not report SIDD. It is **not** a NAFNet or
Restormer replacement on denoise quality. Its value is the continuous dial, which those models
architecturally cannot offer.

Note also that σ nominally means "the AWGN standard deviation of the input". Treating it as a
user-facing strength slider is a product decision; picking σ *automatically* for a real photograph
needs a measured per-ISO noise characterisation, not a guess.

## Conversion

MLX **NHWC**. Two different 4-D transposes, and **they cannot be told apart by shape**:

| | PyTorch | MLX | transpose |
|---|---|---|---|
| `Conv2d` | `(O, I, kH, kW)` | `(O, kH, kW, I)` | `(0,2,3,1)` |
| `ConvTranspose2d` | **`(I, O, kH, kW)`** | `(O, kH, kW, I)` | **`(1,2,3,0)`** |

`m_down3.4.weight` (strideconv 256→512) and `m_up3.0.weight` (transposed 512→256) are **both
`(512, 256, 2, 2)`** with opposite meanings. The wrong transpose loads clean and is silently wrong.
Only the key discriminates: exactly `m_up{1,2,3}.0.weight` are transposed.

## Parity

Gated against the PyTorch oracle on the CPU stream, fp32, relative error:

- **key contract** — 64 tensors / 32,640,960 params / 0 missing / 0 unused, strict load
- **primitives** — 5/5; the **transposed conv is bit-identical (0.00e+00)**
- **σ sweep** — 4/4 at cosine **1.00000000**, with the monotonic response asserted

Weights: MIT, published first-party by the author in the
[`cszn/KAIR` v1.0 release](https://github.com/cszn/KAIR/releases/tag/v1.0). Port code: MIT.
(Third-party mirrors tagged `bsd-3-clause` carry the DeepInverse *library's* licence, not this
model's — this build does not use them.)
"""

def main():
    if "--dry-run" in sys.argv:
        print(CARD); return
    assert os.path.exists(W), W
    print(f"[publish] {REPO} ({os.path.getsize(W)/1e6:.2f} MB) …")
    API.create_repo(REPO, repo_type="model", exist_ok=True)
    API.upload_file(path_or_fileobj=W, path_in_repo="model.safetensors", repo_id=REPO)
    API.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=REPO)
    print(f"[publish]   ok → https://huggingface.co/{REPO}")

main()
