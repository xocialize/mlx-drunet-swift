"""P5 weight conversion: drunet_*.pth -> safetensors in MLX NHWC layout.

TWO DIFFERENT 4-D transposes, and telling them apart is the whole job.

  regular Conv2d      PyTorch (O, I, kH, kW) -> MLX (O, kH, kW, I)   transpose (0,2,3,1)
  ConvTranspose2d     PyTorch (I, O, kH, kW) -> MLX (O, kH, kW, I)   transpose (1,2,3,0)

🔴 They cannot be distinguished by shape. `m_down3.4.weight` (a strideconv, 256->512) and
`m_up3.0.weight` (a transposed conv, 512->256) are **both (512, 256, 2, 2)** — identical shape,
opposite meaning. Apply the wrong one and you get a validly-shaped tensor that loads clean, passes
every structural check, and is silently wrong. The only reliable discriminator is the KEY: the three
transposed convs are exactly `m_up{1,2,3}.0.weight`.

Run:  .venv/bin/python convert.py [stem ...]
"""
import json, os, re, sys
import numpy as np, torch
from safetensors.numpy import save_file

STEMS = sys.argv[1:] or ["drunet_color"]
# The upsample head of each m_up Sequential — and nothing else — is a ConvTranspose2d.
TRANSPOSED = re.compile(r"^m_up[123]\.0\.weight$")

for stem in STEMS:
    src = f"weights/{stem}.pth"
    if not os.path.exists(src):
        print(f"[skip] {src}"); continue
    sd = torch.load(src, map_location="cpu", weights_only=False)
    sd = sd.get("params", sd) if isinstance(sd, dict) and "params" in sd else sd

    out_dir = os.path.join("converted", stem); os.makedirs(out_dir, exist_ok=True)
    conv = {}
    stats = {"conv": 0, "convtranspose": 0, "passthrough": 0}
    for k, v in sd.items():
        a = v.detach().cpu().numpy().astype(np.float32)
        if a.ndim == 4:
            if TRANSPOSED.match(k):
                a = np.transpose(a, (1, 2, 3, 0)); stats["convtranspose"] += 1
            else:
                a = np.transpose(a, (0, 2, 3, 1)); stats["conv"] += 1
        else:
            stats["passthrough"] += 1
        conv[k] = np.ascontiguousarray(a)

    assert stats["convtranspose"] == 3, f"expected exactly 3 transposed convs, got {stats['convtranspose']}"
    total = sum(int(np.prod(v.shape)) for v in conv.values())
    print(f"=== {stem} ===")
    print(f"  tensors: {len(conv)}   conv {stats['conv']} · convtranspose {stats['convtranspose']} "
          f"· passthrough {stats['passthrough']}")
    print(f"  params : {total:,} ({total*4/1e6:.2f} MB fp32)")
    for k in ["m_down3.4.weight", "m_up3.0.weight"]:
        print(f"    {k:20s} {tuple(sd[k].shape)} -> {conv[k].shape}")

    meta = {"format": "pt", "source": f"cszn/KAIR v1.0 {stem}.pth", "license": "MIT",
            "layout": "MLX NHWC; conv (O,kH,kW,I); convtranspose (O,kH,kW,I) from PyTorch (I,O,k,k)",
            "params": str(total)}
    save_file(conv, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    json.dump({"stem": stem, "transforms": stats, "params": total},
              open(os.path.join(out_dir, "CONVERSION.json"), "w"), indent=2)
    print(f"  written: {out_dir}/model.safetensors "
          f"({os.path.getsize(os.path.join(out_dir,'model.safetensors'))/1e6:.2f} MB)\n")
