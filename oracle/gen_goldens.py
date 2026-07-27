"""P5 oracle — per-sub-op goldens for the DRUNet Swift port.

fp32, CPU-torch, numpy-seeded, C-contiguous, PyTorch NCHW.

The noise-level channel gets its own coverage: it is the reason this row exists, and a port that
silently ignored it would still produce plausible-looking denoised output.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "upstream")
from models.network_unet import UNetRes

torch.set_grad_enabled(False)
OUT = "goldens"; os.makedirs(OUT, exist_ok=True)
sd = torch.load("weights/drunet_color.pth", map_location="cpu", weights_only=False)
m = UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4,
            act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
m.load_state_dict(sd, strict=True); m.eval()

manifest = []
def save(n, a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    np.save(f"{OUT}/{n}.npy", a)
    manifest.append(f"{n+'.npy':30s} {str(a.shape):24s} min={a.min():+.6f} max={a.max():+.6f}")
    print(f"  saved {n}.npy {a.shape}")
def dump(n, t): save(n, t.detach().cpu().numpy())
def seeded(s, *sh):
    return torch.from_numpy(np.random.default_rng(s).standard_normal(sh, dtype=np.float32))

print("\n=== 1. ResBlock (conv -> ReLU -> conv, then +x) ===")
xb = seeded(1101, 1, 64, 32, 32)
dump("resblock_in", xb); dump("resblock_out", m.m_down1[0](xb))

print("\n=== 2. strideconv downsample (k2 s2 p0) ===")
dump("down_in", xb); dump("down_out", m.m_down1[4](xb))     # index 4 = the downsample after 4 ResBlocks

print("\n=== 3. ConvTranspose2d upsample (k2 s2 p0) ===")
xu = seeded(1102, 1, 512, 16, 16)
dump("up_in", xu); dump("up_out", m.m_up3[0](xu))

print("\n=== 4. head / tail ===")
xh = seeded(1103, 1, 4, 32, 32)
dump("head_in", xh); dump("head_out", m.m_head(xh))
xt = seeded(1104, 1, 64, 32, 32)
dump("tail_in", xt); dump("tail_out", m.m_tail(xt))

print("\n=== 5. Full model across the NOISE DIAL — the row's whole premise ===")
g = np.random.default_rng(1200)
yy, xx = np.mgrid[0:128, 0:128].astype(np.float32) / 127.0
img = np.stack([0.5 + 0.3*np.sin(xx*9)*np.cos(yy*7),
                0.5 + 0.3*np.cos(xx*6 + yy*5),
                0.5 + 0.25*np.sin((xx+yy)*11)])[None]
img = np.clip(img + 0.06*g.standard_normal(img.shape, dtype=np.float32), 0, 1).astype(np.float32)
base = torch.from_numpy(np.ascontiguousarray(img))
dump("full_rgb_in", base)
for sigma in (0, 15, 25, 50):
    x = torch.cat((base, torch.full((1,1,128,128), sigma/255.)), dim=1)
    dump(f"full_sigma{sigma}_in", x)
    dump(f"full_sigma{sigma}_out", m(x))

with open(f"{OUT}/MANIFEST.txt","w") as f:
    f.write("DRUNet PyTorch goldens — fp32, CPU, NCHW, C-contiguous.\n")
    f.write("checkpoint: weights/drunet_color.pth (cszn/KAIR v1.0 release, MIT)\n")
    f.write("constructor: UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4,\n")
    f.write("             act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')\n")
    f.write("input: RGB [0,1] + a CONSTANT 4th plane = sigma/255. Size must be divisible by 8.\n\n")
    f.write("\n".join(manifest) + "\n")
print(f"\n✅ {len(manifest)} goldens")
