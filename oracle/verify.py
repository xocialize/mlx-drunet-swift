"""P5 Stage 0: does the first-party DRUNet checkpoint load, and what is the real shape?"""
import sys, torch
sys.path.insert(0, "upstream")
from models.network_unet import UNetRes

torch.set_grad_enabled(False)
sd = torch.load("weights/drunet_color.pth", map_location="cpu", weights_only=False)
print("=== checkpoint ===")
print(f"  type={type(sd).__name__}  tensors={len(sd) if hasattr(sd,'__len__') else '?'}")
print(f"  first keys: {list(sd.keys())[:4]}")

# The released config, verbatim from main_dpir_denoising.py:93
m = UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4,
            act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
missing, unexpected = m.load_state_dict(sd, strict=False)
n = sum(p.numel() for p in m.parameters())
nb = sum(p.numel()*p.element_size() for p in m.parameters())
try:
    m.load_state_dict(sd, strict=True); strict = "✅ CLEAN"
except Exception as e:
    strict = f"❌ {str(e)[:100]}"
print(f"\n=== model ===")
print(f"  params      : {n:,}  = {nb/1e6:.2f} MB fp32 / {nb/2e6:.2f} MB fp16")
print(f"  strict load : {strict}  (missing {len(missing)}, unexpected {len(unexpected)})")

m.eval()
# 4-channel input: RGB + a CONSTANT noise-level plane (sigma/255). That plane IS the strength dial.
img = torch.rand(1, 3, 128, 128)
print(f"\n=== the noise-level channel is a continuous dial ===")
for sigma in (0, 15, 25, 50):
    x = torch.cat((img, torch.full((1,1,128,128), sigma/255.)), dim=1)
    y = m(x)
    print(f"  sigma={sigma:3d} -> out {tuple(y.shape)}  mean|y-img|={float((y-img).abs().mean()):.6f}")
print("\n  (a rising delta with sigma = the model is genuinely conditioned on the map,")
print("   not ignoring it — which is the whole premise of this row)")
