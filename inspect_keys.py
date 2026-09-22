"""inspect_keys.py <path-to-checkpoint.pth>
Reports whether a CrabNet checkpoint uses old-style (original repo) or
new-style (sparks-baird fork) encoder key names, i.e. whether it is patched."""
import sys, torch

path = sys.argv[1] if len(sys.argv) > 1 else "models/trained_models/elasticity_log10(K_VRH)0.pth"
net = torch.load(path, map_location="cpu", weights_only=False)
sd = net["weights"] if isinstance(net, dict) and "weights" in net else net

old = [k for k in sd if k.startswith(("encoder.pe.", "encoder.ple."))]
new = [k for k in sd if k.startswith(("encoder.prevalence_encoder", "encoder.prevalence_log_encoder"))]

print("file:", path)
print("old-style keys (encoder.pe.* / encoder.ple.*):", old[:4] if old else "[] none")
print("new-style keys (encoder.prevalence_*):        ", new[:4] if new else "[] none")

if old and not new:
    print("VERDICT: UNPATCHED (original-repo format) -> run patch_weights_dir.py on its folder")
elif new and not old:
    print("VERDICT: PATCHED (sparks-baird-fork compatible)")
elif old and new:
    print("VERDICT: MIXED key families - inspect manually")
else:
    print("VERDICT: neither key family found - not a CrabNet elasticity checkpoint?")