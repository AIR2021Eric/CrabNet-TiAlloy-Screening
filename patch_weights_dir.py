"""patch_weights_dir.py <dir>
Rename original-repo encoder keys to sparks-baird fork keys, in place."""
import sys, os, glob, torch

d = sys.argv[1]
for path in glob.glob(os.path.join(d, "*.pth")):
    try:
        net = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                      # skips non-weight .pth files
        print("skip ", os.path.basename(path), "->", type(e).__name__)
        continue
    if not isinstance(net, dict) or "weights" not in net:
        print("skip ", os.path.basename(path))
        continue
    new_w, changed = {}, False
    for k, v in net["weights"].items():
        nk = (k.replace("encoder.pe.", "encoder.prevalence_encoder.")
                .replace("encoder.ple.", "encoder.prevalence_log_encoder."))
        if nk != k:
            changed = True
        new_w[nk] = v
    if changed:
        net["weights"] = new_w
        torch.save(net, path)
        print("patched  ", os.path.basename(path))
    else:
        print("already  ", os.path.basename(path))