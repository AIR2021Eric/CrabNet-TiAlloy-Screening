import torch

ckpt = torch.load(
    "models/trained_models/aflow__ael_bulk_modulus_vrh.pth",
    map_location="cpu",
    weights_only=False,   # these are 2020-era checkpoints; may contain non-tensor objects
)

print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys()))
    for k, v in ckpt.items():
        print(f"  {k}: {type(v)}", 
              f"shape/len={getattr(v, 'shape', len(v) if hasattr(v,'__len__') else '?')}" 
              if hasattr(v, 'shape') or hasattr(v, '__len__') else "")
else:
    print("Not a dict — it's a full pickled object:", ckpt)