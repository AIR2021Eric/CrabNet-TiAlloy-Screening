import os
import torch
import glob

weight_dir = "models/trained_models"
pth_files = glob.glob(os.path.join(weight_dir, "*.pth"))

print(f"Found {len(pth_files)} weight files to check and patch.")

for filepath in pth_files:
    filename = os.path.basename(filepath)
    if "distutils" in filename: continue
        
    try:
        network = torch.load(filepath, map_location="cpu", weights_only=False)
        weights = network["weights"]
        new_weights = {}
        needs_patch = False
        
        for k, v in weights.items():
            if "encoder.pe." in k or "encoder.ple." in k: needs_patch = True
            new_k = k.replace("encoder.pe.", "encoder.prevalence_encoder.")
            new_k = new_k.replace("encoder.ple.", "encoder.prevalence_log_encoder.")
            new_weights[new_k] = v
            
        if needs_patch:
            network["weights"] = new_weights
            torch.save(network, filepath)
            print(f"  -> Patched {filename}")
        else:
            print(f"  -> {filename} is already compatible.")
    except Exception as e:
        print(f"  -> Error processing {filename}: {e}")

print("\nDone! All weights are now patched and compatible.")