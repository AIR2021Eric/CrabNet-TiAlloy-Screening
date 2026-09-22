import os
import torch
import glob

weight_dir = "models/trained_models"
# Find all the elasticity weight files you extracted earlier
pth_files = glob.glob(os.path.join(weight_dir, "elasticity_*.pth"))

print(f"Found {len(pth_files)} weight files to patch.")

for filepath in pth_files:
    print(f"Patching {os.path.basename(filepath)}...")
    
    # Load the checkpoint (weights_only=False is required for these older checkpoint formats)
    network = torch.load(filepath, map_location="cpu", weights_only=False)
    
    weights = network["weights"]
    new_weights = {}
    
    for k, v in weights.items():
        # Map the original anthony-wang layer names to the new sparks-baird layer names
        new_k = k.replace("encoder.pe.", "encoder.prevalence_encoder.")
        new_k = new_k.replace("encoder.ple.", "encoder.prevalence_log_encoder.")
        new_weights[new_k] = v
        
    network["weights"] = new_weights
    
    # Save it back, overwriting the old file with the corrected keys
    torch.save(network, filepath)

print("\nDone! All elasticity weights are now patched and compatible.")