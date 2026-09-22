import os
import glob
import re
import pandas as pd
import numpy as np
import warnings
import json
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab

# Suppress PyTorch warnings for a clean terminal output
warnings.filterwarnings("ignore", category=UserWarning)

def initialize_model(cb):
    if cb.model is None:
        cb.model = SubCrab(
            compute_device=cb.compute_device, out_dims=cb.out_dims, d_model=cb.d_model,
            d_extend=0, N=cb.N, heads=cb.heads, pe_resolution=cb.pe_resolution,
            ple_resolution=cb.ple_resolution, emb_scaler=cb.emb_scaler,
            pos_scaler=cb.pos_scaler, pos_scaler_log=cb.pos_scaler_log,
            dim_feedforward=cb.dim_feedforward, dropout=cb.dropout,
        ).to(cb.compute_device)

def predict_property(model_name, weight_files, df_input):
    cb = CrabNet(mat_prop=model_name, verbose=False)
    initialize_model(cb)
    
    preds = []
    for wf in weight_files:
        try:
            cb.load_network(os.path.basename(wf))
            pred = cb.predict(df_input)
            if isinstance(pred, tuple): pred = pred[0]
            preds.append(pred[0])
        except Exception:
            pass
            
    return np.mean(preds) if preds else None

def main():
    # 1. DEFINE INPUT
    formula = "Ti90Al6V4"  # Ti-6Al-4V
    df_input = pd.DataFrame({"formula": [formula], "target": [0.0]})
    
    print("="*75)
    print(f"CRABNET COMPREHENSIVE INPUT/OUTPUT VISUALIZATION")
    print("="*75)
    print(f"INPUT (Composition Only): {formula}")
    print("-"*75)
    print("Running inference across all available pretrained models...\n")
    
    # 2. FIND ALL MODELS
    weight_dir = "models/trained_models"
    all_pths = glob.glob(os.path.join(weight_dir, "*.pth"))
    
    prop_groups = {}
    for pth in all_pths:
        name = os.path.basename(pth).replace(".pth", "")
        if "distutils" in name: continue
        
        # Group cross-validation folds (e.g., mp_e_form0, mp_e_form1 -> mp_e_form)
        match = re.match(r"^(.*?)(\d)$", name)
        if match:
            base_name = match.group(1)
            if base_name not in prop_groups: prop_groups[base_name] = []
            prop_groups[base_name].append(pth)
        else:
            prop_groups[name] = [pth]
            
    # 3. RUN INFERENCE
    results = []
    for base_name, files in prop_groups.items():
        val = predict_property(base_name, files, df_input)
        if val is None: continue
        
        display_name = base_name
        unit = ""
        processed_val = val
        
        # Post-processing and unit assignment based on model name
        if "log10(K_VRH)" in base_name:
            processed_val = 10 ** val; display_name = "Bulk Modulus (K_VRH)"; unit = "GPa"
        elif "log10(G_VRH)" in base_name:
            processed_val = 10 ** val; display_name = "Shear Modulus (G_VRH)"; unit = "GPa"
        elif "log10" in base_name:
            processed_val = 10 ** val
        elif "e_form" in base_name or "Energy_per_atom" in base_name or "Formation_Enthalpy" in base_name:
            unit = "eV/atom"
        elif "e_hull" in base_name:
            display_name = "Energy Above Hull (Stability)"; unit = "eV/atom"
        elif "gap" in base_name.lower() or "Egap" in base_name:
            unit = "eV"
        elif "Volume_per_atom" in base_name:
            display_name = "Volume per Atom"; unit = "Å³/atom"
        elif "is_metal" in base_name:
            unit = "Probability"
        elif "yield" in base_name:
            display_name = "Yield Strength (Steels Dataset)"; unit = "MPa"
        elif "bulk_modulus" in base_name.lower() and "log10" not in base_name:
            display_name = "Bulk Modulus (AFLOW)"; unit = "GPa"
        elif "shear_modulus" in base_name.lower() and "log10" not in base_name:
            display_name = "Shear Modulus (AFLOW)"; unit = "GPa"
        elif "thermal_conductivity" in base_name: unit = "W/m-K"
        elif "thermal_expansion" in base_name: unit = "1/K"
        elif "debye" in base_name.lower(): unit = "K"
        
        # Safe formatting
        try:
            val_str = f"{float(processed_val):.4f}"
        except (ValueError, TypeError):
            val_str = str(processed_val)
            
        results.append({"Property": display_name, "Value": val_str, "Unit": unit})

    # 4. PRINT RESULTS
    df_res = pd.DataFrame(results).sort_values(by="Property")
    
    print(f"{'PROPERTY':<35} | {'PREDICTION':<15} | {'UNIT':<15}")
    print("-" * 75)
    for _, row in df_res.iterrows():
        print(f"{row['Property']:<35} | {row['Value']:<15} | {row['Unit']:<15}")
        
    print("\n" + "="*75)
    print("INTERPRETATION NOTES:")
    print("1. 'Energy Above Hull' < 0.05 eV/atom indicates the alloy is synthesizable.")
    print("2. 'Is Metal' probability should be ~1.00 for Titanium alloys.")
    print("3. 'Yield Strength' is trained on steels and will extrapolate poorly for Ti.")
    print("="*75)

    # 5. SAVE RESULTS TO FILES
    # Save to CSV
    df_res.to_csv("crabnet_results.csv", index=False)
    
    # Save to JSON
    json_data = {
        "input_formula": formula,
        "results": df_res.to_dict(orient="records")
    }
    with open("crabnet_results.json", "w") as f:
        json.dump(json_data, f, indent=4)
        
    print("\nResults saved to 'crabnet_results.csv' and 'crabnet_results.json'")

if __name__ == "__main__":
    main()