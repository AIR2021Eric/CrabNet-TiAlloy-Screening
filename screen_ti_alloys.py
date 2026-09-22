import pandas as pd
import numpy as np
import warnings
from pymatgen.core import Composition
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 1. DEFINE THE 5 CRITICAL MODELS TO LOAD
# =====================================================================
MODELS_TO_LOAD = {
    "Bulk_Modulus_K": {
        "files": [f"elasticity_log10(K_VRH){i}.pth" for i in range(5)],
        "log_scale": True
    },
    "Shear_Modulus_G": {
        "files": [f"elasticity_log10(G_VRH){i}.pth" for i in range(5)],
        "log_scale": True
    },
    "Volume_per_Atom": {
        "files": ["OQMD_Volume_per_atom.pth"],
        "log_scale": False
    },
    "Energy_Above_Hull": {
        "files": ["mp_e_hull.pth"],
        "log_scale": False
    },
    "Is_Metal_Prob": {
        "files": [f"mp_is_metal{i}.pth" for i in range(5)],
        "log_scale": False
    }
}

# =====================================================================
# 2. INITIALIZE MODEL ARCHITECTURE
# =====================================================================
def initialize_model(cb):
    if cb.model is None:
        cb.model = SubCrab(
            compute_device=cb.compute_device, out_dims=cb.out_dims, d_model=cb.d_model,
            d_extend=0, N=cb.N, heads=cb.heads, pe_resolution=cb.pe_resolution,
            ple_resolution=cb.ple_resolution, emb_scaler=cb.emb_scaler,
            pos_scaler=cb.pos_scaler, pos_scaler_log=cb.pos_scaler_log,
            dim_feedforward=cb.dim_feedforward, dropout=cb.dropout,
        ).to(cb.compute_device)

# =====================================================================
# 3. ENSEMBLE PREDICTION FUNCTION (Uses all 5 folds for robustness)
# =====================================================================
def predict_with_ensemble(mat_prop, weight_files, df_input, log_scale=False):
    cb = CrabNet(mat_prop=mat_prop, verbose=False)
    initialize_model(cb)
    
    preds = []
    for wf in weight_files:
        try:
            cb.load_network(wf)
            pred = cb.predict(df_input)
            if isinstance(pred, tuple): pred = pred[0]
            preds.append(pred)
        except Exception:
            pass
    
    if not preds:
        return None, None
        
    preds_array = np.array(preds)
    mean_pred = np.mean(preds_array, axis=0)
    std_pred = np.std(preds_array, axis=0)  # Uncertainty metric
    
    if log_scale:
        mean_pred = 10 ** mean_pred
        
    return mean_pred, std_pred

# =====================================================================
# 4. DENSITY CALCULATION FROM VOLUME PER ATOM
# =====================================================================
def calculate_density(formula, volume_per_atom_A3):
    try:
        comp = Composition(formula)
        avg_atomic_weight = comp.weight  # g/mol
        volume_per_atom_cm3 = volume_per_atom_A3 * 1e-24
        mass_per_atom = avg_atomic_weight / 6.022e23
        density = mass_per_atom / volume_per_atom_cm3
        return density
    except Exception:
        return np.nan

# =====================================================================
# 5. MAIN SCREENING FUNCTION
# =====================================================================
def screen_alloys(formulas):
    df_input = pd.DataFrame({
        "formula": formulas,
        "target": [0.0] * len(formulas)
    })
    
    results = {"Formula": formulas}
    
    for prop_name, prop_config in MODELS_TO_LOAD.items():
        print(f"  Running inference for {prop_name}...")
        mean_pred, std_pred = predict_with_ensemble(
            prop_name, prop_config["files"], df_input, prop_config["log_scale"]
        )
        results[prop_name] = mean_pred
        results[f"{prop_name}_std"] = std_pred
    
    df_results = pd.DataFrame(results)
    
    # Calculate derived properties
    K = df_results["Bulk_Modulus_K"]
    G = df_results["Shear_Modulus_G"]
    df_results["Youngs_Modulus_E"] = (9 * K * G) / (3 * K + G)
    df_results["Pughs_Ratio"] = K / G
    
    # Calculate Density
    df_results["Density_g_cm3"] = df_results.apply(
        lambda row: calculate_density(row["Formula"], row["Volume_per_Atom"]), axis=1
    )
    
    # Calculate Specific Stiffness
    df_results["Specific_Stiffness"] = df_results["Youngs_Modulus_E"] / df_results["Density_g_cm3"]
    
    # Apply filters
    df_results["Is_Stable"] = df_results["Energy_Above_Hull"] < 0.05
    df_results["Is_Ductile"] = df_results["Pughs_Ratio"] > 1.75
    df_results["Is_Metal"] = df_results["Is_Metal_Prob"] > 0.5
    
    return df_results

# =====================================================================
# 6. TEST WITH KNOWN Ti ALLOYS (For Comparison)
# =====================================================================
known_alloys = [
    "Ti90Al6V4",          # Ti-6Al-4V
    "Ti95Al5",            # Ti-5Al-2.5Sn (approximate)
    "Ti85Al5Mo5V5Cr3",    # Ti-5553 (approximate)
    "Ti85V10Fe2Al3",      # Ti-10V-2Fe-3Al (approximate)
    "Ti90Al6Sn2Zr4Mo6",   # Ti-6246 (approximate)
]

# =====================================================================
# 7. TEST WITH GENERATED/HYPOTHETICAL Ti ALLOYS
# =====================================================================
generated_alloys = [
    "Ti80V10Cr5Mo5",
    "Ti85Al5V5Mo5",
    "Ti92Zr4Sn4",
    "Ti88Al4Nb4Mo2Zr2",
    "Ti75V15Cr5Fe5",
]

# =====================================================================
# 8. RUN SCREENING
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("SCREENING KNOWN Ti ALLOYS")
    print("="*80)
    df_known = screen_alloys(known_alloys)
    
    # Display only the most relevant columns
    display_cols = [
        "Formula", "Youngs_Modulus_E", "Density_g_cm3", "Specific_Stiffness",
        "Energy_Above_Hull", "Is_Stable", "Is_Ductile", "Is_Metal"
    ]
    print(df_known[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("SCREENING GENERATED Ti ALLOYS")
    print("="*80)
    df_generated = screen_alloys(generated_alloys)
    print(df_generated[display_cols].to_string(index=False))
    
    # Save results
    df_known.to_csv("known_ti_alloys_results.csv", index=False)
    df_generated.to_csv("generated_ti_alloys_results.csv", index=False)
    print("\nResults saved to CSV files.")