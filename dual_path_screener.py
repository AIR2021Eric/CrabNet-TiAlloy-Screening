import pandas as pd
import numpy as np
from pymatgen.core import Composition
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 1. PHYSICS BASELINE (ROM + VRH)
# =====================================================================
ELEMENTAL_PROPS = {
    "Ti": {"K": 110, "G": 42, "density": 4.51, "atomic_mass": 47.867},
    "Al": {"K": 76, "G": 26, "density": 2.70, "atomic_mass": 26.982},
    "V":  {"K": 160, "G": 47, "density": 6.00, "atomic_mass": 50.942},
    "Mo": {"K": 230, "G": 126, "density": 10.22, "atomic_mass": 95.95},
    "Zr": {"K": 91, "G": 33, "density": 6.52, "atomic_mass": 91.224},
    "Cr": {"K": 160, "G": 115, "density": 7.19, "atomic_mass": 51.996},
    "Fe": {"K": 170, "G": 82, "density": 7.87, "atomic_mass": 55.845},
    "Nb": {"K": 170, "G": 38, "density": 8.57, "atomic_mass": 92.906},
    "Sn": {"K": 58, "G": 18, "density": 7.28, "atomic_mass": 118.71},
}

def rom_vrh(formula_str):
    comp = Composition(formula_str)
    total_atoms = sum(comp.values())
    fractions = {str(el): amt/total_atoms for el, amt in comp.items()}
    
    # Filter to elements we have data for
    f = {el: frac for el, frac in fractions.items() if el in ELEMENTAL_PROPS}
    if not f: return np.nan, np.nan
    
    # Re-normalize fractions
    total_f = sum(f.values())
    f = {el: frac/total_f for el, frac in f.items()}
    
    # Voigt-Reuss-Hill for K and G
    Kv = sum(frac * ELEMENTAL_PROPS[el]["K"] for el, frac in f.items())
    Kr = 1 / sum(frac / ELEMENTAL_PROPS[el]["K"] for el, frac in f.items())
    K_rom = (Kv + Kr) / 2
    
    Gv = sum(frac * ELEMENTAL_PROPS[el]["G"] for el, frac in f.items())
    Gr = 1 / sum(frac / ELEMENTAL_PROPS[el]["G"] for el, frac in f.items())
    G_rom = (Gv + Gr) / 2
    
    E_rom = 9 * K_rom * G_rom / (3 * K_rom + G_rom)
    
    # Density
    rho_rom = 1 / sum(frac / ELEMENTAL_PROPS[el]["density"] for el, frac in f.items())
    
    return E_rom, rho_rom

# =====================================================================
# 2. CRABNET SETUP
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

def predict_crabnet(prop, weights, df_input):
    cb = CrabNet(mat_prop=prop, verbose=False)
    initialize_model(cb)
    cb.load_network(weights[0]) # Using fold 0 for speed in consensus
    pred = cb.predict(df_input)
    if isinstance(pred, tuple): pred = pred[0]
    return pred

# =====================================================================
# 3. DUAL-PATH CONSENSUS SCREENER
# =====================================================================
def run_dual_path_screening(formulas):
    print(f"Running Dual-Path Consensus Screening on {len(formulas)} alloys...")
    
    df_input = pd.DataFrame({"formula": formulas, "target": [0.0]*len(formulas)})
    
    # 1. CrabNet Path (Elasticity & Stability)
    log_k = predict_crabnet("elasticity_log10(K_VRH)", ["elasticity_log10(K_VRH)0.pth"], df_input)
    log_g = predict_crabnet("elasticity_log10(G_VRH)", ["elasticity_log10(G_VRH)0.pth"], df_input)
    hull = predict_crabnet("mp_e_hull", ["mp_e_hull.pth"], df_input)
    
    K_nn = 10 ** log_k
    G_nn = 10 ** log_g
    E_nn = (9 * K_nn * G_nn) / (3 * K_nn + G_nn)
    
    # 2. Physics Path (ROM)
    E_rom_list = []
    rho_rom_list = []
    for f in formulas:
        e, r = rom_vrh(f)
        E_rom_list.append(e)
        rho_rom_list.append(r)
        
    E_rom = np.array(E_rom_list)
    rho_rom = np.array(rho_rom_list)
    
    # 3. Consensus & Filtering
    # We use ROM density because it's highly accurate and requires no neural net.
    # We require E_nn and E_rom to agree within 15%.
    e_diff_pct = (np.abs(E_nn - E_rom) / E_rom) * 100
    consensus_passed = (e_diff_pct < 15.0) & (hull < 0.10) & (~np.isnan(E_rom))
    
    # Specific Stiffness
    spec_stiff = E_rom / rho_rom
    
    df = pd.DataFrame({
        "Formula": formulas,
        "E_CrabNet": E_nn,
        "E_ROM": E_rom,
        "E_Diff_%": e_diff_pct,
        "Density_ROM": rho_rom,
        "Spec_Stiffness": spec_stiff,
        "Hull_eV": hull,
        "Consensus_Pass": consensus_passed
    })
    
    return df

# =====================================================================
# 4. RUN ON KNOWN ALLOYS
# =====================================================================
if __name__ == "__main__":
    known_alloys = [
        "Ti90Al6V4",          # Ti-6Al-4V
        "Ti85Al5Mo5V5Cr3",    # Ti-5553
        "Ti90Al6Sn2Zr4Mo6",   # Ti-6246
        "Ti85V10Fe2Al3",      # Ti-10-2-3
    ]
    
    df_results = run_dual_path_screening(known_alloys)
    
    print("\n" + "="*80)
    print("DUAL-PATH CONSENSUS RESULTS (KNOWN ALLOYS)")
    print("="*80)
    
    # Format for nice printing
    df_print = df_results[["Formula", "E_ROM", "E_CrabNet", "E_Diff_%", "Density_ROM", "Hull_eV", "Consensus_Pass"]].copy()
    df_print["E_ROM"] = df_print["E_ROM"].round(1)
    df_print["E_CrabNet"] = df_print["E_CrabNet"].round(1)
    df_print["E_Diff_%"] = df_print["E_Diff_%"].round(1)
    df_print["Density_ROM"] = df_print["Density_ROM"].round(2)
    df_print["Hull_eV"] = df_print["Hull_eV"].round(3)
    
    print(df_print.to_string(index=False))
    
    # Save results
    df_results.to_csv("dual_path_known_alloys.csv", index=False)
    print("\nResults saved to 'dual_path_known_alloys.csv'")