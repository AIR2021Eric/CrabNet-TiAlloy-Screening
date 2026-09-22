import numpy as np
import pandas as pd
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab
import warnings

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

def get_ensemble_pred(weight_prefix, df_input):
    """Runs all 5 folds and returns the ensemble average."""
    preds = []
    for i in range(5):
        cb = CrabNet(mat_prop="ensemble_test", verbose=False)
        initialize_model(cb)
        cb.load_network(f"{weight_prefix}{i}.pth")
        p = cb.predict(df_input)
        if isinstance(p, tuple): p = p[0]
        preds.append(p)
    return np.mean(preds, axis=0)

def test_2_pure_elements():
    print("=" * 70)
    print("DIAGNOSTIC 2: PURE ELEMENTS (The building blocks of Ti alloys)")
    print("=" * 70)
    
    # Reference values (GPa) from ASM Handbook / standard literature
    elements = {
        "Ti": {"K": 110, "G": 42},
        "Al": {"K": 76,  "G": 26},
        "V":  {"K": 160, "G": 47},
        "Mo": {"K": 230, "G": 126},
        "Zr": {"K": 91,  "G": 33},
        "Cr": {"K": 160, "G": 115},
        "Fe": {"K": 170, "G": 82},
    }
    
    df_input = pd.DataFrame({
        "formula": [f"{el}1" for el in elements.keys()],
        "target": [0.0] * len(elements)
    })
    
    print("\nRunning 5-fold ensemble on pure elements...")
    print("Predicting Bulk Modulus (K)...")
    log_k = get_ensemble_pred("elasticity_log10(K_VRH)", df_input)
    pred_k = 10 ** log_k
    
    print("Predicting Shear Modulus (G)...")
    log_g = get_ensemble_pred("elasticity_log10(G_VRH)", df_input)
    pred_g = 10 ** log_g
    
    print(f"\n{'Element':<6} | {'Real K':<10} | {'Pred K':<10} | {'K Err%':<10} | {'Real G':<10} | {'Pred G':<10} | {'G Err%':<10}")
    print("-" * 75)
    
    k_errors = []
    g_errors = []
    
    for i, el in enumerate(elements.keys()):
        real_k = elements[el]["K"]
        real_g = elements[el]["G"]
        
        err_k = abs(pred_k[i] - real_k) / real_k * 100
        err_g = abs(pred_g[i] - real_g) / real_g * 100
        
        k_errors.append(err_k)
        g_errors.append(err_g)
        
        print(f"{el:<6} | {real_k:<10.1f} | {pred_k[i]:<10.1f} | {err_k:<10.1f} | {real_g:<10.1f} | {pred_g[i]:<10.1f} | {err_g:<10.1f}")
        
    avg_k_err = np.mean(k_errors)
    avg_g_err = np.mean(g_errors)
    
    print("-" * 75)
    print(f"Average K Error: {avg_k_err:.1f}%")
    print(f"Average G Error: {avg_g_err:.1f}%")
    
    # PASS CRITERIA: < 20% average error is acceptable for high-throughput screening
    if avg_k_err < 20 and avg_g_err < 20:
        print("\n✅ PASS: Model accurately predicts pure metals.")
        print("   -> The failure on Matbench is likely due to complex ceramics/oxides.")
        print("   -> CrabNet is SAFE to use for metallic Ti-alloy screening.")
        return True
    else:
        print("\n❌ FAIL: Model fails on pure metals.")
        print("   -> The 2020 weights are fundamentally inadequate for your project.")
        print("   -> We MUST switch models or rely on the Physics Baseline (Test G).")
        return False

if __name__ == "__main__":
    test_2_pure_elements()