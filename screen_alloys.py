import pandas as pd
import numpy as np
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab

def initialize_model(cb):
    if cb.model is None:
        cb.model = SubCrab(
            compute_device=cb.compute_device, out_dims=cb.out_dims, d_model=cb.d_model,
            d_extend=0, N=cb.N, heads=cb.heads, pe_resolution=cb.pe_resolution,
            ple_resolution=cb.ple_resolution, emb_scaler=cb.emb_scaler,
            pos_scaler=cb.pos_scaler, pos_scaler_log=cb.pos_scaler_log,
            dim_feedforward=cb.dim_feedforward, dropout=cb.dropout,
        ).to(cb.compute_device)

# 1. Define a batch of novel hypothetical Ti-alloys to screen
# (In a real scenario, you would load a CSV of 100,000 formulas here)
candidates = [
    "Ti90Al6V4",       # Baseline: Ti-6Al-4V
    "Ti85Al5V5Mo5",    # Novel: Alpha-Beta alloy
    "Ti80V10Cr5Mo5",   # Novel: Beta-rich alloy (high strength)
    "Ti92Zr4Sn4",      # Novel: Near-alpha (high temp/creep)
    "Ti88Al4Nb4Mo2Zr2" # Novel: Complex multi-component
]

df_input = pd.DataFrame({
    "formula": candidates,
    "target": [0.0] * len(candidates)
})

print(f"Screening {len(candidates)} alloys...\n")

# We will just use Fold 0 for speed during batch screening 
# (Using all 5 folds is better for final validation, but Fold 0 is fine for initial screening)
cb_k = CrabNet(mat_prop="elasticity_log10(K_VRH)", verbose=False)
cb_g = CrabNet(mat_prop="elasticity_log10(G_VRH)", verbose=False)
initialize_model(cb_k)
initialize_model(cb_g)

cb_k.load_network("elasticity_log10(K_VRH)0.pth")
cb_g.load_network("elasticity_log10(G_VRH)0.pth")

# 2. Run Batch Inference
pred_k = cb_k.predict(df_input)
pred_g = cb_g.predict(df_input)

if isinstance(pred_k, tuple): pred_k = pred_k[0]
if isinstance(pred_g, tuple): pred_g = pred_g[0]

# 3. Calculate Engineering Properties
K_GPa = 10 ** pred_k
G_GPa = 10 ** pred_g
E_GPa = (9 * K_GPa * G_GPa) / (3 * K_GPa + G_GPa)
Pughs_Ratio = K_GPa / G_GPa

# 4. Create Results DataFrame
df_results = pd.DataFrame({
    "Formula": candidates,
    "Bulk_Modulus_K": K_GPa,
    "Shear_Modulus_G": G_GPa,
    "Youngs_Modulus_E": E_GPa,
    "Pughs_Ratio": Pughs_Ratio,
    "Is_Ductile": Pughs_Ratio > 1.75
})

# Sort by highest Young's Modulus (Stiffness)
df_results = df_results.sort_values(by="Youngs_Modulus_E", ascending=False)

print(df_results.to_string(index=False))
print("\nScreening complete! Best candidates are at the top.")