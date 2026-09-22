import pandas as pd
import numpy as np
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab  # Import the underlying PyTorch model

def initialize_model(cb):
    """
    Manually initialize the SubCrab model architecture.
    This bypasses the need to call .fit() before loading weights.
    """
    if cb.model is None:
        cb.model = SubCrab(
            compute_device=cb.compute_device,
            out_dims=cb.out_dims,
            d_model=cb.d_model,
            d_extend=0,  # 0 because we aren't using extended state features
            N=cb.N,
            heads=cb.heads,
            pe_resolution=cb.pe_resolution,
            ple_resolution=cb.ple_resolution,
            emb_scaler=cb.emb_scaler,
            pos_scaler=cb.pos_scaler,
            pos_scaler_log=cb.pos_scaler_log,
            dim_feedforward=cb.dim_feedforward,
            dropout=cb.dropout,
        ).to(cb.compute_device)

def test_crabnet_ti_alloy():
    # Define the input composition (Ti-6Al-4V)
    df_input = pd.DataFrame({
        "formula": ["Ti90Al6V4"], 
        "target": [0.0]  # Dummy target required by the API
    })
    
    print(f"Running 5-fold ensemble inference on {df_input['formula'][0]}...\n")

    # Lists to store predictions from the 5 cross-validation folds
    K_predictions = []
    G_predictions = []

    # Loop through the 5 folds
    for fold in range(5):
        # --- 1. Predict Bulk Modulus (K) ---
        k_filename = f"elasticity_log10(K_VRH){fold}.pth"
        cb_k = CrabNet(mat_prop="elasticity_log10(K_VRH)", verbose=False)
        
        # FIX: Initialize the model architecture before loading weights
        initialize_model(cb_k)
        cb_k.load_network(k_filename)
        
        # Predict (returns a numpy array of predictions)
        pred_k = cb_k.predict(df_input)
        if isinstance(pred_k, tuple): pred_k = pred_k[0] # Safety unwrap
            
        # Convert log10(K) back to actual K in GPa
        K_GPa = 10 ** pred_k[0]
        K_predictions.append(K_GPa)
        print(f"Fold {fold} Bulk Modulus (K): {K_GPa:.1f} GPa")

        # --- 2. Predict Shear Modulus (G) ---
        g_filename = f"elasticity_log10(G_VRH){fold}.pth"
        cb_g = CrabNet(mat_prop="elasticity_log10(G_VRH)", verbose=False)
        
        # FIX: Initialize the model architecture before loading weights
        initialize_model(cb_g)
        cb_g.load_network(g_filename)
        
        # Predict
        pred_g = cb_g.predict(df_input)
        if isinstance(pred_g, tuple): pred_g = pred_g[0] # Safety unwrap
            
        # Convert log10(G) back to actual G in GPa
        G_GPa = 10 ** pred_g[0]
        G_predictions.append(G_GPa)
        print(f"Fold {fold} Shear Modulus (G): {G_GPa:.1f} GPa")
        print("-" * 30)

    # Calculate Ensemble Averages
    avg_K = np.mean(K_predictions)
    std_K = np.std(K_predictions)
    
    avg_G = np.mean(G_predictions)
    std_G = np.std(G_predictions)

    # Calculate Young's Modulus (E) using Voigt-Reuss-Hill approximation
    # E = 9KG / (3K + G)
    avg_E = (9 * avg_K * avg_G) / (3 * avg_K + avg_G)

    # Calculate Pugh's Ratio (K/G) - Ductility Proxy
    pughs_ratio = avg_K / avg_G

    # Print Final Results
    print("\n" + "="*50)
    print("CRABNET ENSEMBLE PREDICTION RESULTS")
    print("="*50)
    print(f"Alloy: Ti-6Al-4V (Ti90Al6V4)")
    print("-" * 50)
    print(f"Bulk Modulus (K):    {avg_K:.1f} ± {std_K:.1f} GPa")
    print(f"Shear Modulus (G):   {avg_G:.1f} ± {std_G:.1f} GPa")
    print(f"Young's Modulus (E): {avg_E:.1f} GPa")
    print(f"Pugh's Ratio (K/G):  {pughs_ratio:.2f} (>1.75 is ductile)")
    print("-" * 50)
    print("REAL EXPERIMENTAL VALUES FOR COMPARISON:")
    print(f"Bulk Modulus (K):    ~110 - 114 GPa")
    print(f"Shear Modulus (G):   ~42 - 44 GPa")
    print(f"Young's Modulus (E): ~110 - 114 GPa")
    print("="*50)

if __name__ == "__main__":
    test_crabnet_ti_alloy()