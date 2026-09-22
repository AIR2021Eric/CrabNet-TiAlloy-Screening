import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import warnings
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab

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

def ensemble_predict(task_name, weight_prefix, df_input, n_folds=5):
    """Average predictions across all 5 CV folds, exactly as the paper does."""
    fold_preds = []
    for fold in range(n_folds):
        cb = CrabNet(mat_prop=task_name, verbose=False)
        initialize_model(cb)
        cb.load_network(f"{weight_prefix}{fold}.pth")
        p = cb.predict(df_input)
        if isinstance(p, tuple):
            p = p[0]
        fold_preds.append(np.asarray(p))
    return np.mean(fold_preds, axis=0), fold_preds

def test_a():
    print("=" * 70)
    print("TEST A: REPRODUCE PAPER MAE (5-fold ensemble, correct targets)")
    print("=" * 70)

    tasks = {
        "matbench_log_kvrh": {"pub_mae": 0.068, "weight_prefix": "elasticity_log10(K_VRH)", "target_col": "log10(K_VRH)"},
        "matbench_log_gvrh": {"pub_mae": 0.092, "weight_prefix": "elasticity_log10(G_VRH)", "target_col": "log10(G_VRH)"},
    }

    all_passed = True
    for task_name, info in tasks.items():
        print(f"\nTesting {task_name} ...")
        df = load_dataset(task_name)
        if "structure" in df.columns:
            df["formula"] = df["structure"].apply(lambda s: s.composition.reduced_formula)

        df_input = df[["formula"]].copy()
        df_input["target"] = 0.0
        y_true = df[info["target_col"]].values

        pred, fold_preds = ensemble_predict(task_name, info["weight_prefix"], df_input)

        mae_ens = mean_absolute_error(y_true, pred)
        bias = np.mean(pred - y_true)
        corr, _ = pearsonr(y_true, pred)
        fold_maes = [mean_absolute_error(y_true, fp) for fp in fold_preds]

        print(f"  Per-fold MAE (full data): {[f'{m:.3f}' for m in fold_maes]}")
        print(f"  Ensemble MAE (full data): {mae_ens:.4f}")
        print(f"  Published ensemble MAE (test folds, 2021): {info['pub_mae']:.4f}")
        print(f"  Bias (pred - true): {bias:+.4f}   |   Pearson r: {corr:.4f}")

        # Acceptance: ensemble on full data should be at or below published test MAE
        # if weights + pipeline are faithful. Allow 1.3x slack for checkpoint vintage.
        if mae_ens <= 1.3 * info["pub_mae"] and corr > 0.90 and abs(bias) < 0.05:
            print("  ✅ PASS: pipeline faithfully reproduces published performance.")
        else:
            print("  ❌ FAIL: mismatch beyond acceptable slack.")
            print("     -> If bias ~0 and r > 0.9: weights are a weaker/earlier snapshot")
            print("        (your archive is dated 2020-12-04, pre-publication).")
            print("     -> If bias large or r < 0.9: pipeline/loading bug. Stop and debug.")
            all_passed = False
    return all_passed

if __name__ == "__main__":
    ok = test_a()
    print("\n" + "=" * 70)
    print("TEST A PASSED — proceed to Tests B–G." if ok
          else "TEST A FLAGGED — read diagnostics above before proceeding.")
    print("=" * 70)


    print("\n" + "=" * 70)
    print("Diagnostic 1: Element-stratified MAE (is the failure confined to f-block chemistry?)")
    import numpy as np, pandas as pd
    from matminer.datasets import load_dataset
    from sklearn.metrics import mean_absolute_error
    # reuse ensemble_predict + initialize_model from your test1 script

    df = load_dataset("matbench_log_kvrh")
    df["formula"] = df["structure"].apply(lambda s: s.composition.reduced_formula)
    df["max_Z"] = df["structure"].apply(lambda s: max(el.Z for el in s.composition.elements))

    light = df[df["max_Z"] <= 56].sample(3000, random_state=42)   # no lanthanides/actinides
    df_in = light[["formula"]].copy(); df_in["target"] = 0.0
    y = light["log10(K_VRH)"].values

    pred, _ = ensemble_predict("matbench_log_kvrh", "elasticity_log10(K_VRH)", df_in)
    print("MAE on light-element subset:", round(mean_absolute_error(y, pred), 4), "(published: 0.068)")
    print("=" * 70)