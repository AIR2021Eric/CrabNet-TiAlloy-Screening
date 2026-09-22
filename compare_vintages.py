"""compare_vintages.py - A/B test of CrabNet weight vintages on Ti-relevant checks."""
import os, numpy as np, pandas as pd, torch
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab
from crabnet.utils.utils import Scaler
import comprehensive_validation as cv   # constants, proxy, helpers

DIRS = {"v1_2020-12-04": "models/trained_models_v1_2020-12-04_patched",
        "v2_2021-02-23": "models/v2_staging"}

def cb_from_path(path):
    cb = CrabNet(mat_prop="cmp", verbose=False)
    if cb.model is None:
        cb.model = SubCrab(compute_device=cb.compute_device, out_dims=cb.out_dims,
                           d_model=cb.d_model, d_extend=0, N=cb.N, heads=cb.heads,
                           pe_resolution=cb.pe_resolution, ple_resolution=cb.ple_resolution,
                           emb_scaler=cb.emb_scaler, pos_scaler=cb.pos_scaler,
                           pos_scaler_log=cb.pos_scaler_log,
                           dim_feedforward=cb.dim_feedforward,
                           dropout=cb.dropout).to(cb.compute_device)
    net = torch.load(path, map_location=cb.compute_device, weights_only=False)
    cb.model.load_state_dict(net["weights"])
    cb.scaler = Scaler(torch.zeros(3))
    cb.scaler.load_state_dict(net["scaler_state"])
    cb.model.eval()
    return cb

def pred(tag, weight, formulas):
    p = cb_from_path(os.path.join(DIRS[tag], weight)).predict(cv.df_in(formulas))
    return np.asarray(p[0] if isinstance(p, tuple) else p, float).ravel()

def ensemble(tag, prefix, formulas, folds=5):
    return np.mean([pred(tag, f"{prefix}{i}.pth", formulas) for i in range(folds)], axis=0)

rows = []
# Check 1: pure-element K/G vs experimental refs (Test B analogue)
els = sorted(cv.ELEMENT_MODULI_EXPT)
for tag in DIRS:
    K = 10 ** ensemble(tag, "elasticity_log10(K_VRH)", [f"{e}1" for e in els])
    G = 10 ** ensemble(tag, "elasticity_log10(G_VRH)", [f"{e}1" for e in els])
    kE = np.mean([abs(K[i]-cv.ELEMENT_MODULI_EXPT[e][0])/cv.ELEMENT_MODULI_EXPT[e][0]*100 for i, e in enumerate(els)])
    gE = np.mean([abs(G[i]-cv.ELEMENT_MODULI_EXPT[e][1])/cv.ELEMENT_MODULI_EXPT[e][1]*100 for i, e in enumerate(els)])
    rows += [[tag, "B: pure-el K err %", round(kE, 1), ""],
             [tag, "B: pure-el G err %", round(gE, 1), ""]]

# Check 2: known-alloy E (Test G analogue)
for tag in DIRS:
    errs = []
    for n in cv.KNOWN_ALLOYS:
        f = cv.alloy_formula(n)
        PK = 10 ** ensemble(tag, "elasticity_log10(K_VRH)", [f])
        PG = 10 ** ensemble(tag, "elasticity_log10(G_VRH)", [f])
        E_nn = float(9*PK[0]*PG[0]/(3*PK[0]+PG[0]))
        errs.append(abs(E_nn-cv.KNOWN_ALLOYS[n]["E"])/cv.KNOWN_ALLOYS[n]["E"]*100)
    rows.append([tag, "G: alloy E MAE %", round(float(np.mean(errs)), 1), ""])

# Check 3: hull band on known alloys (Test H analogue)
for tag in DIRS:
    h = [float(pred(tag, "mp_e_hull.pth", [cv.alloy_formula(n)])[0]) for n in cv.KNOWN_ALLOYS]
    rows.append([tag, "H: hull mean eV/atom", round(float(np.mean(h)), 3),
                 "in 0.05-0.10 band" if all(0.0 <= x < 0.10 for x in h) else "OUT OF BAND"])

# Check 4: identical-protocol MatBench subset MAE (Test A analogue)
from matminer.datasets import load_dataset
from sklearn.metrics import mean_absolute_error
for task, cfg in cv.TASK_CFG.items():
    df = load_dataset(task)
    formulas, _ = cv._extract_cols(df)
    y = df[cfg["target_col"]].values
    idx = np.random.RandomState(42).choice(len(df), 3000, replace=False)
    for tag in DIRS:
        P = np.mean([pred(tag, f"{cfg['prefix']}{i}.pth", [formulas[j] for j in idx]) for i in range(5)], axis=0)
        rows.append([tag, f"A: {task} subset MAE", round(float(mean_absolute_error(y[idx], P)), 4),
                     f"leaderboard {cv.MATBENCH_LEADERBOARD_MAE[task]}"])

out = pd.DataFrame(rows, columns=["vintage", "check", "value", "note"])
print(out.to_string(index=False))
out.to_csv("results_archive/vintage_comparison.csv", index=False)