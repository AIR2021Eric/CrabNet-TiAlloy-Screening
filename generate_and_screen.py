"""
generate_and_screen.py  (v2 - consensus-ranked, metallurgically bounded)
Changes vs v1:
  - SOLUTE_CAPS: Cr 8->4, Fe 8->4 (embrittler/intermetallic limits)
  - New columns: spec_stiff_cons (mean of proxy & NN specific stiffness), rank_clean
  - rank_clean = pass_all & Cr<=4 & Fe<=4 & Al<=11 & NN-proxy gap < 10%
  - Ranking (top_candidates.csv, report top-20) uses rank_clean + spec_stiff_cons
  - screening_results.csv keeps ALL rows/gates/flags for audit

Stages:
  1  Generate integer-at% candidates (Ti 70-95 step 5; 1-4 solutes from
     {Al,V,Mo,Zr,Sn,Cr,Fe} within caps); inject 4 known alloys as controls.
  2  Chunked inference: 5-fold elasticity ensemble, elemental_mixing_proxy,
     four K heads, dual-vintage hull.
  3  Gates: hull < 0.10 eV/atom under BOTH vintages; |E_nn-E_proxy| < 20%;
     rho_proxy in [3.8, 4.8]. Flags: K spread > 20 GPa; K fold-std > 8 GPa;
     red-team (Al>9, Cr>5, Fe>5); Pugh proxy < 1.75.
  4  Rank rank_clean survivors by spec_stiff_cons; sanity-check controls.
  5  Write screening_results.csv, top_candidates.csv, screening_report.md.

Usage:
  python generate_and_screen.py --n 20000 --chunk 10000 --seed 42 --top 100
"""
import os, re, time, argparse, datetime, warnings
import numpy as np
import pandas as pd
import comprehensive_validation as cv

warnings.filterwarnings("ignore", category=UserWarning)

for _attr in ["elemental_mixing_proxy", "hull_both_vintages", "predict_matrix", "df_in",
              "WEIGHTS_VINTAGE", "K_SOURCE_POLICY", "HULL_HEURISTIC_EV", "CONSENSUS_THRESH_PCT"]:
    if not hasattr(cv, _attr):
        raise AttributeError(
            f"comprehensive_validation.py lacks '{_attr}'. Ensure the corrected "
            "(2026-09-17, v2-weights) version is on disk before screening.")

FOLDS = 5
SOLUTES = ["Al", "V", "Mo", "Zr", "Sn", "Cr", "Fe"]
SOLUTE_CAPS = {"Al": 12, "V": 15, "Mo": 12, "Zr": 10, "Sn": 6, "Cr": 4, "Fe": 4}
TI_STEPS = [70, 75, 80, 85, 90, 95]
RED_TEAM = {"Al": 9, "Cr": 5, "Fe": 5}                 # review labels
CLEAN_LIMITS = {"Cr": 4, "Fe": 4, "Al": 11, "gap_pct": 10.0}   # ranking tier
RHO_RANGE = (3.8, 4.8)
K_SPREAD_FLAG = 20.0
K_FOLDSTD_FLAG = 8.0
PUGH_FLAG = 1.75
CONTROLS = ["Ti86Al10V4", "Ti81Al9Mo2V5Cr3", "Ti83Al11Sn1Zr2Mo3", "Ti84V9Fe2Al5"]

# --------------------------------------------------------------------------
# Stage 1 - candidate generation
# --------------------------------------------------------------------------
def parse_formula_int(f):
    return {m[0]: int(m[1]) for m in re.findall(r"([A-Z][a-z]?)(\d+)", f)}

def generate_candidates(n_target, seed):
    rng = np.random.default_rng(seed)
    seen = {}
    attempts = 0
    while len(seen) < n_target and attempts < 100 * n_target:
        attempts += 1
        ti = int(rng.choice(TI_STEPS))
        rem = 100 - ti
        k = int(rng.integers(1, 5))
        els = [str(e) for e in rng.choice(SOLUTES, size=k, replace=False)]
        parts = np.maximum(np.round(rng.dirichlet(np.ones(k)) * rem), 1).astype(int)
        parts[int(np.argmax(parts))] += rem - parts.sum()
        if parts.min() < 1:
            continue
        at = {"Ti": ti}
        if any(p > SOLUTE_CAPS[e] for e, p in zip(els, parts)):
            continue
        at.update({e: int(p) for e, p in zip(els, parts)})
        if sum(at.values()) != 100:
            continue
        formula = f"Ti{ti}" + "".join(f"{e}{at[e]}" for e in SOLUTES if e in at)
        seen[formula] = at
    for c in CONTROLS:
        seen[c] = parse_formula_int(c)
    return seen

# --------------------------------------------------------------------------
# Stage 2 - chunked inference
# --------------------------------------------------------------------------
def infer_chunk(formulas):
    df = cv.df_in(formulas)
    PK = 10 ** cv.predict_matrix([f"elasticity_log10(K_VRH){i}.pth" for i in range(FOLDS)], df)
    PG = 10 ** cv.predict_matrix([f"elasticity_log10(G_VRH){i}.pth" for i in range(FOLDS)], df)
    K_nn, K_std = PK.mean(0), PK.std(0)
    G_nn, G_std = PG.mean(0), PG.std(0)
    E_nn = np.mean(9 * PK * PG / (3 * PK + PG), axis=0)
    K_aflow = cv.predict_matrix(["aflow__ael_bulk_modulus_vrh.pth"], df)[0]
    K_mp = cv.predict_matrix(["mp_bulk_modulus.pth"], df)[0]
    px = np.array([cv.elemental_mixing_proxy(f) for f in formulas])   # (n,4): K,G,E,rho
    K_px, G_px, E_px, rho_px = px[:, 0], px[:, 1], px[:, 2], px[:, 3]
    hull_v1, hull_v2 = cv.hull_both_vintages(formulas)
    heads = np.stack([K_nn, K_aflow, K_mp, K_px], axis=1)
    return dict(K_nn=K_nn, K_fold_std=K_std, G_nn=G_nn, G_fold_std=G_std, E_nn=E_nn,
                K_aflow=K_aflow, K_mp=K_mp, K_proxy=K_px, G_proxy=G_px,
                K_range_lo=heads.min(1), K_range_hi=heads.max(1),
                K_spread=heads.max(1) - heads.min(1),
                E_proxy=E_px, rho_proxy=rho_px, hull_v1=hull_v1, hull_v2=hull_v2)

def build_rows(formulas, ats, pred):
    cols = {"formula": formulas, "Ti_at": [ats[f]["Ti"] for f in formulas],
            "n_solutes": [len(ats[f]) - 1 for f in formulas]}
    for e in SOLUTES:
        cols[f"{e}_at"] = [ats[f].get(e, 0) for f in formulas]
    cols.update({k: np.round(v, 3) for k, v in pred.items()})
    d = pd.DataFrame(cols)
    d["pugh_proxy"] = (d["K_proxy"] / d["G_proxy"]).round(3)
    d["spec_stiff"] = (d["E_proxy"] / d["rho_proxy"]).round(3)
    d["spec_stiff_nn"] = (d["E_nn"] / d["rho_proxy"]).round(3)
    d["E_consensus_diff_pct"] = (np.abs(d["E_nn"] - d["E_proxy"]) / d["E_proxy"] * 100).round(2)
    d["spec_stiff_cons"] = ((d["spec_stiff"] + d["spec_stiff_nn"]) / 2).round(3)
    # Stage 3 gates and flags
    d["gate_hull_both"] = (d["hull_v1"] < cv.HULL_HEURISTIC_EV) & (d["hull_v2"] < cv.HULL_HEURISTIC_EV)
    d["gate_consensus"] = d["E_consensus_diff_pct"] < cv.CONSENSUS_THRESH_PCT
    d["gate_rho"] = d["rho_proxy"].between(*RHO_RANGE)
    d["pass_all"] = d["gate_hull_both"] & d["gate_consensus"] & d["gate_rho"]
    d["flag_k_spread"] = d["K_spread"] > K_SPREAD_FLAG
    d["flag_k_foldstd"] = d["K_fold_std"] > K_FOLDSTD_FLAG
    d["flag_pugh"] = d["pugh_proxy"] < PUGH_FLAG
    d["flag_red_team"] = [any(ats[f].get(e, 0) > v for e, v in RED_TEAM.items()) for f in formulas]
    # Stage 4 ranking tier (metallurgically bounded + estimator consensus)
    d["rank_clean"] = (d["pass_all"]
                       & (d["Cr_at"] <= CLEAN_LIMITS["Cr"])
                       & (d["Fe_at"] <= CLEAN_LIMITS["Fe"])
                       & (d["Al_at"] <= CLEAN_LIMITS["Al"])
                       & (d["E_consensus_diff_pct"] < CLEAN_LIMITS["gap_pct"]))
    return d

# --------------------------------------------------------------------------
# Stages 3-5 - gate, rank, report
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", default="screening_output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[Stage 1] generating ~{args.n} candidates (seed={args.seed}) ...")
    cand = generate_candidates(args.n, args.seed)
    formulas = list(cand.keys())
    print(f"          unique candidates: {len(formulas)} (incl. {len(CONTROLS)} controls)")

    results_path = os.path.join(args.out, "screening_results.csv")
    frames = []
    t0 = time.time()
    with open(results_path, "w", encoding="utf-8") as fh:
        for i in range(0, len(formulas), args.chunk):
            chunk = formulas[i:i + args.chunk]
            print(f"[Stage 2] chunk {i // args.chunk + 1}: {len(chunk)} formulas ...")
            tc = time.time()
            pred = infer_chunk(chunk)
            d = build_rows(chunk, cand, pred)
            frames.append(d)
            d.to_csv(fh, header=(i == 0), index=False)
            print(f"          done in {time.time() - tc:.1f}s")
    df = pd.concat(frames, ignore_index=True)

    print("[Stage 3-4] gating and ranking ...")
    surv = df[df["rank_clean"]].sort_values("spec_stiff_cons", ascending=False)
    top = surv.head(args.top)
    top.to_csv(os.path.join(args.out, "top_candidates.csv"), index=False)

    ctrl = df[df["formula"].isin(CONTROLS)]
    ctrl_ok = bool(ctrl["pass_all"].all())

    rates = {c: float(df[c].mean()) for c in ["gate_hull_both", "gate_consensus", "gate_rho", "pass_all", "rank_clean"]}
    flags = {c: int(df[c].sum()) for c in ["flag_k_spread", "flag_k_foldstd", "flag_pugh", "flag_red_team"]}
    top_red = float(top["flag_red_team"].mean()) if len(top) else np.nan

    L = ["# Ti-Alloy High-Throughput Screening Report (v2 consensus ranking)",
         f"\nDate: {datetime.date.today()} | seed: {args.seed} | candidates: {len(df)} | chunk: {args.chunk}",
         f"\nWeights: {cv.WEIGHTS_VINTAGE}",
         f"\nK-source policy: {cv.K_SOURCE_POLICY}",
         f"\nGates: hull < {cv.HULL_HEURISTIC_EV} eV/atom under BOTH vintages; "
         f"|E_nn-E_proxy| < {cv.CONSENSUS_THRESH_PCT}%; rho in {RHO_RANGE} g/cm3.",
         f"Clean ranking tier: pass_all & Cr<={CLEAN_LIMITS['Cr']} & Fe<={CLEAN_LIMITS['Fe']} & "
         f"Al<={CLEAN_LIMITS['Al']} & NN-proxy gap < {CLEAN_LIMITS['gap_pct']}%; ranked by spec_stiff_cons.",
         f"Flags (review only): K spread > {K_SPREAD_FLAG} GPa; K fold-std > {K_FOLDSTD_FLAG} GPa; "
         f"Pugh proxy < {PUGH_FLAG}; red-team {RED_TEAM} at%.",
         "\n## Gate / tier survival\n",
         "| gate | pass rate |", "|---|---|"] + \
        [f"| {k} | {v:.3f} |" for k, v in rates.items()] + \
        ["\n## Flag counts\n", "| flag | count |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in flags.items()] + \
        [f"\nRed-team fraction in top-{args.top}: {top_red:.2f}",
         "\n## Positive controls (gate sanity; clean tier optional)\n",
         "| formula | hull_v1 | hull_v2 | E_proxy | E_nn | gap% | pass_all | rank_clean |",
         "|---|---|---|---|---|---|---|---|"] + \
        [f"| {r.formula} | {r.hull_v1} | {r.hull_v2} | {r.E_proxy} | {r.E_nn} | "
         f"{r.E_consensus_diff_pct} | {r.pass_all} | {r.rank_clean} |" for r in ctrl.itertuples()] + \
        ["" if ctrl_ok else "\n**WARNING: a positive control FAILED the gates - inspect before trusting this run.**",
         "\nNote: a control that passes gates but not rank_clean (e.g. Ti-5553, gap ~14%) is retained as a "
         "gate sanity check and flagged for estimator-disagreement review; this is expected behaviour, not a defect.",
         "\n## Top 20 by consensus specific stiffness\n",
         "| formula | Ti_at | E_proxy | rho_proxy | spec_stiff_cons | gap% | K_range | hull_v1/v2 | flags |",
         "|---|---|---|---|---|---|---|---|---|"] + \
        [f"| {r.formula} | {r.Ti_at} | {r.E_proxy} | {r.rho_proxy} | {r.spec_stiff_cons} | "
         f"{r.E_consensus_diff_pct} | {r.K_range_lo}-{r.K_range_hi} | {r.hull_v1}/{r.hull_v2} | "
         f"{'K' if r.flag_k_spread else ''}{'F' if r.flag_k_foldstd else ''}"
         f"{'P' if r.flag_pugh else ''}{'R' if r.flag_red_team else ''} |"
         for r in top.head(20).itertuples()] + \
        ["\n## Limitations",
         "- Screening under validated guardrails is NOT validation; shortlisted candidates require roadmap "
         "Tests I-M (CALPHAD phase map / beta-transus, interstitial O/N/H/C bounds, processing-conditioned "
         "properties, blind experimental holdout, red-team fail-safe) before any experimental claim.",
         "- Hull values are metastability context signals only (Aykol 2018; Sun 2016).",
         "- E and rho come from the elemental-mixing proxy; CrabNet E is the cross-check; K is quoted as a range.",
         "\n## References",
         "Wang et al. npj Comput. Mater. 7, 77 (2021); Dunn et al. npj Comput. Mater. 6, 138 (2020); "
         "Aykol et al. Sci. Adv. 4, eaaq0148 (2018); Sun et al. Sci. Adv. 2, e1600225 (2016); "
         "Hill Proc. Phys. Soc. A 65, 349 (1952); Pugh Philos. Mag. 45, 823 (1954); "
         "CRC Handbook 95th ed.; TIMET datasheets."]
    with open(os.path.join(args.out, "screening_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(x) for x in L))

    print("\n" + "=" * 78)
    print(f"candidates: {len(df)} | pass_all: {int(df['pass_all'].sum())} ({rates['pass_all']:.1%}) | "
          f"rank_clean: {int(df['rank_clean'].sum())} ({rates['rank_clean']:.1%})")
    print(f"controls pass gates: {ctrl_ok} | red-team fraction in top-{args.top}: {top_red:.2f}")
    print(f"top-{args.top} written to {os.path.join(args.out, 'top_candidates.csv')}")
    print(f"full results: {results_path} | report: {os.path.join(args.out, 'screening_report.md')}")
    print(f"total runtime: {time.time() - t0:.1f}s")
    print("=" * 78)

if __name__ == "__main__":
    main()