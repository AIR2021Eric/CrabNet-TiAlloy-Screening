# CrabNet Ti-Alloy Screening Pipeline

A rigorously validated, guardrailed machine-learning pipeline for high-throughput
screening of hypothetical titanium alloys, built on the composition-only
**CrabNet** architecture (Wang et al., *npj Comput. Mater.* 7, 77, 2021).

The repository documents an 8-test validation suite (Tests A–H) that establishes
the model's **domain of applicability and failure boundaries** for Ti-rich
aerospace chemistry, byte-level weight-vintage forensics, a metallurgically
bounded exhaustive screening campaign (~39.5k unique coarse-bin compositions),
and a CPU throughput/scaling benchmark.

> **Status:** validation complete; model cleared as a *ranking/screening engine*
> under explicit guardrails. Outputs are screening hypotheses, not validated
> alloys (see Roadmap).

---

## 1. What is (and is not) in this repository

**Included:** all scripts, validation reports, screening archives, benchmarks,
figures, environment lock file, and the neutral cross-model comparison baseline.

**Excluded (gitignored), recoverable:**
- `venv/` — recreate from `requirements_crabnet.txt` (Python 3.10.11).
- `models/**/*.pth` and `*.7z` — pretrained weights; download from
  Zenodo record [10.5281/zenodo.4633866](https://zenodo.org/records/4633866)
  (see §3).
- `validation_output/`, `screening_output/` — regenerable; canonical copies
  archived under `results_archive/`.

---

## 2. Environment & setup

```powershell
python -m venv venv                      # Python 3.10.11
.\venv\Scripts\Activate.ps1
pip install -r requirements_crabnet.txt
```

Key pins: `crabnet==2.0.8` (PyPI; official release by Wang, Kauwe, Baird —
uses original attribute names, natively compatible with the unmodified Zenodo
weights), `torch==2.14.0+cpu`, `scikit-learn==1.7.2`, `matminer==0.9.3`,
`pymatgen==2025.10.7`, `pandas==2.3.3`, `numpy==2.2.6`, `scipy==1.15.3`,
`matplotlib==3.10.9`.

---

## 3. Weight provenance & preparation

| Item | Value |
|---|---|
| Source | Zenodo 10.5281/zenodo.4633866 (mat2vec, bsz_auto) |
| Active vintage | **2021-02-23** → `models/trained_models/` |
| Backup vintage | **2020-12-04** → `models/trained_models_v1_active_backup/` (conservative hull head) |
| Forensics | 74/83 heads byte-identical (SHA-256) between vintages, incl. all 10 elasticity folds and all AFLOW heads; 9 single-model MP/OQMD heads retrained in 2021 (incl. `mp_e_hull`, `mp_bulk_modulus`) |
| Archive integrity | `results_archive/v2_archive_sha256.txt` (MD5 matches Zenodo listing) |

Preparation (after downloading the two `.7z` files from Zenodo):

```powershell
python extract_weights_v2.py                          # extract 2021 archive into staging
python patch_weights_dir.py models\v2_staging         # rename encoder keys if required by your crabnet build
python inspect_keys.py "models\v2_staging\elasticity_log10(K_VRH)0.pth"
# then place: 2021 set -> models\trained_models ; 2020 set -> models\trained_models_v1_active_backup
```

Manifests: `results_archive/manifest_v1_2020-12-04.csv`,
`manifest_v2_active.csv`, `vintage_comparison.csv`.

---

## 4. Validation suite (Tests A–H)

Run: `python comprehensive_validation.py --full` (or without `--full` for quick mode).
Outputs: `validation_output/validation_report_full.md`, `validation_results_full.json`, `test*_full.csv`.

| Test | Purpose | Result | Verdict |
|---|---|---|---|
| A | MatBench out-of-fold reproduction (log_kvrh / log_gvrh) | OOF MAE 0.175 / 0.187 vs leaderboard 0.0758 / 0.1014; r = 0.87 / 0.85; bias ≈ 0 | FLAGGED — ranking preserved, absolute values degraded |
| B | Pure-element endpoints (K, G) vs experiment | K err 6.1%; G err 20.9% (V, Cr, Al worst) | SCOPED |
| C | Composition-string invariance | max diff 0.00 | PASS |
| D | Response-surface smoothness (Ti–Al sweep) | max step 10.8 GPa vs proxy slope 0.78 GPa/step | PASS-WITH-CAVEAT — coarse bins only |
| E | OOD context vs Segal et al. (2025) CrabNet column | in-domain 6–9% vs published OOD collapses | CONTEXT |
| F | Cross-head bulk-modulus disagreement | mean spread 23.3 GPa; matbench head = systematic low outlier | REPORT — K quoted as range |
| G | Zero-parameter physics baseline | proxy matches/beats CrabNet for E (CrabNet E MAE 8.8%; proxy inside datasheet ranges) | Consensus required |
| H | Dual-path consensus screen on 4 aerospace alloys | 4/4 pass; hull v2 0.026–0.043, v1 0.057–0.071 eV/atom | PASS |

Archived per-vintage reports: `results_archive/v1_2020-12-04/`,
`results_archive/v2_2021-02-23/`.

---

## 5. Guardrails & operating rules (frozen)

- **Hard gates:** dual-vintage hull < 0.10 eV/atom under **both** weight heads;
  |E_nn − E_proxy| < 20%; ρ_proxy ∈ [3.8, 4.8] g/cm³.
- **Ranking tier (`rank_clean`):** gates + Cr ≤ 4, Fe ≤ 4, Al ≤ 11 at%,
  proxy–NN gap < 10%; ranked by consensus specific stiffness
  `spec_stiff_cons = mean(E_proxy, E_nn)/ρ_proxy`.
- **K-source policy:** bulk modulus always quoted as a range across four heads
  (matbench-weights, AFLOW, MP, elemental proxy); flag spread > 20 GPa; never
  use the matbench head alone.
- **E and ρ:** elemental-mixing proxy primary (mass-fraction ideal mixing);
  CrabNet E as cross-check.
- **Hull:** metastability context signal only (Aykol 2018; Sun 2016) — not a
  synthesizability proof.
- **Search space:** coarse bins (Ti 70–95 at% step 5; ≤4 solutes from
  Al, V, Mo, Zr, Sn, Cr, Fe within caps). Never gradient-optimize (Test D).
- **Review flags (not rejections):** red-team Al > 9, Cr > 5, Fe > 5 at%;
  K fold-std > 8 GPa; Pugh proxy < 1.75.

---

## 6. Screening campaign

Run: `python generate_and_screen.py --n 40000 --chunk 15000 --seed 42 --top 500`

| Run (seed 42) | Candidates | pass_all | rank_clean | Controls |
|---|---|---|---|---|
| 2026-09-18 (exhaustive) | 39,480 (100% of capped space) | 34.6% | 15.3% | 4/4 |
| 2026-09-21 (92% coverage) | 36,467 | 35.8% | 16.0% | 4/4 |

- **Reproducibility:** both runs produced identical top candidates
  (Δ = 0.000) — ranking is stable, not a sampling artifact.
- **Top candidates:** Ti85V15 (26.10), Ti90V9Fe1 (25.92), Ti90V9Cr1 (25.88),
  Ti90Al9V1 (25.88) vs Ti-6-4V control 25.42 (consensus specific stiffness,
  GPa·cm³/g) — plausible α+β / metastable-β chemistries.
- **Failure mode caught & fixed:** an early E/ρ-only ranking piled candidates
  at the Cr/Fe generation caps (Laves-phase territory invisible to the linear
  proxy; 100% of top-100 red-flagged). Fixed by tightened caps + consensus
  ranking tier (§5).
- Archives: `results_archive/screen_exhaustive_39k_seed42/`,
  `screen_36467_seed42/`, `screen_pilot20k_v2ranking_seed42/`,
  `screen_pilot_20k_seed42/`.

---

## 7. Performance benchmark

Run: `python benchmark_speed.py` (sweep + linearity + integrated plotting);
replot from CSV: `python plot_benchmark_v2.py`.

- Throughput plateaus at **≈244 samples/s for chunk 12.5k–20k** (true optimum
  ≈15,000; declines to ≈220 s/s at 25–30k; ≈180 s/s at 1k).
- Linear scaling in N at fixed chunk: **R² = 0.9979** to 100k tested;
  extrapolation **≈69 min per 1M candidates** (CPU).
- The capped coarse-bin design space (~39.5k unique alloys) screens in < 3 min.
- Figures: `results_archive/slide18_throughput_visuals.png`;
  data: `benchmark_results_v2.csv`.

---

## 8. Cross-model comparison baseline

`comparison_baseline/` holds the neutral, model-agnostic artifacts used to
benchmark alternative composition-only models (Roost, MODNet, Automatminer, …):

- `eval_set_20k.csv` — fixed 20,004-formula evaluation set (seed 42).
- `controls.csv` — four aerospace reference alloys.
- `model_results_crabnet.json` — standardized CrabNet summary in the shared
  result schema (Test A/B/C/D/G/H metrics, throughput, screening funnel).

---

## 9. Repository layout

```
comprehensive_validation.py   8-test suite (Tests A–H), guardrail constants
generate_and_screen.py        screening engine (generate → infer → gate → rank)
benchmark_speed.py            chunk sweep + linearity + plotting
patch_weights_dir.py          weight key renaming for fork compatibility
inspect_keys.py               checkpoint key-format verifier
extract_weights_v2.py         Zenodo 7z extraction helper
compare_vintages.py           v1-vs-v2 head-to-head A/B
plot_slides_17_22.py          forensics & screening figures
results_archive/              canonical reports, manifests, screen archives, figures
comparison_baseline/          neutral cross-model evaluation artifacts
requirements_crabnet.txt      exact environment lock (Python 3.10.11)
```

---

## 10. References

Wang et al., *npj Comput. Mater.* 7, 77 (2021), DOI 10.1038/s41524-021-00545-1 ·
Dunn et al., *npj Comput. Mater.* 6, 138 (2020), DOI 10.1038/s41524-020-00406-3 ·
MatBench v0.1 leaderboards (matbench.materialsproject.org) ·
Segal et al., arXiv:2502.05970 (2025), Table 1 ·
Sun et al., *Sci. Adv.* 2, e1600225 (2016) · Aykol et al., *Sci. Adv.* 4, eaaq0148 (2018) ·
Hill, *Proc. Phys. Soc.* A 65, 349 (1952) · Pugh, *Philos. Mag.* 45, 823 (1954) ·
CRC Handbook 95th ed. · TIMET datasheets / ASM Handbook Vol. 2.

---

*Weights are not redistributed here; cite the original Zenodo record and the
CrabNet paper when reusing this pipeline.*
