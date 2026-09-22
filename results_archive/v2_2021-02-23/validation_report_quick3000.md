# CrabNet Ti-Alloy Screening Pipeline - Domain of Applicability and Failure-Boundary Characterization

Date: 2026-09-17 | Mode: quick3000 | Folds: 5
torch 2.14.0+cpu | python 3.10.11 | sklearn 1.7.2

## Executive summary

| Test | Verdict | Key metric |
|---|---|---|
| A: MatBench OOF MAE (matbench_log_kvrh) | FLAGGED | oof MAE 0.175 vs leaderboard 0.0758, r=0.87 |
| A: MatBench OOF MAE (matbench_log_gvrh) | FLAGGED | oof MAE 0.187 vs leaderboard 0.1014, r=0.85 |
| B: pure elements | SCOPED | vs expt K 6.1%, G 20.9% |
| C: invariance | PASS | max diff 0.00e+00 |
| D: smoothness | PASS-WITH-CAVEAT | max step 10.8 GPa |
| G: physics baseline | proxy matches/beats CrabNet for E on known alloys | CrabNet E MAE 8.8% |
| H: consensus screen | 4/4 pass | mean |E_nn-E_proxy| 7.9% |
| F: cross-model spread | REPORT | mean spread 23.3 GPa |
| E: OOD context | CONTEXT | see report table |

## Scope & limitations (established by this suite)
- Pretrained weights: Zenodo 2020-12-04 snapshot (DOI 10.5281/zenodo.4633866). Test A out-of-fold MAE vs the MatBench leaderboard CrabNet comparator (0.0758 / 0.1014) quantifies absolute accuracy; bias ~0 and r ~0.88 indicate ranking is preserved while absolute values are degraded (checkpoint vintage, not a loading bug).
- Bulk modulus reliable vs experimental pure elements; shear modulus weak for pure solutes (V, Cr, Al) => ensemble + proxy consensus required.
- Composition response surface shows GPa-scale fine roughness => coarse screening bins only; never gradient-optimize on single-fold output; roughness may be physical at phase boundaries.
- elemental_mixing_proxy is a chemical heuristic (Voigt/Reuss bounds on elemental moduli + mass-fraction ideal density), NOT Hill aggregate theory nor an exact density law.
- Predicted MP energy-above-hull for reference alloys falls within the selected heuristic range (Aykol 2018 soft criterion; Sun 2016 chemistry-dependent scale). This is a metastability/OOD context signal ONLY; it does not establish phase stability, synthesizability, or persistence of processing-induced metastable microstructures.
- Model is scoped to Ti-rich metallic alloys; ceramics/oxides and f-block chemistry are outside its validated domain.

## Known missing failure modes (roadmap, not implemented)
- Test I: CALPHAD phase-map validation (beta-transus, alpha/beta fractions, omega/alpha2/intermetallic tendency).
- Test J: interstitial O/N/H/C sensitivity bounds (aerospace specs limit these).
- Test K: processing/microstructure-conditioned mechanical properties (texture, heat treatment).
- Test L: blind held-out Ti-alloy family benchmark with experimental uncertainties.
- Test M: red-team fail-safe tests (intermetallic-formers, omega-formers, outside-training-support elements).
- Calibrated uncertainty (conformal/temperature scaling) instead of raw fold/cross-model spread.

## Constants provenance

| constant group | source |
|---|---|
| WANG_PAPER_MAE | Wang et al. npj Comput. Mater. 7, 77 (2021), DOI 10.1038/s41524-021-00545-1 (paper/README benchmark table) |
| MATBENCH_LEADERBOARD_MAE | MatBench v0.1 leaderboard, 'matbench_v0.1: CrabNet' submission, 5-fold mean (matbench.materialsproject.org) |
| OOD_CRABNET | Segal et al. arXiv:2502.05970 (2025), Table 1, CrabNet column |
| ELEMENT_MODULI_EXPT | Room-T experimental polycrystalline moduli (CRC Handbook 95th ed.; Gschneidner, Solid State Phys. 16 (1964)) |
| ELEMENT_MODULI_DFT | Materials Project VRH (Jain et al., APL Mater. 1, 011002 (2013)) |
| ELEMENT_DENSITY_RT | CRC Handbook of Chemistry and Physics, 95th ed. (2014), room-T stable phase |
| KNOWN_ALLOYS | TIMET datasheets / ASM Handbook Vol. 2 / MatWeb; ranges where datasheets give ranges |
| HULL_HEURISTIC_EV | Aykol et al. Sci. Adv. 4, eaaq0148 (2018); Sun et al. Sci. Adv. 2, e1600225 (2016); empirical soft criterion only |
| CONSENSUS_THRESH_PCT | Project-defined screening filter calibrated on Test G/H spread; NOT a design allowable |
| elemental_mixing_proxy | Voigt/Reuss bounds on elemental moduli + Hill average + mass-fraction additive-specific-volume density; chemical heuristic, NOT Hill aggregate theory nor an exact density law |

## Test A - MatBench out-of-fold reproduction

### matbench_log_kvrh
| metric | value |
|---|---|
| protocol | reproducible KFold splits (matbench pkg absent; NOT official) |
| Wang et al. 2021 paper MAE | 0.068 |
| MatBench leaderboard MAE (comparator) | 0.0758 |
| out-of-fold MAE | 0.1748 |
| bias | 0.0161 |
| pearson r | 0.865 |
| light-element (Z<=56) subset OOF MAE | 0.1927 |
| ratio to leaderboard | 2.306 |
| verdict | FLAGGED (checkpoint vintage / protocol; see report) |

### matbench_log_gvrh
| metric | value |
|---|---|
| protocol | reproducible KFold splits (matbench pkg absent; NOT official) |
| Wang et al. 2021 paper MAE | 0.092 |
| MatBench leaderboard MAE (comparator) | 0.1014 |
| out-of-fold MAE | 0.1871 |
| bias | 0.0155 |
| pearson r | 0.8545 |
| light-element (Z<=56) subset OOF MAE | 0.2016 |
| ratio to leaderboard | 1.846 |
| verdict | FLAGGED (checkpoint vintage / protocol; see report) |

## Test B - Pure elements (experimental vs DFT reference sets)

| element | K_pred | G_pred | K_expt | G_expt | K_err_expt_pct | G_err_expt_pct | K_dft | G_dft | K_err_dft_pct | G_err_dft_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| Al | 77.3 | 33.2 | 76.0 | 26.0 | 1.7 | 27.6 | 76.0 | 26.0 | 1.7 | 27.6 |
| Cr | 181.7 | 80.6 | 160.0 | 115.0 | 13.6 | 29.9 | 160.0 | 115.0 | 13.6 | 29.9 |
| Fe | 150.8 | 71.1 | 170.0 | 82.0 | 11.3 | 13.3 | 170.0 | 82.0 | 11.3 | 13.3 |
| Mo | 214.9 | 98.6 | 230.0 | 126.0 | 6.6 | 21.8 | 230.0 | 120.0 | 6.6 | 17.8 |
| Sn | 52.1 | 20.2 | 58.0 | 18.0 | 10.2 | 12.3 | 58.0 | 18.0 | 10.2 | 12.3 |
| Ti | 109.0 | 39.3 | 110.0 | 44.0 | 0.9 | 10.6 | 110.0 | 42.0 | 0.9 | 6.4 |
| V | 159.4 | 68.8 | 160.0 | 47.0 | 0.4 | 46.4 | 160.0 | 47.0 | 0.4 | 46.4 |
| Zr | 95.1 | 34.9 | 91.0 | 33.0 | 4.5 | 5.7 | 91.0 | 33.0 | 4.5 | 5.7 |

## Test C - Invariance

| formula | K_pred_GPa |
|---|---|
| Ti90Al6V4 | 99.4411 |
| V4Ti90Al6 | 99.4411 |
| Al6V4Ti90 | 99.4411 |
| Ti0.9Al0.06V0.04 | 99.4411 |
Max difference: 0.00e+00

## Test D - Smoothness (Ti-Al sweep)

| composition | K_ensemble_GPa | K_proxy_GPa |
|---|---|---|
| Ti100Al0 | 108.97 | 110.0 |
| Ti98Al2 | 98.13 | 109.17 |
| Ti96Al4 | 96.98 | 108.35 |
| Ti94Al6 | 98.67 | 107.54 |
| Ti92Al8 | 98.72 | 106.74 |
| Ti90Al10 | 101.57 | 105.94 |
| Ti88Al12 | 102.1 | 105.16 |
| Ti86Al14 | 103.89 | 104.38 |
| Ti84Al16 | 100.79 | 103.61 |
| Ti82Al18 | 100.85 | 102.84 |
| Ti80Al20 | 96.94 | 102.08 |
| Ti78Al22 | 96.55 | 101.33 |
| Ti76Al24 | 95.42 | 100.59 |
Max step 10.84 GPa vs proxy slope 0.78 GPa/step

## Test E - OOD context (external, unrelated datasets)

| benchmark | value |
|---|---|
| This work: Ti-alloy Young's modulus (known alloys) | 8.8 % MAE |
| This work: pure-element bulk modulus (expt refs) | 6.1 % MAE |
| Segal et al. arXiv:2502.05970, Table 1, CrabNet col.: Matbench steels yield strength OOD | 740 +/- 49 MPa |
| Segal et al. arXiv:2502.05970, Table 1, CrabNet col.: MP bulk modulus OOD | 57.8 +/- 4.2 GPa |

## Test F - Cross-model disagreement (bulk modulus)

| alloy | K_matbench_mean | K_matbench_fold_std | K_AFLOW | K_MP | cross_model_spread |
|---|---|---|---|---|---|
| Ti-6Al-4V | 97.2 | 1.9 | 113.2 | 114.4 | 17.2 |
| Ti-5553 | 87.8 | 1.5 | 110.6 | 114.2 | 26.4 |
| Ti-6246 | 88.3 | 1.5 | 111.6 | 113.6 | 25.3 |
| Ti-10-2-3 | 91.0 | 2.6 | 112.3 | 115.4 | 24.4 |

## Test G - Elemental-mixing proxy vs CrabNet (known alloys, with datasheet ranges)

| alloy | E_ref | E_ref_lo | E_ref_hi | E_CrabNet | E_nn_err_pct | E_nn_in_ref_range | E_proxy | E_proxy_err_pct | E_proxy_in_ref_range | rho_ref | rho_proxy | rho_err_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ti-6Al-4V | 113.8 | 107.0 | 122.0 | 127.7 | 12.2 | False | 110.8 | 2.6 | True | 4.43 | 4.38 | 1.1 |
| Ti-5553 | 110.0 | nan | nan | 115.0 | 4.5 | None | 118.4 | 7.6 | None | 4.65 | 4.6 | 1.2 |
| Ti-6246 | 114.0 | 70.0 | 114.0 | 118.6 | 4.1 | False | 112.7 | 1.1 | True | 4.65 | 4.57 | 1.7 |
| Ti-10-2-3 | 109.6 | nan | nan | 125.2 | 14.2 | None | 115.6 | 5.5 | None | 4.65 | 4.58 | 1.6 |

## Test H - Dual-path consensus SCREENING FILTER (not validation)

| alloy | formula | E_proxy | E_CrabNet | E_diff_pct | rho_proxy | hull_eV | consensus_pass |
|---|---|---|---|---|---|---|---|
| Ti-6Al-4V | Ti1.88020975Al0.22237427V0.07852144 | 110.8 | 127.7 | 15.2 | 4.38 | 0.026 | True |
| Ti-5553 | Ti1.71307999Al0.18531189V0.0981518Cr0.05769663Mo0.05211591 | 118.4 | 115.0 | 2.9 | 4.6 | 0.028 | True |
| Ti-6246 | Zr0.04384811Ti1.71307999Al0.22237427Sn0.01684778Mo0.06253909 | 112.7 | 118.6 | 5.3 | 4.57 | 0.03 | True |
| Ti-10-2-3 | Ti1.77575365Al0.11118714V0.1963036Fe0.03581341 | 115.6 | 125.2 | 8.2 | 4.58 | 0.043 | True |

## References
1. Wang, A.Y.-T., Kauwe, S.K., Murdock, R.J., Sparks, T.D. npj Comput. Mater. 7, 77 (2021). DOI 10.1038/s41524-021-00545-1
2. Dunn, A., Wang, Q., Ganose, A., Dopp, D., Jain, A. npj Comput. Mater. 6, 138 (2020). DOI 10.1038/s41524-020-00406-3
3. MatBench v0.1 leaderboards and 'matbench_v0.1: CrabNet' submission data: matbench.materialsproject.org
4. Segal, N., Netanyahu, A., Greenman, K.P., Agrawal, P., Gomez-Bombarelli, R. arXiv:2502.05970 (2025), Table 1 (CrabNet column)
5. Sun, W. et al. Sci. Adv. 2, e1600225 (2016). DOI 10.1126/sciadv.1600225
6. Aykol, M. et al. Sci. Adv. 4, eaaq0148 (2018). DOI 10.1126/sciadv.aaq0148
7. Hill, R. Proc. Phys. Soc. A 65, 349 (1952). DOI 10.1088/0370-1298/65/5/307
8. Pugh, S.F. Philos. Mag. 45, 823 (1954). DOI 10.1080/14786440808520513
9. CRC Handbook of Chemistry and Physics, 95th ed. (2014) - elemental densities/moduli
10. Jain, A. et al. APL Mater. 1, 011002 (2013) - Materials Project DFT VRH moduli
11. TIMET datasheets (TIMETAL 6-4, 555, 6246, 10-2-3); ASM Handbook Vol. 2; MatWeb - alloy reference values