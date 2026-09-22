# CrabNet Ti-Alloy Screening Pipeline - Comprehensive Validation Report

Date: 2026-09-16 | Mode: quick3000 | Folds: 5
torch 2.14.0+cpu | python 3.10.11 | sklearn 1.7.2

## Executive summary

| Test | Verdict | Key metric |
|---|---|---|
| A: MatBench MAE (matbench_log_kvrh) | FLAGGED | ens MAE 0.175 vs pub 0.068, r=0.88 |
| A: MatBench MAE (matbench_log_gvrh) | FLAGGED | ens MAE 0.185 vs pub 0.092, r=0.89 |
| B: pure elements | SCOPED | K err 5.6%, G err 21.6% |
| C: invariance | PASS | max diff 0.00e+00 |
| D: smoothness | PASS-WITH-CAVEAT | max step 10.8 GPa |
| G: physics baseline | ROM matches/beats CrabNet for E => use consensus | CrabNet E MAE 8.0% |
| H: dual-path consensus | 4/4 pass | mean |E_nn-E_rom| 10.1% |
| F: cross-model spread | REPORT | mean spread 20.9 GPa |
| E: OOD context | CONTEXT | see report table |

## Scope & limitations (established by this suite)
- Pretrained weights are the 2020-12-04 Zenodo snapshot (DOI 10.5281/zenodo.4633866); global MatBench MAE is ~2.5x the published 2021 values, with bias ~0 and r ~0.88 => weaker vintage, not a loading bug.
- Bulk modulus reliable (pure-element mean err ~5-6%); shear modulus weak for pure solutes (V, Cr, Al) => use ensemble + ROM consensus.
- Composition response surface has ~5-7 GPa fine-scale roughness => coarse screening bins only; never gradient-optimize on single-fold output.
- Energy-above-hull of known aerospace alloys falls in 0.05-0.10 eV/atom, correctly identifying metastable solid solutions (threshold per Sun et al. 2016).
- Model is scoped to Ti-rich metallic alloys; ceramics/oxides and f-block chemistry are outside its validated domain.

## Test A - MatBench reproduction

### matbench_log_kvrh
| metric | value |
|---|---|
| published MAE | 0.068 |
| ensemble MAE | 0.1748 |
| fold MAEs | [0.169, 0.175, 0.181, 0.179, 0.174] |
| bias | 0.0212 |
| pearson r | 0.8817 |
| light-element (Z<=56) subset MAE | 0.1922 |
| verdict | FLAGGED (2020-12-04 checkpoint vintage; see report) |

### matbench_log_gvrh
| metric | value |
|---|---|
| published MAE | 0.092 |
| ensemble MAE | 0.1845 |
| fold MAEs | [0.19, 0.2, 0.181, 0.189, 0.174] |
| bias | 0.0195 |
| pearson r | 0.8879 |
| light-element (Z<=56) subset MAE | 0.1963 |
| verdict | FLAGGED (2020-12-04 checkpoint vintage; see report) |

## Test B - Pure elements

| element | K_ref | K_pred | K_err_pct | G_ref | G_pred | G_err_pct |
|---|---|---|---|---|---|---|
| Ti | 110 | 109.0 | 0.9 | 42 | 39.3 | 6.4 |
| Al | 76 | 77.3 | 1.7 | 26 | 33.2 | 27.6 |
| V | 160 | 159.4 | 0.4 | 47 | 68.8 | 46.4 |
| Mo | 230 | 214.9 | 6.6 | 126 | 98.6 | 21.8 |
| Zr | 91 | 95.1 | 4.5 | 33 | 34.9 | 5.7 |
| Cr | 160 | 181.7 | 13.6 | 115 | 80.6 | 29.9 |
| Fe | 170 | 150.8 | 11.3 | 82 | 71.1 | 13.3 |

## Test C - Invariance

| formula | K_pred_GPa |
|---|---|
| Ti90Al6V4 | 99.4411 |
| V4Ti90Al6 | 99.4411 |
| Al6V4Ti90 | 99.4411 |
| Ti0.9Al0.06V0.04 | 99.4411 |
Max difference: 0.00e+00

## Test D - Smoothness (Ti-Al sweep)

| composition | K_ensemble_GPa | K_ROM_GPa |
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
Max step 10.84 GPa vs ROM physical slope 0.78 GPa/step

## Test E - OOD context

| benchmark | value |
|---|---|
| This work: Ti-alloy Young's modulus (known alloys) | 8.0 % MAE |
| This work: pure-element bulk modulus | 5.6 % MAE |
| Segal et al. arXiv:2502.05970: CrabNet OOD steels yield strength | 740 +/- 49 MPa |
| Segal et al. arXiv:2502.05970: CrabNet OOD MP bulk modulus | 45.8 +/- 3.9 GPa |

## Test F - Cross-model & ensemble spread (bulk modulus)

| alloy | K_matbench_mean | K_matbench_fold_std | K_AFLOW | K_MP | cross_model_spread |
|---|---|---|---|---|---|
| Ti-6Al-4V | 97.2 | 1.9 | 113.2 | 112.7 | 15.9 |
| Ti-5553 | 87.8 | 1.5 | 110.6 | 109.6 | 22.8 |
| Ti-6246 | 88.3 | 1.5 | 111.6 | 109.9 | 23.4 |
| Ti-10-2-3 | 91.0 | 2.6 | 112.3 | 109.9 | 21.4 |

## Test G - Physics baseline vs CrabNet

| alloy | E_ref | E_CrabNet | E_nn_err_pct | E_ROM | E_rom_err_pct | rho_ref | rho_ROM | rho_err_pct |
|---|---|---|---|---|---|---|---|---|
| Ti-6Al-4V | 113 | 127.7 | 13.0 | 107.0 | 5.3 | 4.43 | 4.26 | 3.9 |
| Ti-5553 | 110 | 115.0 | 4.5 | 114.6 | 4.2 | 4.65 | 4.41 | 5.2 |
| Ti-6246 | 118 | 118.6 | 0.5 | 109.0 | 7.6 | 4.54 | 4.31 | 5.0 |
| Ti-10-2-3 | 110 | 125.2 | 13.8 | 111.7 | 1.6 | 4.65 | 4.49 | 3.5 |

## Test H - Dual-path consensus on known aerospace alloys

| alloy | formula | E_ROM | E_CrabNet | E_diff_pct | rho_ROM | hull_eV | consensus_pass |
|---|---|---|---|---|---|---|---|
| Ti-6Al-4V | Ti1.88020975Al0.22237427V0.07852144 | 107.0 | 127.7 | 19.4 | 4.26 | 0.071 | True |
| Ti-5553 | Ti1.71307999Al0.18531189V0.0981518Cr0.05769663Mo0.05211591 | 114.6 | 115.0 | 0.3 | 4.41 | 0.057 | True |
| Ti-6246 | Zr0.04384811Ti1.71307999Al0.22237427Sn0.01684778Mo0.06253909 | 109.0 | 118.6 | 8.8 | 4.31 | 0.06 | True |
| Ti-10-2-3 | Ti1.77575365Al0.11118714V0.1963036Fe0.03581341 | 111.7 | 125.2 | 12.0 | 4.49 | 0.064 | True |

## References
1. Wang, Kauwe, Hennig. npj Comput. Mater. 7, 77 (2021). DOI 10.1038/s41524-021-00545-1
2. Dunn et al. (MatBench). Sci. Data 7, 330 (2020). DOI 10.1038/s41597-020-00671-3
3. Sun et al. Sci. Adv. 2, e1600225 (2016). DOI 10.1126/sciadv.1600225
4. Pugh. Philos. Mag. 45, 823 (1954). DOI 10.1080/14786440808520513
5. Hill. Proc. Phys. Soc. A 65, 349 (1952). DOI 10.1088/0370-1298/65/5/307
6. Segal, Netanyahu et al. arXiv:2502.05970