# Ti-Alloy High-Throughput Screening Report (v2 consensus ranking)

Date: 2026-09-18 | seed: 42 | candidates: 39480 | chunk: 20000

Weights: 2021-02-23 mat2vec (bsz_auto), Zenodo 10.5281/zenodo.4633866; 74/83 heads byte-identical to the 2020-12-04 set (all elasticity/AFLOW); 9 MP/OQMD single-model heads retrained in the 2021 run (incl. mp_e_hull, mp_bulk_modulus)

K-source policy: Bulk modulus is reported as a range across heads (matbench-weights, AFLOW, MP, elemental proxy); the matbench-weights head is a systematic low outlier for Ti-rich alloys and is never used alone.

Gates: hull < 0.1 eV/atom under BOTH vintages; |E_nn-E_proxy| < 20.0%; rho in (3.8, 4.8) g/cm3.
Clean ranking tier: pass_all & Cr<=4 & Fe<=4 & Al<=11 & NN-proxy gap < 10.0%; ranked by spec_stiff_cons.
Flags (review only): K spread > 20.0 GPa; K fold-std > 8.0 GPa; Pugh proxy < 1.75; red-team {'Al': 9, 'Cr': 5, 'Fe': 5} at%.

## Gate / tier survival

| gate | pass rate |
|---|---|
| gate_hull_both | 0.988 |
| gate_consensus | 0.789 |
| gate_rho | 0.357 |
| pass_all | 0.346 |
| rank_clean | 0.153 |

## Flag counts

| flag | count |
|---|---|
| flag_k_spread | 39328 |
| flag_k_foldstd | 0 |
| flag_pugh | 0 |
| flag_red_team | 5815 |

Red-team fraction in top-500: 0.18

## Positive controls (gate sanity; clean tier optional)

| formula | hull_v1 | hull_v2 | E_proxy | E_nn | gap% | pass_all | rank_clean |
|---|---|---|---|---|---|---|---|
| Ti86Al10V4 | 0.085 | 0.029 | 110.986 | 112.068 | 0.97 | True | True |
| Ti81Al9Mo2V5Cr3 | 0.057 | 0.028 | 117.953 | 101.384 | 14.05 | True | False |
| Ti83Al11Sn1Zr2Mo3 | 0.062 | 0.03 | 112.393 | 105.933 | 5.75 | True | True |
| Ti84V9Fe2Al5 | 0.063 | 0.04 | 115.96 | 111.78 | 3.6 | True | True |


Note: a control that passes gates but not rank_clean (e.g. Ti-5553, gap ~14%) is retained as a gate sanity check and flagged for estimator-disagreement review; this is expected behaviour, not a defect.

## Top 20 by consensus specific stiffness

| formula | Ti_at | E_proxy | rho_proxy | spec_stiff_cons | gap% | K_range | hull_v1/v2 | flags |
|---|---|---|---|---|---|---|---|---|
| Ti85V15 | 85 | 118.27 | 4.705 | 26.096 | 7.63 | 103.446-117.638 | 0.017/0.034 |  |
| Ti90V9Fe1 | 90 | 118.302 | 4.649 | 25.921 | 3.73 | 95.177-115.045 | 0.04/0.031 |  |
| Ti90V9Cr1 | 90 | 118.762 | 4.644 | 25.88 | 2.4 | 95.023-114.359 | 0.045/0.029 |  |
| Ti90Al9V1 | 90 | 111.22 | 4.368 | 25.879 | 3.27 | 94.767-113.883 | 0.035/0.026 |  |
| Ti90V9Mo1 | 90 | 118.965 | 4.677 | 25.811 | 2.95 | 95.609-114.747 | 0.042/0.028 |  |
| Ti90V8Fe2 | 90 | 118.933 | 4.659 | 25.78 | 1.98 | 95.253-114.761 | 0.039/0.029 |  |
| Ti90V8Cr2 | 90 | 119.853 | 4.65 | 25.754 | 0.16 | 95.148-114.274 | 0.038/0.027 |  |
| Ti90Al7V3 | 90 | 112.612 | 4.428 | 25.705 | 2.15 | 94.723-114.408 | 0.038/0.034 |  |
| Ti90Al1V9 | 90 | 116.929 | 4.608 | 25.604 | 1.81 | 94.523-114.023 | 0.038/0.028 |  |
| Ti90Al6V4 | 90 | 113.316 | 4.457 | 25.568 | 1.13 | 94.76-114.036 | 0.061/0.032 |  |
| Ti90Al8V2 | 90 | 111.913 | 4.398 | 25.541 | 0.75 | 94.928-113.91 | 0.036/0.027 |  |
| Ti90V8Mo2 | 90 | 120.261 | 4.716 | 25.538 | 0.29 | 95.845-115.221 | 0.035/0.026 |  |
| Ti90V9Zr1 | 90 | 117.223 | 4.652 | 25.528 | 2.61 | 94.583-113.914 | 0.035/0.028 |  |
| Ti85Al11V4 | 85 | 110.414 | 4.371 | 25.514 | 2.0 | 94.697-115.501 | 0.049/0.027 | KR |
| Ti85V11Cr4 | 85 | 122.659 | 4.729 | 25.472 | 3.59 | 95.358-116.455 | 0.043/0.031 | K |
| Ti86Al10V4 | 86 | 110.986 | 4.388 | 25.416 | 0.97 | 96.71-114.454 | 0.085/0.029 | R |
| Ti90Al2V8 | 90 | 116.194 | 4.578 | 25.414 | 0.26 | 94.624-113.934 | 0.046/0.026 |  |
| Ti90Al9Cr1 | 90 | 112.271 | 4.374 | 25.373 | 2.3 | 94.47-113.842 | 0.055/0.028 |  |
| Ti90V4Cr4Fe2 | 90 | 123.308 | 4.683 | 25.36 | 7.37 | 91.909-114.422 | 0.054/0.033 | K |
| Ti90Al8V1Cr1 | 90 | 112.968 | 4.403 | 25.349 | 2.4 | 90.841-112.593 | 0.046/0.034 | K |

## Limitations
- Screening under validated guardrails is NOT validation; shortlisted candidates require roadmap Tests I-M (CALPHAD phase map / beta-transus, interstitial O/N/H/C bounds, processing-conditioned properties, blind experimental holdout, red-team fail-safe) before any experimental claim.
- Hull values are metastability context signals only (Aykol 2018; Sun 2016).
- E and rho come from the elemental-mixing proxy; CrabNet E is the cross-check; K is quoted as a range.

## References
Wang et al. npj Comput. Mater. 7, 77 (2021); Dunn et al. npj Comput. Mater. 6, 138 (2020); Aykol et al. Sci. Adv. 4, eaaq0148 (2018); Sun et al. Sci. Adv. 2, e1600225 (2016); Hill Proc. Phys. Soc. A 65, 349 (1952); Pugh Philos. Mag. 45, 823 (1954); CRC Handbook 95th ed.; TIMET datasheets.