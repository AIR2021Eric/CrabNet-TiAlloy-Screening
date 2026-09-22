# Ti-Alloy High-Throughput Screening Report

Date: 2026-09-18 | seed: 42 | candidates: 20004 | chunk: 10000

Weights: 2021-02-23 mat2vec (bsz_auto), Zenodo 10.5281/zenodo.4633866; 74/83 heads byte-identical to the 2020-12-04 set (all elasticity/AFLOW); 9 MP/OQMD single-model heads retrained in the 2021 run (incl. mp_e_hull, mp_bulk_modulus)

K-source policy: Bulk modulus is reported as a range across heads (matbench-weights, AFLOW, MP, elemental proxy); the matbench-weights head is a systematic low outlier for Ti-rich alloys and is never used alone.

Gates: hull < 0.1 eV/atom under BOTH vintages; |E_nn-E_proxy| < 20.0%; rho in (3.8, 4.8) g/cm3.
Flags: K spread > 20.0 GPa; K fold-std > 8.0 GPa; Pugh proxy < 1.75; red-team {'Al': 9, 'Cr': 5, 'Fe': 5} at%.

## Gate survival

| gate | pass rate |
|---|---|
| gate_hull_both | 0.984 |
| gate_consensus | 0.816 |
| gate_rho | 0.442 |
| pass_all | 0.417 |

## Flag counts

| flag | count |
|---|---|
| flag_k_spread | 19816 |
| flag_k_foldstd | 0 |
| flag_pugh | 0 |
| flag_red_team | 6286 |

## Positive controls (must pass)

| formula | hull_v1 | hull_v2 | E_proxy | E_nn | diff% | pass_all |
|---|---|---|---|---|---|---|
| Ti86Al10V4 | 0.085 | 0.029 | 110.986 | 112.068 | 0.97 | True |
| Ti81Al9Mo2V5Cr3 | 0.057 | 0.028 | 117.953 | 101.384 | 14.05 | True |
| Ti83Al11Sn1Zr2Mo3 | 0.062 | 0.03 | 112.393 | 105.933 | 5.75 | True |
| Ti84V9Fe2Al5 | 0.063 | 0.04 | 115.96 | 111.78 | 3.6 | True |


## Top 20 by specific stiffness (E_proxy / rho_proxy)

| formula | Ti_at | E_proxy | rho_proxy | spec_stiff | E_nn | K_range | hull_v1/v2 | flags |
|---|---|---|---|---|---|---|---|---|
| Ti90Cr8Fe2 | 90 | 127.697 | 4.707 | 27.129 | 112.643 | 95.021-114.526 | 0.039/0.036 | R |
| Ti85Al2Cr8Fe5 | 85 | 128.688 | 4.744 | 27.126 | 103.581 | 91.136-115.078 | 0.088/0.038 | KR |
| Ti85Al1V1Cr8Fe5 | 85 | 129.487 | 4.776 | 27.112 | 106.199 | 88.693-115.949 | 0.088/0.038 | KR |
| Ti90Mo1Cr8Fe1 | 90 | 128.366 | 4.736 | 27.104 | 110.896 | 91.436-114.821 | 0.053/0.037 | KR |
| Ti85Al2Mo1Cr8Fe4 | 85 | 129.357 | 4.773 | 27.102 | 106.175 | 89.159-115.478 | 0.075/0.035 | KR |
| Ti90Mo2Cr8 | 90 | 129.035 | 4.765 | 27.08 | 112.123 | 95.616-115.221 | 0.037/0.035 | R |
| Ti90Al1Cr8Fe1 | 90 | 126.27 | 4.666 | 27.062 | 108.212 | 90.603-113.486 | 0.05/0.036 | KR |
| Ti85Al3Cr8Fe4 | 85 | 127.262 | 4.703 | 27.06 | 104.335 | 90.939-115.202 | 0.083/0.036 | KR |
| Ti90V1Cr8Fe1 | 90 | 127.056 | 4.697 | 27.05 | 112.59 | 91.16-114.348 | 0.048/0.037 | KR |
| Ti85Al2V1Cr8Fe4 | 85 | 128.048 | 4.734 | 27.049 | 107.193 | 88.999-115.004 | 0.078/0.036 | KR |
| Ti85Al1V2Cr8Fe4 | 85 | 128.842 | 4.765 | 27.039 | 108.019 | 88.942-115.875 | 0.073/0.037 | KR |
| Ti90Al1Mo1Cr8 | 90 | 126.936 | 4.695 | 27.036 | 108.32 | 90.914-113.883 | 0.049/0.034 | KR |
| Ti85V3Cr8Fe4 | 85 | 129.644 | 4.797 | 27.026 | 112.086 | 91.416-116.753 | 0.08/0.039 | KR |
| Ti90V1Mo1Cr8 | 90 | 127.724 | 4.726 | 27.026 | 112.605 | 91.441-114.747 | 0.05/0.034 | KR |
| Ti85Al3Mo2Cr8Fe2 | 85 | 128.594 | 4.76 | 27.016 | 105.479 | 89.157-114.937 | 0.058/0.03 | KR |
| Ti90Cr7Fe3 | 90 | 127.239 | 4.712 | 27.003 | 109.688 | 94.725-115.453 | 0.056/0.046 | KR |
| Ti85Al2Cr7Fe6 | 85 | 128.231 | 4.749 | 27.002 | 102.867 | 91.081-115.306 | 0.084/0.037 | KR |
| Ti85Al4Cr8Fe3 | 85 | 125.851 | 4.661 | 27.001 | 103.484 | 90.883-115.631 | 0.065/0.034 | KR |
| Ti90Al2Cr8 | 90 | 124.859 | 4.625 | 26.997 | 107.557 | 94.374-113.682 | 0.052/0.028 | R |
| Ti85Al1V1Cr7Fe6 | 85 | 129.027 | 4.78 | 26.993 | 106.969 | 88.678-116.023 | 0.088/0.038 | KR |

## Limitations
- Screening under validated guardrails is NOT validation; top candidates require roadmap Tests I-M (CALPHAD phase map / beta-transus, interstitial O/N/H/C bounds, processing-conditioned properties, blind experimental holdout, red-team fail-safe) before any experimental claim.
- Hull values are metastability context signals only (Aykol 2018; Sun 2016).
- E and rho come from the elemental-mixing proxy (chemical heuristic); CrabNet E is a cross-check.

## References
Wang et al. npj Comput. Mater. 7, 77 (2021); Dunn et al. npj Comput. Mater. 6, 138 (2020); Aykol et al. Sci. Adv. 4, eaaq0148 (2018); Sun et al. Sci. Adv. 2, e1600225 (2016); Hill Proc. Phys. Soc. A 65, 349 (1952); Pugh Philos. Mag. 45, 823 (1954); CRC Handbook 95th ed.; TIMET datasheets.