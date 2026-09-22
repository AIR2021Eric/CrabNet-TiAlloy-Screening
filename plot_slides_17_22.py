"""
plot_slides_17_22.py
Generates:
  slide17_version_forensics.png  - head-comparison bar + hull-vintage grouped bars
  slide22_screening_results.png  - funnel + top-10 (reproducibility) + solute prevalence
Live data where possible (SHA-256 of both weight dirs; top_candidates.csv),
with documented fallbacks from the archived reports if files are missing.
"""
import os, hashlib, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})

# ================= SLIDE 17 data =================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

V2_DIR = "models/trained_models"
V1_CANDIDATES = ["models/trained_models_v1_active_backup",
                 "models/trained_models_v1_2020-12-04_patched"]
v1_dir = next((d for d in V1_CANDIDATES if os.path.isdir(d)), None)

if v1_dir and os.path.isdir(V2_DIR):
    names = sorted(n for n in set(os.listdir(V2_DIR)) & set(os.listdir(v1_dir))
                   if n.endswith(".pth"))
    identical = [n for n in names if sha256(os.path.join(V2_DIR, n)) == sha256(os.path.join(v1_dir, n))]
    retrained = [n for n in names if n not in identical]
    print(f"[live SHA-256] {len(identical)} identical / {len(retrained)} retrained of {len(names)}")
else:
    print("[warn] weight dirs not found - using documented forensic values")
    identical = [f"head_{i}" for i in range(74)]
    retrained = ["mp_e_hull.pth", "mp_bulk_modulus.pth", "mp_shear_modulus.pth", "mp_mu_b.pth",
                 "mp_elastic_anisotropy.pth", "OQMD_Bandgap.pth", "OQMD_Energy_per_atom.pth",
                 "OQMD_Formation_Enthalpy.pth", "OQMD_Volume_per_atom.pth"]

n_id, n_re = len(identical), len(retrained)
# hull per alloy from validation report (testH): v1 backup / v2 active
ALLOYS = ["Ti-6-4V", "Ti-5553", "Ti-6246", "Ti-10-2-3"]
HULL_V1 = [0.071, 0.057, 0.060, 0.064]
HULL_V2 = [0.026, 0.028, 0.030, 0.043]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel 1: 74 identical | 9 retrained ---
ax1.barh([0], [n_id], color="#2ca02c", height=0.45)
ax1.barh([0], [n_re], left=[n_id], color="#d62728", height=0.45)
ax1.text(n_id / 2, 0, f"{n_id}\nidentical", ha="center", va="center",
         color="w", fontweight="bold", fontsize=13)
ax1.text(n_id + n_re / 2, 0.32, f"{n_re}", ha="center", va="bottom",
         color="#d62728", fontweight="bold", fontsize=13)
ax1.set_xlim(0, n_id + n_re)
ax1.set_ylim(-0.95, 0.75)
ax1.set_yticks([])
ax1.set_xlabel("checkpoint heads (83 total)")
ax1.set_title("SHA-256 comparison: 2020-12-04 vs 2021-02-23 weights")
ax1.grid(axis="x", ls="--", alpha=.5)
list_txt = textwrap.fill("retrained: " + ", ".join(n.replace(".pth", "") for n in retrained), 70)
ax1.text(0.5, -0.55, list_txt, transform=ax1.transAxes, ha="center", va="top",
         fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="k"))

# --- Panel 2: hull vintage sensitivity ---
x = np.arange(len(ALLOYS)); w = 0.35
ax2.bar(x - w / 2, HULL_V1, w, label="v1 2020-12-04 (backup)", color="#1f77b4")
ax2.bar(x + w / 2, HULL_V2, w, label="v2 2021-02-23 (active)", color="#ff7f0e")
ax2.axhline(0.10, ls="--", color="k", lw=1.5)
ax2.text(len(ALLOYS) - 0.45, 0.103, "dual-vintage gate: 0.10 eV/atom",
         ha="right", va="bottom", fontsize=10)
ax2.set_xticks(x); ax2.set_xticklabels(ALLOYS)
ax2.set_ylim(0, 0.125)
ax2.set_ylabel("energy above hull (eV/atom)")
ax2.set_title("Hull vintage sensitivity\n(mean 0.063 → 0.032 eV/atom)")
ax2.legend(loc="upper left", framealpha=0.95)
ax2.grid(axis="y", ls="--", alpha=.5)
plt.tight_layout()
plt.savefig("slide17_version_forensics.png", dpi=300, bbox_inches="tight")
print("[plot] saved slide17_version_forensics.png")
plt.close()

# ================= SLIDE 22 data =================
ELS = ["Al", "V", "Mo", "Zr", "Sn", "Cr", "Fe"]
FALLBACK_TOP = [("Ti85V15", 26.096), ("Ti90V9Fe1", 25.921), ("Ti90V9Cr1", 25.880),
                ("Ti90Al9V1", 25.879), ("Ti90V9Mo1", 25.811), ("Ti90V8Fe2", 25.780),
                ("Ti90V8Cr2", 25.754), ("Ti90Al7V3", 25.705), ("Ti90Al1V9", 25.604),
                ("Ti90Al6V4", 25.568)]
FALLBACK_MEAN = [5.6, 3.6, 0.6, 0.1, 0.0, 1.6, 1.1]
FALLBACK_MAX = [11, 15, 4, 3, 1, 4, 4]
CTRL = 25.416

try:
    df = pd.read_csv("screening_output/top_candidates.csv")
    top = df.head(10)
    formulas = top["formula"].tolist()
    vals = top["spec_stiff_cons"].tolist()
    prev = df.head(500)
    mean_at = [round(prev[f"{e}_at"].mean(), 1) for e in ELS]
    max_at = [int(prev[f"{e}_at"].max()) for e in ELS]
    c = df[df["formula"] == "Ti86Al10V4"]
    if len(c):
        CTRL = float(c["spec_stiff_cons"].iloc[0])
    print("[live] top_candidates.csv read")
except Exception:
    print("[warn] top_candidates.csv not found - using report values")
    formulas = [f for f, _ in FALLBACK_TOP]
    vals = [v for _, v in FALLBACK_TOP]
    mean_at, max_at = FALLBACK_MEAN, FALLBACK_MAX

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.2))

# --- Panel 1: funnel ---
stages = ["Generated", "pass_all gates", "rank_clean tier", "Top-500"]
counts = [39480, 13650, 6036, 500]
run2 = ["", "run2: 35.8%", "run2: 16.0%", ""]
y = np.arange(len(stages))[::-1]
ax1.barh(y, counts, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
for i, (v, r) in enumerate(zip(counts, run2)):
    ax1.text(v + 900, y[i], f"{v:,}" + (f"   ({r})" if r else ""), va="center", fontsize=10)
ax1.set_yticks(y); ax1.set_yticklabels(stages)
ax1.set_xlim(0, 39480 * 1.55)
ax1.set_title("Screening funnel\n(run 1: 39,480 candidates; run-2 rates in parentheses)")
ax1.grid(axis="x", ls="--", alpha=.5)

# --- Panel 2: top-10 + control (reproducibility in title) ---
y2 = np.arange(len(vals))[::-1]
ax2.barh(y2, vals, color="#2ca02c")
for v, yy in zip(vals, y2):
    ax2.text(v + 0.08, yy, f"{v:.2f}", va="center", fontsize=9)
ax2.axvline(CTRL, ls="--", color="k", lw=1.5)
ax2.text(CTRL, len(vals) - 0.2, f" Ti-6-4V control {CTRL:.2f}", va="bottom", ha="left",
         fontsize=9, rotation=90)
ax2.set_yticks(y2); ax2.set_yticklabels(formulas, fontsize=9)
ax2.set_xlim(0, 29)
ax2.set_xlabel("consensus specific stiffness (GPa·cm³/g)")
ax2.set_title("Top-10 candidates\nidentical in both runs (Δ = 0.000)")
ax2.grid(axis="x", ls="--", alpha=.5)

# --- Panel 3: solute prevalence ---
ax3.bar(ELS, mean_at, color="#4c72b0")
for i, (m, mx) in enumerate(zip(mean_at, max_at)):
    ax3.text(i, m + 0.25, f"{m:.1f}\n(max {mx})", ha="center", va="bottom", fontsize=9)
ax3.set_ylim(0, 8.5)
ax3.set_ylabel("mean at% in Top-500")
ax3.set_title("Solute prevalence in Top-500\n(V-dominated β / α+β families)")
ax3.grid(axis="y", ls="--", alpha=.5)
plt.tight_layout()
plt.savefig("slide22_screening_results.png", dpi=300, bbox_inches="tight")
print("[plot] saved slide22_screening_results.png")