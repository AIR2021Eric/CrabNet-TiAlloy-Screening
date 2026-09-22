"""
plot_benchmark_v2.py
Redraw the Slide-18 visuals from benchmark_results_v2.csv with text boxes
placed in provably empty regions so nothing overlaps curves or markers.

Placement logic:
  LEFT  (throughput vs chunk): after the first tick, every point stays high,
        so a box pinned to the LOWER band (right-aligned) cannot touch the
        curve or the star (star lives at the top).
  RIGHT (time vs N): the data+fit form a rising diagonal; the bottom-right
        corner is below the line everywhere, and the top-left corner is above
        the line's start -> legend there.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "benchmark_results_v2.csv"
OUT = "slide18_throughput_visuals.png"

df = pd.read_csv(CSV)
sweep = df[df["sweep"] == "chunk_sweep"].sort_values("chunk_size").reset_index(drop=True)
lin = df[df["sweep"] == "linearity"].sort_values("N_run").reset_index(drop=True)

best = sweep.loc[sweep["throughput_sps"].idxmax()]
best_chunk, best_thr = int(best["chunk_size"]), float(best["throughput_sps"])
last_thr = float(sweep.iloc[-1]["throughput_sps"])
n_sweep = int(sweep["N_run"].iloc[0])
chunk_lin = int(lin["chunk_size"].iloc[0]) if len(lin) else best_chunk

verdict = (f"TRUE OPTIMUM at chunk={best_chunk:,} (throughput declines beyond it)"
           if last_thr < best_thr * 0.97 else
           f"SATURATED beyond ~10k (plateau ~{best_thr:.0f} samples/s)")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ---------------- LEFT: throughput vs chunk size ----------------
ax1.plot(sweep["chunk_size"], sweep["throughput_sps"],
         marker="o", lw=2.5, color="#1f77b4", zorder=3)
ax1.plot(best_chunk, best_thr, "*", ms=20, color="red", zorder=4)
ax1.set_xscale("log")
ax1.set_xticks(sweep["chunk_size"].unique())
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.tick_params(axis="x", labelrotation=45)
ax1.set_ylim(top=best_thr * 1.08)          # headroom so the star is never clipped
ax1.set_xlabel("Chunk Size (Batch Size)")
ax1.set_ylabel("Throughput (samples / sec)")
ax1.set_title(f"Throughput vs Chunk Size  (N = {n_sweep:,})")
ax1.grid(ls="--", alpha=.6, zorder=0)

# info box in the empty LOWER band, right-aligned
ax1.text(0.98, 0.04,
         f"best in tested range: {best_thr:.0f} samples/s @ chunk {best_chunk:,}\n{verdict}",
         transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=11, fontweight="bold", zorder=5,
         bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="k"))

# ---------------- RIGHT: inference time vs sample count ----------------
if len(lin):
    x = lin["N_run"].values.astype(float)
    y = lin["inf_time_s"].values.astype(float)
    ax2.plot(x, y, marker="D", lw=2.5, color="#d62728",
             label="Actual inference time", zorder=3)
    z = np.polyfit(x, y, 1); p = np.poly1d(z)
    r2 = 1 - np.sum((y - p(x)) ** 2) / np.sum((y - y.mean()) ** 2)
    xs = np.linspace(x.min(), x.max(), 100)
    ax2.plot(xs, p(xs), "--", color="gray", lw=2,
             label=f"Linear fit (R$^2$ = {r2:.4f})", zorder=2)

    _leg = ax2.legend(loc="upper left", framealpha=0.95)   # above diagonal start
    _leg.set_zorder(5)                                     # version-safe z-order

    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}k"))
    ax2.set_xlabel("Total Samples (N)")
    ax2.set_ylabel("Inference Time (seconds)")
    ax2.set_title(f"Inference Time vs Sample Count  (chunk = {chunk_lin:,})")
    ax2.grid(ls="--", alpha=.6, zorder=0)

    # extrapolation box in the empty BOTTOM-RIGHT corner, below the diagonal
    ax2.text(0.98, 0.04,
             f"extrapolation to 1M samples:\n≈ {p(1_000_000) / 60:.0f} min @ chunk {chunk_lin:,}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=11, fontweight="bold", zorder=5,
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="k"))

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"[plot] saved {OUT}")
plt.show()