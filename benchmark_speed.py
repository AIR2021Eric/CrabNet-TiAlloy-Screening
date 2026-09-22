"""
benchmark_speed.py (v2) - extended chunk sweep + linearity check + integrated plotting.

What changed vs v1:
  * Chunk sweep extended to 40k (11 points) at fixed N = full real space, so we can
    distinguish "throughput saturates" from "there is a true optimum then decline".
  * Automatic verdict printed: TRUE OPTIMUM / SATURATED / STILL RISING.
  * Linearity check run only at the best chunk (saves time).
  * Plotting integrated; text boxes placed in empty axes regions (no overlaps):
      - left panel : info box top-left (empty triangle above rising curve), best point
        marked with a red star (no arrow crossing the curve)
      - right panel: legend top-left (empty), extrapolation box bottom-right (below line)
Outputs: benchmark_results_v2.csv + slide18_throughput_visuals.png
Usage:
  python benchmark_speed.py                       # full (~45 min, unattended OK)
  python benchmark_speed.py --skip-linearity      # left panel only (~30 min)
  python benchmark_speed.py --linearity-max 60000 # cheaper right panel
"""
import argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generate_and_screen import generate_candidates, infer_chunk

CHUNKS = [1000, 2000, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000, 40000]

def dummy_formulas(pool, n, seed=7):
    rng = np.random.default_rng(seed)
    return list(rng.choice(pool, size=n, replace=True))

def time_inference(formulas, chunk):
    t0 = time.time()
    for i in range(0, len(formulas), chunk):
        _ = infer_chunk(formulas[i:i + chunk])
    return time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-n", type=int, default=40000)
    ap.add_argument("--linearity-max", type=int, default=100000)
    ap.add_argument("--skip-linearity", action="store_true")
    ap.add_argument("--out-csv", default="benchmark_results_v2.csv")
    ap.add_argument("--out-png", default="slide18_throughput_visuals.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})

    print("[setup] generating real composition space ...")
    space = generate_candidates(args.sweep_n + 5000, 42)
    pool = list(space.keys())
    n_sweep = min(args.sweep_n, len(pool))
    sweep = pool[:n_sweep]
    print(f"        real space = {len(pool):,}; sweep N = {n_sweep:,}\n")

    # ---------------- Stage 1: extended chunk-size sweep ----------------
    rows = []
    chunks = [c for c in CHUNKS if c <= n_sweep]
    for c in chunks:
        t = time_inference(sweep, c)
        thr = n_sweep / t
        print(f"  chunk={c:>6,} | time={t:7.1f}s | throughput={thr:6.1f} samples/s")
        rows.append({"sweep": "chunk_sweep", "N_run": n_sweep, "chunk_size": c,
                     "inf_time_s": round(t, 2), "throughput_sps": round(thr, 2)})
    df_sweep = pd.DataFrame(rows)

    best = df_sweep.loc[df_sweep["throughput_sps"].idxmax()]
    best_chunk, best_thr = int(best["chunk_size"]), float(best["throughput_sps"])
    last_thr = float(df_sweep.iloc[-1]["throughput_sps"])
    t10 = df_sweep.loc[df_sweep.chunk_size == 10000, "throughput_sps"]
    t10 = float(t10.iloc[0]) if len(t10) else np.nan
    if last_thr < best_thr * 0.97:
        verdict = f"TRUE OPTIMUM at chunk={best_chunk:,} (throughput declines beyond it)"
    elif not np.isnan(t10) and (best_thr - t10) / t10 < 0.03:
        verdict = f"SATURATED: gain beyond 10k < 3%; plateau ~{best_thr:.0f} samples/s"
    else:
        verdict = f"STILL RISING at chunk={int(df_sweep.iloc[-1]['chunk_size']):,}; test larger chunks"
    print("\n[verdict]", verdict, "\n")

    # ---------------- Stage 2: linearity at the best chunk ----------------
    lin_rows = []
    if not args.skip_linearity:
        for n in sorted({10000, 20000, 40000, 80000, args.linearity_max}):
            f = pool[:n] if n <= len(pool) else dummy_formulas(pool, n)
            t = time_inference(f, best_chunk)
            print(f"  linearity N={n:>7,} | chunk={best_chunk:,} | time={t:7.1f}s")
            lin_rows.append({"sweep": "linearity", "N_run": n, "chunk_size": best_chunk,
                             "inf_time_s": round(t, 2), "throughput_sps": round(n / t, 2)})
    df_lin = pd.DataFrame(lin_rows)

    pd.concat([df_sweep, df_lin], ignore_index=True).to_csv(args.out_csv, index=False)

    # ---------------- Stage 3: integrated plotting (no overlaps) ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT: throughput vs chunk size. Info box in empty top-left triangle; best
    # point marked with a star (no arrow, so nothing crosses the curve).
    ax1.plot(df_sweep.chunk_size, df_sweep.throughput_sps, marker="o", lw=2.5, color="#1f77b4")
    ax1.plot(best_chunk, best_thr, "*", ms=20, color="red", zorder=5)
    ax1.set_xscale("log")
    ax1.set_xticks(chunks)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.set_xlabel("Chunk Size (Batch Size)")
    ax1.set_ylabel("Throughput (samples / sec)")
    ax1.set_title(f"Throughput vs Chunk Size  (N = {n_sweep:,})")
    ax1.grid(ls="--", alpha=.6)
    ax1.text(0.03, 0.97, f"best in tested range:\n{best_thr:.0f} samples/s @ chunk {best_chunk:,}\n\n{verdict}",
             transform=ax1.transAxes, va="top", ha="left", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="k"))

    # RIGHT: time vs N at best chunk. Legend top-left (empty), extrapolation box
    # bottom-right (below the rising line) - the two never touch.
    if len(df_lin):
        x, y = df_lin.N_run.values.astype(float), df_lin.inf_time_s.values.astype(float)
        ax2.plot(x, y, marker="D", lw=2.5, color="#d62728", label="Actual inference time")
        z = np.polyfit(x, y, 1); p = np.poly1d(z)
        r2 = 1 - np.sum((y - p(x)) ** 2) / np.sum((y - y.mean()) ** 2)
        xs = np.linspace(x.min(), x.max(), 100)
        ax2.plot(xs, p(xs), "--", color="gray", lw=2, label=f"Linear fit (R$^2$ = {r2:.4f})")
        ax2.legend(loc="upper left")
        ax2.text(0.97, 0.04, f"extrapolation to 1M samples:\n≈ {p(1_000_000) / 60:.0f} min "
                             f"@ chunk {best_chunk:,}",
                 transform=ax2.transAxes, ha="right", va="bottom", fontsize=11,
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="k"))
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}k"))
        ax2.set_xlabel("Total Samples (N)")
        ax2.set_ylabel("Inference Time (seconds)")
        ax2.set_title(f"Inference Time vs Sample Count  (chunk = {best_chunk:,})")
        ax2.grid(ls="--", alpha=.6)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    print(f"\n[plot] saved {args.out_png} | [data] saved {args.out_csv}")
    plt.show()

if __name__ == "__main__":
    main()