import pandas as pd

df = pd.read_csv("screening_output/top_candidates.csv")

cols = ["formula", "Ti_at", "Al_at", "V_at", "Mo_at", "Zr_at", "Sn_at", "Cr_at", "Fe_at",
        "spec_stiff_cons", "E_consensus_diff_pct", "flag_red_team"]

# fail loudly with a helpful message if the schema ever changes again
missing = [c for c in cols if c not in df.columns]
if missing:
    raise KeyError(f"columns not in top_candidates.csv: {missing}\n"
                   f"available columns: {list(df.columns)}")

print("Top 20 by consensus specific stiffness:")
print(df.head(20)[cols].to_string(index=False))

top = df.head(500)
print("\nSolute prevalence in Top 500 (at%):")
for col in ["Al_at", "V_at", "Mo_at", "Zr_at", "Sn_at", "Cr_at", "Fe_at"]:
    print(f"  {col:>5}: mean={top[col].mean():5.1f}  max={top[col].max()}")

print(f"\nRed-team fraction in top-500: {float(top['flag_red_team'].mean()):.2f}")
print(f"Mean proxy-NN gap in top-500:   {top['E_consensus_diff_pct'].mean():.2f} %")