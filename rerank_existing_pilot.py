import pandas as pd
df = pd.read_csv("screening_output/screening_results.csv")
df["gap_pct"] = (df["spec_stiff"] - df["spec_stiff_nn"]).abs() / df["spec_stiff"] * 100
clean = df[df["pass_all"] & (df["Cr_at"] <= 4) & (df["Fe_at"] <= 4)
           & (df["Al_at"] <= 11) & (df["gap_pct"] < 10)].copy()
clean["spec_stiff_cons"] = ((clean["spec_stiff"] + clean["spec_stiff_nn"]) / 2).round(3)
clean = clean.sort_values("spec_stiff_cons", ascending=False)
print(clean.head(20)[["formula","Ti_at","Al_at","V_at","Mo_at","Zr_at","Sn_at",
                      "Cr_at","Fe_at","spec_stiff","spec_stiff_nn","spec_stiff_cons",
                      "hull_v1","hull_v2"]])