import pandas as pd
t = pd.read_csv("screening_output/top_candidates.csv")
print("red-team fraction in top-100:", t["flag_red_team"].mean())
print(t.head(20)[["formula","Ti_at","Al_at","Cr_at","Fe_at","spec_stiff","spec_stiff_nn","flag_red_team"]])