# make_eval_set.py  (run inside CrabNet_TiAlloy_Test with its venv)
import pandas as pd
from generate_and_screen import generate_candidates
cand = generate_candidates(20000, 42)                 # reproducible: seed 42
pd.DataFrame({"formula": list(cand.keys())}).to_csv(r"..\comparison\eval_set_20k.csv", index=False)
pd.DataFrame({"formula": ["Ti86Al10V4", "Ti81Al9Mo2V5Cr3",
                          "Ti83Al11Sn1Zr2Mo3", "Ti84V9Fe2Al5"]}).to_csv(r"..\comparison\controls.csv", index=False)
print("eval set written:", len(cand), "formulas")