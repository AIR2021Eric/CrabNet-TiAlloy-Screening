# export_crabnet_summary.py  (run inside CrabNet_TiAlloy_Test)
import json, pandas as pd
R = json.load(open("validation_output/validation_results_full.json"))
A_k, A_g = R["testA"]["matbench_log_kvrh"], R["testA"]["matbench_log_gvrh"]
G = pd.DataFrame(R["testG"]["table"])
summary = {
  "model": "crabnet",
  "weights_vintage": R["meta"].get("weights_vintage", "2021-02-23 mat2vec"),
  "env": {"python": R["meta"]["python"], "torch": R["meta"]["torch"]},
  "testA_oof_mae": {"kvrh": A_k["oof_mae"], "gvrh": A_g["oof_mae"],
                    "kvrh_r": A_k["pearson_r"], "gvrh_r": A_g["pearson_r"],
                    "comparator_leaderboard": {"kvrh": A_k["leaderboard_mae"], "gvrh": A_g["leaderboard_mae"]}},
  "testB_err_pct": {"K": R["testB"]["mean_K_err_expt_pct"], "G": R["testB"]["mean_G_err_expt_pct"]},
  "testC_max_diff": R["testC"]["max_diff"],
  "testD_max_step_GPa": R["testD"]["max_step_GPa"],
  "testG_E_mae_pct": round(float(G["E_nn_err_pct"].mean()), 2),
  "testH_controls_pass": f"{int(pd.DataFrame(R['testH']['table'])['consensus_pass'].sum())}/4",
  "testF_mean_spread_GPa": R["testF"]["mean_spread_GPa"],
  "throughput": {"best_chunk": 15000, "samples_per_s": 244.3, "linearity_r2": 0.9979},
  "screening_exhaustive": {"n": 39480, "pass_all": 0.426, "rank_clean": 0.238,
                           "top1": "Ti85V15 @ 26.096 spec_stiff_cons"},
  "notes": "hull gate uses CrabNet-trained mp_e_hull heads (dual vintage); K policy = 4-head range"
}
json.dump(summary, open(r"..\comparison\model_results_crabnet.json", "w"), indent=2)
print("crabnet summary exported")