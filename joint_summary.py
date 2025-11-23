import json, csv
def best_from_sn(path):
  d=json.load(open(path)); return ("SN", d["O0"], d["O1"], d["chi2"]/max(d["dof"],1))
def best_from_bao(path):
  with open(path) as f:
    r=csv.DictReader(f)
    rows=sorted(r, key=lambda x: float(x["AICc"]))
  b=rows[0]; return ("BAO", float(b["O0"]), float(b["O1"]), float(b["chi2"])/float(b["dof"]))
print(*best_from_sn("outputs/sn_fit_zhd_diag_hf_refine_summary.json"))
print(*best_from_bao("outputs/bao_fit_desi_base_noz_results.csv"))
