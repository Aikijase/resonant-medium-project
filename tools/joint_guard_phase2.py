import json, os
# include ISW amplitude scores if present
try:
    isw_amp = json.load(open("outputs/phase2/isw_score.json"))
    summ.update({
      "isw_amp_WWI":  isw_amp.get("WWI"),
      "isw_amp_AIC":  isw_amp.get("AIC",{}).get("value"),
      "isw_amp_dAIC": isw_amp.get("AIC",{}).get("delta"),
    })
except Exception:
    pass
import json, os
def safe(f):
    try: return json.load(open(f))
    except: return {}
phase1 = safe("outputs/phase1/pred_eval.phase1.json")
fs8    = safe("outputs/phase2/fs8_eval.json")
isw    = safe("outputs/phase2/isw_eval.json")
def wwi(dA,dB):
    if dA is None or dB is None: return None
    return 100*min(1.0, max(0,-dA/10.0), max(0,-dB/10.0))
summ = {
  "phase1_WWI": wwi(phase1.get("AIC",{}).get("delta"), phase1.get("BIC",{}).get("delta")),
  "fs8_WWI":    fs8.get("WWI"),
  "isw_info":   "ISW placeholder only (no data yet)",
}
os.makedirs("outputs/phase2", exist_ok=True)
json.dump(summ, open("outputs/phase2/pred_eval.phase2_joint.json","w"), indent=2)
print("Wrote outputs/phase2/pred_eval.phase2_joint.json")
