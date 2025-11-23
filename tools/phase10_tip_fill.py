#!/usr/bin/env python3
import csv, json, re, subprocess, sys, os
from pathlib import Path

OUT = "outputs/phase10/boundary_scan_wide.csv"
Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)
if not os.path.exists(OUT):
    with open(OUT,"w",newline="") as f:
        csv.writer(f).writerow(["omega2","seed","Kphi_min_est","sync_at_est","target","status"])

# quick tip coverage
W2S   = [2.76, 2.78, 2.80, 2.82]
SEEDS = [0, 1]
TARGET= 0.95
TOL   = 0.002
K_LO, K_HI, K_MAX, GROW, MAX_TRIES = 0.10, 0.90, 1.80, 1.6, 14
STEPS, BURN = 15000, 300

# resume set
done=set()
with open(OUT, newline="") as f:
    for r in csv.DictReader(f):
        try: done.add((round(float(r["omega2"]),5), int(r["seed"])))
        except: pass

def parse_sync(s: str) -> float:
    s=s.strip()
    try: return float(json.loads(s)["metrics"]["sync_index"])
    except Exception: pass
    for m in reversed(list(re.finditer(r"\{.*?\}", s, flags=re.DOTALL))):
        try:
            obj=json.loads(m.group(0))
            if "metrics" in obj and "sync_index" in obj["metrics"]:
                return float(obj["metrics"]["sync_index"])
        except Exception: pass
    raise ValueError("no JSON metrics in stdout")

def sync_index_at(w2, K, seed):
    cmd=[sys.executable,"tools/phase10_phasecouple_demo.py",
         "--preset","neuron","--omega2",f"{w2:.5f}",
         "--kv","0.10","--kx","0.20","--Kphi",f"{K:.6f}",
         "--eps","0.06","--adapt_every","15","--noise","0.01",
         "--steps",str(STEPS),"--burn_in",str(BURN),
         "--seed1",str(seed),"--seed2",str(seed+101),
         "--prefix",f"bdry_w{w2:.3f}_seed{seed}"]
    p=subprocess.run(cmd, capture_output=True, text=True)
    p.check_returncode()
    return parse_sync(p.stdout)

def ensure_bracket(w2, seed):
    s_lo=sync_index_at(w2,K_LO,seed)
    if s_lo>=TARGET: return ("at_lower",(K_LO,s_lo))
    s_hi=sync_index_at(w2,K_HI,seed)
    if s_hi>=TARGET: return ("bracket",(K_LO,s_lo),(K_HI,s_hi))
    k,s,tries=K_HI,s_hi,0
    while s<TARGET and k<K_MAX and tries<MAX_TRIES:
        k=min(K_MAX,k*GROW); s=sync_index_at(w2,k,seed); tries+=1
    if s>=TARGET: return ("bracket",(K_LO,s_lo),(k,s))
    return ("fail",s)

def bisect_boundary(w2, seed, br):
    (kL,sL),(kH,sH)=br
    for _ in range(18):
        km=0.5*(kL+kH); sm=sync_index_at(w2,km,seed)
        if abs(sm-TARGET)<=TOL: return km,sm
        if sm>=TARGET: kH,sH=km,sm
        else: kL,sL=km,sm
    return km,sm

with open(OUT,"a",newline="") as f:
    wr=csv.writer(f)
    for w2 in W2S:
        for sd in SEEDS:
            key=(round(w2,5),sd)
            if key in done: 
                continue
            try:
                kind,*rest=ensure_bracket(w2,sd)
                if kind=="at_lower":
                    k_est,s_est=rest[0]
                    wr.writerow([f"{w2:.5f}",sd,f"{k_est:.6f}",f"{s_est:.6f}",f"{TARGET:.4f}","at_lower"]); f.flush()
                    print(f"[tip] w2={w2:.3f} seed={sd} -> at_lower  K≈{k_est:.6f}  s≈{s_est:.6f}")
                elif kind=="fail":
                    s_last=rest[0]
                    wr.writerow([f"{w2:.5f}",sd,"",f"{s_last:.6f}",f"{TARGET:.4f}","unreachable"]); f.flush()
                    print(f"[tip] w2={w2:.3f} seed={sd} -> unreachable  s≈{s_last:.6f}")
                else:
                    k_est,s_est=bisect_boundary(w2,sd,rest)
                    wr.writerow([f"{w2:.5f}",sd,f"{k_est:.6f}",f"{s_est:.6f}",f"{TARGET:.4f}","ok"]); f.flush()
                    print(f"[tip] w2={w2:.3f} seed={sd} -> ok         K≈{k_est:.6f}  s≈{s_est:.6f}")
            except subprocess.CalledProcessError as e:
                wr.writerow([f"{w2:.5f}",sd,"","",f"{TARGET:.4f}",f"subprocess_error:{e.returncode}"]); f.flush()
                print(f"[tip] w2={w2:.3f} seed={sd} -> subprocess_error:{e.returncode}")
            except Exception as e:
                wr.writerow([f"{w2:.5f}",sd,"","",f"{TARGET:.4f}",f"error:{type(e).__name__}"]); f.flush()
                print(f"[tip] w2={w2:.3f} seed={sd} -> error:{type(e).__name__}")
print("[tip] done.")
