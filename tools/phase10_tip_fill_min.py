#!/usr/bin/env python3
import csv, json, re, subprocess, sys, os
from pathlib import Path

OUT = "outputs/phase10/boundary_scan_wide.csv"
Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)
if not os.path.exists(OUT):
    with open(OUT,"w",newline="") as f:
        csv.writer(f).writerow(["omega2","seed","Kphi_min_est","sync_at_est","target","status"])

# targets to fill (tip region)
W2S   = [2.76, 2.78, 2.80, 2.82]
SEEDS = [0, 1]
TARGET= 0.95
TOL   = 0.002
K_LO, K_HI, K_MAX, GROW, MAX_TRIES = 0.10, 0.90, 2.00, 1.6, 14
STEPS, BURN = 15000, 300

# resume set
done=set()
with open(OUT, newline="") as f:
    for r in csv.DictReader(f):
        try:
            w2=round(float(r.get("omega2","")),5)
            sd=int(r.get("seed",""))
            status=(r.get("status","") or "").strip().lower()
            k=float(r.get("Kphi_min_est","nan"))
            if status in ("ok","at_lower") and (k==k): # finite
                done.add((w2,sd))
        except: pass

def parse_sync(s: str) -> float:
    s=s.strip()
    try: return float(json.loads(s)["metrics"]["sync_index"])
    except: pass
    for m in reversed(list(re.finditer(r"\{.*?\}", s, flags=re.DOTALL))):
        try:
            o=json.loads(m.group(0))
            if "metrics" in o and "sync_index" in o["metrics"]:
                return float(o["metrics"]["sync_index"])
        except: pass
    raise ValueError("no JSON metrics")

def s_at(w2, K, seed):
    cmd=[sys.executable,"tools/phase10_phasecouple_demo.py",
         "--preset","neuron","--omega2",f"{w2:.5f}",
         "--kv","0.10","--kx","0.20","--Kphi",f"{K:.6f}",
         "--eps","0.06","--adapt_every","15","--noise","0.01",
         "--steps",str(STEPS),"--burn_in",str(BURN),
         "--seed1",str(seed),"--seed2",str(seed+101),
         "--prefix",f"bdry_w{w2:.3f}_seed{seed}"]
    p=subprocess.run(cmd,capture_output=True,text=True)
    p.check_returncode()
    return parse_sync(p.stdout)

def bracket(w2, seed):
    slo=s_at(w2,K_LO,seed)
    if slo>=TARGET: return ("at_lower",(K_LO,slo))
    shi=s_at(w2,K_HI,seed)
    if shi>=TARGET: return ("bracket",(K_LO,slo),(K_HI,shi))
    k, s, t = K_HI, shi, 0
    while s<TARGET and k<K_MAX and t<MAX_TRIES:
        k=min(K_MAX,k*GROW); s=s_at(w2,k,seed); t+=1
    if s>=TARGET: return ("bracket",(K_LO,slo),(k,s))
    return ("fail",s)

def bisect(w2, seed, br):
    (kL,sL),(kH,sH)=br
    for _ in range(18):
        km=0.5*(kL+kH); sm=s_at(w2,km,seed)
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
                kind,*rest=bracket(w2,sd)
                if kind=="at_lower":
                    k,s=rest[0]
                    wr.writerow([f"{w2:.5f}",sd,f"{k:.6f}",f"{s:.6f}",f"{TARGET:.4f}","at_lower"]); f.flush()
                    print(f"[tip] w2={w2:.3f} seed={sd} -> at_lower  K≈{k:.6f}  s≈{s:.6f}")
                elif kind=="fail":
                    s=rest[0]
                    wr.writerow([f"{w2:.5f}",sd,"",f"{s:.6f}",f"{TARGET:.4f}","unreachable"]); f.flush()
                    print(f"[tip] w2={w2:.3f} seed={sd} -> unreachable  s≈{s:.6f}")
                else:
                    k,s=bisect(w2,sd,rest)
                    wr.writerow([f"{w2:.5f}",sd,f"{k:.6f}",f"{s:.6f}",f"{TARGET:.4f}","ok"]); f.flush()
                    print(f"[tip] w2={w2:.3f} seed={sd} -> ok         K≈{k:.6f}  s≈{s:.6f}")
            except subprocess.CalledProcessError as e:
                wr.writerow([f"{w2:.5f}",sd,"","",f"{TARGET:.4f}",f"subprocess_error:{e.returncode}"]); f.flush()
                print(f"[tip] w2={w2:.3f} seed={sd} -> subprocess_error:{e.returncode}")
            except Exception as e:
                wr.writerow([f"{w2:.5f}",sd,"","",f"{TARGET:.4f}",f"error:{type(e).__name__}"]); f.flush()
                print(f"[tip] w2={w2:.3f} seed={sd} -> error:{type(e).__name__}")
print("[tip] done.")
