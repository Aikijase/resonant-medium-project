#!/usr/bin/env python3
import argparse, csv, json, re, subprocess, sys
from pathlib import Path

def parse_sync(stdout: str) -> float:
    s = stdout.strip()
    # Try clean JSON first
    try:
        return float(json.loads(s)["metrics"]["sync_index"])
    except Exception:
        pass
    # Fallback: grab the last JSON object from mixed stdout
    for m in reversed(list(re.finditer(r"\{.*?\}", s, flags=re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
            if "metrics" in obj and "sync_index" in obj["metrics"]:
                return float(obj["metrics"]["sync_index"])
        except Exception:
            continue
    raise ValueError("Could not parse JSON metrics from subprocess stdout.")

def sync_index_at(w2, K, a):
    cmd = [sys.executable, "tools/phase10_phasecouple_demo.py",
           "--preset", a.preset, "--omega2", f"{w2:.5f}",
           "--kv", a.kv, "--kx", a.kx, "--Kphi", f"{K:.6f}",
           "--eps", a.eps, "--adapt_every", a.adapt_every, "--noise", a.noise,
           "--steps", str(a.steps), "--burn_in", str(a.burn_in),
           "--seed1", str(a.seed), "--seed2", str(a.seed+101),
           "--prefix", f"bdry_w{w2:.3f}_seed{a.seed}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    p.check_returncode()
    return parse_sync(p.stdout)

def ensure_bracket(w2, target, a):
    s_lo = sync_index_at(w2, a.k_lo, a)
    if s_lo >= target: return ("at_lower", (a.k_lo, s_lo))
    s_hi = sync_index_at(w2, a.k_hi, a)
    if s_hi >= target: return ("bracket", (a.k_lo, s_lo), (a.k_hi, s_hi))
    k, s, tries = a.k_hi, s_hi, 0
    while s < target and k < a.max_k and tries < a.max_grow_tries:
        k = min(a.max_k, k * a.grow)
        s = sync_index_at(w2, k, a)
        tries += 1
    if s >= target: return ("bracket", (a.k_lo, s_lo), (k, s))
    return ("fail", s)

def bisect_boundary(w2, target, bracket, a):
    (kL, sL), (kH, sH) = bracket
    for _ in range(a.max_iter):
        km = 0.5*(kL+kH)
        sm = sync_index_at(w2, km, a)
        if abs(sm - target) <= a.tol: return km, sm
        if sm >= target: kH, sH = km, sm
        else: kL, sL = km, sm
    return km, sm

def already_done(path):
    done=set(); p=Path(path)
    if p.exists():
        with p.open(newline="") as f:
            for row in csv.DictReader(f):
                try: done.add((round(float(row["omega2"]),5), int(row["seed"])))
                except: pass
    return done

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--w2-min", type=float, required=True)
    ap.add_argument("--w2-max", type=float, required=True)
    ap.add_argument("--w2-step", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--k-lo", type=float, default=0.10)
    ap.add_argument("--k-hi", type=float, default=0.90)
    ap.add_argument("--max-k", type=float, default=1.60)
    ap.add_argument("--grow", type=float, default=1.6)
    ap.add_argument("--max-grow-tries", type=int, default=14)
    ap.add_argument("--max-iter", type=int, default=18)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--preset", default="neuron")
    ap.add_argument("--kv", default="0.10"); ap.add_argument("--kx", default="0.20")
    ap.add_argument("--eps", default="0.06"); ap.add_argument("--noise", default="0.01")
    ap.add_argument("--adapt-every", dest="adapt_every", default="15")
    ap.add_argument("--out", default="outputs/phase10/boundary_scan_wide.csv")
    ap.add_argument("--seed", type=int, default=0)  # overwritten per loop
    a=ap.parse_args()

    outp=Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    if not outp.exists():
        with outp.open("w", newline="") as f:
            csv.writer(f).writerow(["omega2","seed","Kphi_min_est","sync_at_est","target","status"])

    done=already_done(a.out)
    with outp.open("a", newline="") as f:
        wr=csv.writer(f)
        w2=a.w2_min
        while w2 <= a.w2_max + 1e-12:
            for sd in a.seeds:
                if (round(w2,5), sd) in done: continue
                a.seed=sd
                try:
                    kind,*rest = ensure_bracket(w2, a.target, a)
                    if kind=="at_lower":
                        k_est,s_est=rest[0]
                        wr.writerow([f"{w2:.5f}", sd, f"{k_est:.6f}", f"{s_est:.6f}", f"{a.target:.4f}", "at_lower"]); f.flush()
                        print(f"[scan] w2={w2:.3f} seed={sd} -> at_lower  K≈{k_est:.6f}  s≈{s_est:.6f}")
                    elif kind=="fail":
                        s_last=rest[0]
                        wr.writerow([f"{w2:.5f}", sd, "", f"{s_last:.6f}", f"{a.target:.4f}", "unreachable"]); f.flush()
                        print(f"[scan] w2={w2:.3f} seed={sd} -> unreachable  s≈{s_last:.6f}")
                    else:
                        k_est,s_est=bisect_boundary(w2, a.target, rest, a)
                        wr.writerow([f"{w2:.5f}", sd, f"{k_est:.6f}", f"{s_est:.6f}", f"{a.target:.4f}", "ok"]); f.flush()
                        print(f"[scan] w2={w2:.3f} seed={sd} -> ok         K≈{k_est:.6f}  s≈{s_est:.6f}")
                except subprocess.CalledProcessError as e:
                    wr.writerow([f"{w2:.5f}", sd, "", "", f"{a.target:.4f}", f"subprocess_error:{e.returncode}"]); f.flush()
                    print(f"[scan] w2={w2:.3f} seed={sd} -> subprocess_error:{e.returncode}")
                except Exception as e:
                    wr.writerow([f"{w2:.5f}", sd, "", "", f"{a.target:.4f}", f"error:{type(e).__name__}"]); f.flush()
                    print(f"[scan] w2={w2:.3f} seed={sd} -> error:{type(e).__name__}")
            w2 = round(w2 + a.w2_step, 5)

if __name__ == "__main__":
    main()
