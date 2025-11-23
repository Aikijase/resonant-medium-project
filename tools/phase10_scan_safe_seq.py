#!/usr/bin/env python3
import csv, json, math, sys, subprocess, re
from pathlib import Path

PYX = sys.executable

def parse_sync(stdout: str) -> float:
    stdout = stdout.strip()
    # Try clean JSON
    try:
        obj = json.loads(stdout)
        return float(obj["metrics"]["sync_index"])
    except Exception:
        pass
    # Fallback: last JSON object in mixed logs
    candidates = list(re.finditer(r"\{.*?\}", stdout, flags=re.DOTALL))
    for m in reversed(candidates):
        try:
            obj = json.loads(m.group(0))
            if "metrics" in obj and "sync_index" in obj["metrics"]:
                return float(obj["metrics"]["sync_index"])
        except Exception:
            continue
    raise ValueError("Could not parse JSON metrics from subprocess stdout.")

def sync_index_at(omega2, Kphi, preset="neuron", kv="0.10", kx="0.20", eps="0.06",
                  adapt_every="15", noise="0.01", steps=15000, burn_in=300,
                  seed=None, prefix=None) -> float:
    cmd = [PYX, "tools/phase10_phasecouple_demo.py",
           "--preset", preset, "--omega2", str(omega2),
           "--kv", kv, "--kx", kx, "--Kphi", str(Kphi),
           "--eps", eps, "--adapt_every", adapt_every,
           "--noise", noise, "--steps", str(steps), "--burn_in", str(burn_in)]
    if seed is not None:
        cmd += ["--seed1", str(seed), "--seed2", str(seed+101)]
    if prefix:
        cmd += ["--prefix", prefix]
    p = subprocess.run(cmd, capture_output=True, text=True)
    p.check_returncode()
    return parse_sync(p.stdout)

def ensure_bracket(omega2, target, k_lo, k_hi, max_k, grow, max_tries, **kw):
    s_lo = sync_index_at(omega2, k_lo, **kw)
    if s_lo >= target:
        return ("at_lower", (k_lo, s_lo))
    s_hi = sync_index_at(omega2, k_hi, **kw)
    if s_hi >= target:
        return ("bracket", (k_lo, s_lo), (k_hi, s_hi))
    k, s, tries = k_hi, s_hi, 0
    while s < target and k < max_k and tries < max_tries:
        k = min(max_k, k * grow)
        s = sync_index_at(omega2, k, **kw)
        tries += 1
    if s >= target:
        return ("bracket", (k_lo, s_lo), (k, s))
    return ("fail", s)

def bisect_boundary(omega2, target, bracket, tol, max_iter, **kw):
    (kL, sL), (kH, sH) = bracket
    km = None; sm = None
    for _ in range(max_iter):
        km = 0.5*(kL+kH)
        sm = sync_index_at(omega2, km, **kw)
        if abs(sm - target) <= tol:
            return km, sm
        if sm >= target:
            kH, sH = km, sm
        else:
            kL, sL = km, sm
    return km, sm

def already_done(path):
    done = set()
    p = Path(path)
    if p.exists():
        with p.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    done.add( (round(float(row["omega2"]),5), int(row["seed"])) )
                except Exception:
                    pass
    return done

def write_header_if_new(path):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(["omega2","seed","Kphi_min_est","sync_at_est","target","status"])

def main():
    # Tip region first (fast). Widen later.
    w2_min, w2_max, w2_step = 2.74, 2.82, 0.01
    seeds = [0,1]
    target, tol = 0.95, 0.002
    # Ceiling high enough to actually hit 0.95 on left wing
    k_lo, k_hi, max_k, grow, max_grow_tries = 0.10, 0.90, 1.60, 1.6, 14
    steps, burn_in = 15000, 300

    out = "outputs/phase10/boundary_scan_wide.csv"
    write_header_if_new(out)
    done = already_done(out)

    with open(out, "a", newline="") as f:
        wr = csv.writer(f)
        w2 = w2_min
        while w2 <= w2_max + 1e-12:
            for seed in seeds:
                key = (round(w2,5), seed)
                if key in done:
                    continue
                cfg = dict(preset="neuron", kv="0.10", kx="0.20", eps="0.06",
                           adapt_every="15", noise="0.01", steps=steps, burn_in=burn_in,
                           seed=seed, prefix=f"bdry_w{w2:.3f}_seed{seed}")
                try:
                    kind, *rest = ensure_bracket(w2, target, k_lo, k_hi, max_k, grow, max_grow_tries, **cfg)
                    if kind == "at_lower":
                        k_est, s_est = rest[0]
                        wr.writerow([f"{w2:.5f}", seed, f"{k_est:.6f}", f"{s_est:.6f}", f"{target:.4f}", "at_lower"])
                        print(f"[safe] w2={w2:.3f} seed={seed} -> at_lower K≈{k_est:.6f} s≈{s_est:.6f}")
                    elif kind == "fail":
                        s_last = rest[0]
                        wr.writerow([f"{w2:.5f}", seed, "", f"{s_last:.6f}", f"{target:.4f}", "unreachable"])
                        print(f"[safe] w2={w2:.3f} seed={seed} -> unreachable s≈{s_last:.6f}")
                    else:
                        bracket = rest
                        k_est, s_est = bisect_boundary(w2, target, bracket, tol, 18, **cfg)
                        wr.writerow([f"{w2:.5f}", seed, f"{k_est:.6f}", f"{s_est:.6f}", f"{target:.4f}", "ok"])
                        print(f"[safe] w2={w2:.3f} seed={seed} -> ok K≈{k_est:.6f} s≈{s_est:.6f}")
                except subprocess.CalledProcessError as e:
                    wr.writerow([f"{w2:.5f}", seed, "", "", f"{target:.4f}", f"subprocess_error:{e.returncode}"])
                    print(f"[safe] w2={w2:.3f} seed={seed} -> subprocess_error:{e.returncode}")
                except Exception as e:
                    wr.writerow([f"{w2:.5f}", seed, "", "", f"{target:.4f}", f"error:{type(e).__name__}"])
                    print(f"[safe] w2={w2:.3f} seed={seed} -> error:{type(e).__name__}")
            w2 = round(w2 + w2_step, 5)

if __name__ == "__main__":
    main()
