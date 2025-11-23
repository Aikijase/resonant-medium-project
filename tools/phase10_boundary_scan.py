#!/usr/bin/env python3
import sys, subprocess, json, math, csv, pathlib

PYTHON = sys.executable  # use current interpreter

def si(w2, K):
    p = subprocess.run(
        [PYTHON, "tools/phase10_phasecouple_demo.py",
         "--preset","neuron","--omega2",str(w2),
         "--kv","0.10","--kx","0.20","--Kphi",str(K),
         "--eps","0.06","--adapt_every","15","--noise","0.01",
         "--steps","20000","--burn_in","300","--prefix",f"bdry_w{w2}_K{K}"],
        capture_output=True, text=True, check=True
    )
    data = json.loads(p.stdout)
    return float(data["metrics"]["sync_index"])

def main():
    w1  = 2.8
    dws = [round(w1+d,2) for d in [-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,-0.02,0.0,0.02,0.04,0.06,0.1,0.2,0.3,0.4,0.5,0.6]]
    Ks  = [0.1,0.2,0.3,0.5,0.7,0.9,1.1,1.3,1.6]

    out_rows = []
    print("delta_omega,omega2,Kphi_min,sync_index")
    for w2 in dws:
        best = None
        for K in Ks:
            try:
                s = si(w2, K)
            except subprocess.CalledProcessError as e:
                # child script crashed; skip this K (uncomment next line to debug)
                # print(f"# phase10 child failed for w2={w2}, K={K}:\n{e.stderr}", file=sys.stderr)
                continue
            except Exception:
                # malformed JSON or missing key; skip
                continue
            if not math.isnan(s) and s >= 0.95:
                best = (K, s)
                break
        if best:
            print(f"{w2 - w1:.3f},{w2},{best[0]},{best[1]:.3f}")
            out_rows.append([f"{w2 - w1:.3f}", w2, best[0], f"{best[1]:.3f}"])
        else:
            print(f"{w2 - w1:.3f},{w2},NA,NA")
            out_rows.append([f"{w2 - w1:.3f}", w2, "", ""])

    # Save a CSV too
    out_path = pathlib.Path("outputs/phase10/boundary_scan.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_omega","omega2","Kphi_min","sync_index"])
        w.writerows(out_rows)
    print(f"# CSV saved: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
