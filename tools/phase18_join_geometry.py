#!/usr/bin/env python3
import argparse, csv, os, sys, math, json
from pathlib import Path

def read_shell(path, si_min=0.0, max_rows=None, stride=1):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for i,row in enumerate(r):
            if i%stride!=0: continue
            try:
                w2=float(row["omega2"]); K=float(row["Kphi"])
                si=float(row.get("sync_index", row.get("si", "0")))
            except Exception:
                continue
            if si>=si_min: rows.append((w2,K,si))
            if max_rows and len(rows)>=max_rows: break
    return rows

def pairwise_knn(points, k=6):
    # points: list of (w2, K, si)
    import numpy as np
    X=np.array([(p[0],p[1]) for p in points], dtype=float)
    n=X.shape[0]
    # compute distances efficiently in blocks to avoid huge memory for very large n
    # but n is usually manageable; fallback to full matrix for simplicity
    D = ((X[:,None,:]-X[None,:,:])**2).sum(axis=2)**0.5
    for i in range(n):
        D[i,i]=1e9
    # indices of k nearest neighbors for each node
    nbr_idx = np.argpartition(D, k, axis=1)[:, :k]
    nbr_dist = D[np.arange(n)[:,None], nbr_idx]
    return nbr_idx, nbr_dist

def build_edges(points, nbr_idx, nbr_dist, sigma_si=0.02, sigma_len=0.05):
    # weight favors small Δsi and short distance
    import numpy as np
    n=len(points)
    edges=[]
    for i in range(n):
        w2i,Ki,sii=points[i]
        for j_idx,dist in zip(nbr_idx[i], nbr_dist[i]):
            if j_idx<0 or j_idx==i: continue
            w2j,Kj,sij=points[j_idx]
            dsi=abs(sii-sij)
            # simple smooth weight; clamp to [0,1]
            w = math.exp(-dsi/max(sigma_si,1e-9)) * math.exp(-dist/max(sigma_len,1e-9))
            edges.append((i,j_idx,dist,dsi,w))
    return edges

def components(n, edges, w_min=0.2):
    # BFS over edges with weight >= w_min (undirected)
    adj=[[] for _ in range(n)]
    for i,j,dist,dsi,w in edges:
        if w>=w_min:
            adj[i].append(j); adj[j].append(i)
    comp=[-1]*n; cid=0
    from collections import deque
    for i in range(n):
        if comp[i]!=-1: continue
        q=deque([i]); comp[i]=cid
        while q:
            u=q.popleft()
            for v in adj[u]:
                if comp[v]==-1:
                    comp[v]=cid; q.append(v)
        cid+=1
    return comp, cid

def write_csvs(outdir, prefix, points, edges, comp):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # nodes
    with open(Path(outdir,f"{prefix}_nodes.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["id","omega2","Kphi","sync_index","component"])
        for i,(w2,K,si) in enumerate(points):
            w.writerow([i,f"{w2:.6f}",f"{K:.6f}",f"{si:.6f}",comp[i]])
    # edges
    with open(Path(outdir,f"{prefix}_edges.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["src","dst","dist","delta_si","weight"])
        for (i,j,dist,dsi,wt) in edges:
            w.writerow([i,j,f"{dist:.6f}",f"{dsi:.6f}",f"{wt:.6f}"])
    # components summary
    from collections import defaultdict
    S=defaultdict(list)
    for i,(w2,K,si) in enumerate(points): S[comp[i]].append((w2,K,si))
    with open(Path(outdir,f"{prefix}_components.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["component","size","omega2_mean","Kphi_mean","si_mean"])
        for c,v in sorted(S.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            n=len(v)
            w2m=sum(x[0] for x in v)/n; Km=sum(x[1] for x in v)/n; sim=sum(x[2] for x in v)/n
            w.writerow([c,n,f"{w2m:.6f}",f"{Km:.6f}",f"{sim:.6f}"])

def plot_join_map(outdir, prefix, points, comp, edges, max_edges=4000):
    try:
        import numpy as np, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X=np.array([(p[0],p[1],p[2],comp[i]) for i,p in enumerate(points)])
        fig=plt.figure(figsize=(7,6))
        # scatter colored by component (mod a few colors)
        cm = plt.cm.get_cmap('tab20')
        for c in sorted(set(comp)):
            m = (X[:,3]==c)
            plt.scatter(X[m,1], X[m,0], s=8, alpha=0.9, label=f"C{c}", color=cm(c%20))
        # light edges
        draw=edges if len(edges)<=max_edges else edges[:max_edges]
        for (i,j,dist,dsi,w) in draw:
            w2i,Ki,_=points[i]; w2j,Kj,_=points[j]
            plt.plot([Ki,Kj],[w2i,w2j], linewidth=0.4, alpha=0.15, color="k")
        plt.xlabel("Kphi"); plt.ylabel("omega2"); plt.title(f"{prefix} — join map")
        plt.tight_layout()
        fig.savefig(Path(outdir,f"{prefix}_join_map.png"), dpi=160)
        plt.close(fig)
    except Exception as e:
        sys.stderr.write(f"[warn] plot failed: {e}\n")

def main():
    ap=argparse.ArgumentParser(description="Phase-18 Shell-Join Geometry")
    ap.add_argument("--shell-csv", required=True, help="Phase-14 shell file (omega2,Kphi,sync_index)")
    ap.add_argument("--outdir", default="outputs/phase18")
    ap.add_argument("--prefix", default="p18")
    ap.add_argument("--si-min", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sigma-si", type=float, default=0.02)
    ap.add_argument("--sigma-len", type=float, default=0.05)
    ap.add_argument("--w-min", type=float, default=0.20)
    ap.add_argument("--max-nodes", type=int, default=4000, help="cap for pairwise calc")
    ap.add_argument("--stride", type=int, default=1, help="subsample every Nth row")
    args=ap.parse_args()

    pts = read_shell(args.shell_csv, si_min=args.si_min, max_rows=args.max_nodes, stride=args.stride)
    if not pts:
        print("no points after filtering; relax --si-min or stride", file=sys.stderr); return 2

    nbr_idx, nbr_dist = pairwise_knn(pts, k=args.k)
    edges = build_edges(pts, nbr_idx, nbr_dist, sigma_si=args.sigma_len if args.sigma_si<=0 else args.sigma_si,
                        sigma_len=args.sigma_len)
    comp, ncomp = components(len(pts), edges, w_min=args.w_min)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    write_csvs(args.outdir, args.prefix, pts, edges, comp)
    plot_join_map(args.outdir, args.prefix, pts, comp, edges)

    # tiny report
    with open(Path(args.outdir,f"{args.prefix}_report.md"),"w") as f:
        f.write("# Phase 18 — Shell-Join Geometry (Lock)\n")
        f.write(f"- shell_csv: {args.shell_csv}\n")
        f.write(f"- nodes: {len(pts)}  k: {args.k}  si_min: {args.si_min}\n")
        f.write(f"- sigmas: si={args.sigma_si} len={args.sigma_len}  w_min={args.w_min}\n")

if __name__=="__main__":
    raise SystemExit(main())
