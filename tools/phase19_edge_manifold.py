#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

def read_nodes(nodes_csv, comp_csv):
    # pick largest component id
    comps=[]
    with open(comp_csv) as f:
        r=csv.DictReader(f)
        for row in r:
            comps.append((int(row["component"]), int(row["size"])))
    comps.sort(key=lambda x: -x[1])
    largest = comps[0][0]

    nodes=[]
    with open(nodes_csv) as f:
        r=csv.DictReader(f)
        for row in r:
            if int(row["component"])==largest:
                nodes.append((int(row["id"]), float(row["omega2"]), float(row["Kphi"])))
    nodes.sort(key=lambda x: x[0])
    return largest, nodes

def read_edges(edges_csv, allowed_ids):
    allow = set(i for (i,_,_) in allowed_ids)
    edges=[]
    with open(edges_csv) as f:
        r=csv.DictReader(f)
        for row in r:
            i=int(row["src"]); j=int(row["dst"])
            if i in allow and j in allow:
                edges.append((i,j,float(row["dist"])))
    return edges

def mst_kruskal(n, edges):
    # edges: (i,j,dist). return adjacency list of MST
    parent=list(range(n))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb: return False
        parent[rb]=ra; return True

    # remap node ids to 0..n-1
    idx_map={nid:k for k,(nid,_,_) in enumerate(sorted_nodes)}
    inv_id=[nid for nid,_,_ in sorted_nodes]

    es=[]
    for i,j,d in edges:
        if i in idx_map and j in idx_map:
            es.append((idx_map[i], idx_map[j], d))
    es.sort(key=lambda x: x[2])

    adj=[[] for _ in range(n)]
    taken=0
    for a,b,d in es:
        if union(a,b):
            adj[a].append((b,d))
            adj[b].append((a,d))
            taken+=1
            if taken==n-1: break
    return adj, inv_id

def bfs_longest(adj, start):
    from collections import deque
    N=len(adj)
    dist=[math.inf]*N; par=[-1]*N
    dist[start]=0.0
    q=deque([start])
    while q:
        u=q.popleft()
        for v,d in adj[u]:
            if dist[v]==math.inf:
                dist[v]=dist[u]+d; par[v]=u; q.append(v)
    far=max(range(N), key=lambda i: dist[i])
    return far, dist, par

def extract_path(adj):
    # 2-BFS diameter on tree
    a,_,_=bfs_longest(adj, 0)
    b,_,par=bfs_longest(adj, a)
    path=[]
    u=b
    while u!=-1:
        path.append(u); u=par[u]
    path=path[::-1]
    # arc-length s
    s=[0.0]
    for i in range(1,len(path)):
        u,v=path[i-1],path[i]
        # find edge length
        w=next(d for x,d in adj[u] if x==v)
        s.append(s[-1]+w)
    return path, s

def smooth_polyline(xs, ys, win=5):
    if win<3: return xs, ys
    half=win//2
    X=[]; Y=[]
    for i in range(len(xs)):
        lo=max(0,i-half); hi=min(len(xs),i+half+1)
        X.append(sum(xs[lo:hi])/(hi-lo))
        Y.append(sum(ys[lo:hi])/(hi-lo))
    return X, Y

def curvature(xs, ys):
    # discrete curvature using finite differences
    k=[0.0]*len(xs)
    for i in range(1,len(xs)-1):
        x1,y1=xs[i-1],ys[i-1]; x2,y2=xs[i],ys[i]; x3,y3=xs[i+1],ys[i+1]
        a=((x2-x1),(y2-y1)); b=((x3-x2),(y3-y2))
        cross=abs(a[0]*b[1]-a[1]*b[0])
        la=(a[0]**2+a[1]**2)**0.5; lb=(b[0]**2+b[1]**2)**0.5
        if la*lb>0:
            k[i]=cross/(la*lb*(la+lb)/2.0)
    k[0]=k[1]; k[-1]=k[-2]
    return k

def main():
    ap=argparse.ArgumentParser(description="Phase-19 Unified Edge Manifold")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--edges-csv", required=True)
    ap.add_argument("--components-csv", required=True)
    ap.add_argument("--outdir", default="outputs/phase19")
    ap.add_argument("--prefix", default="p19_edge")
    ap.add_argument("--smooth", type=int, default=7)
    args=ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    global sorted_nodes
    comp_id, sorted_nodes = read_nodes(args.nodes_csv, args.components_csv)
    n=len(sorted_nodes)

    edges=read_edges(args.edges_csv, sorted_nodes)
    adj, inv_id = mst_kruskal(n, edges)

    path, s = extract_path(adj)
    w2=[ sorted_nodes[i][1] for i in path ]
    K =[ sorted_nodes[i][2] for i in path ]

    Ws, Ks = smooth_polyline(w2, K, win=args.smooth)
    kappa = curvature(Ws, Ks)

    # write curve
    outcsv=Path(args.outdir, f"{args.prefix}_curve.csv")
    with open(outcsv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["s","omega2","Kphi","kappa"])
        for i in range(len(Ws)):
            w.writerow([f"{s[i]:.6f}", f"{Ws[i]:.6f}", f"{Ks[i]:.6f}", f"{kappa[i]:.6e}"])

    # plots
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(6,5))
        plt.plot(Ks, Ws, '-', linewidth=2)
        plt.xlabel("Kphi"); plt.ylabel("omega2"); plt.title("Unified Edge Manifold")
        fig.tight_layout(); fig.savefig(Path(args.outdir, f"{args.prefix}_curve.png"), dpi=160); plt.close(fig)

        fig=plt.figure(figsize=(6,3.6))
        plt.plot(s, kappa, '-'); plt.xlabel("arc length s"); plt.ylabel("curvature κ(s)")
        fig.tight_layout(); fig.savefig(Path(args.outdir, f"{args.prefix}_curvature.png"), dpi=160); plt.close(fig)
    except Exception as e:
        pass

    # tiny report
    with open(Path(args.outdir,f"{args.prefix}_report.md"),"w") as f:
        f.write("# Phase 19 — Unified Edge Manifold (Lock)\n")
        f.write(f"- largest_component_id: {comp_id}\n")
        f.write(f"- nodes_in_component: {n}\n")
        f.write(f"- smoothing_window: {args.smooth}\n")
        f.write(f"- outputs: {outcsv.name}, {args.prefix}_curve.png, {args.prefix}_curvature.png\n")

if __name__=="__main__":
    raise SystemExit(main())
