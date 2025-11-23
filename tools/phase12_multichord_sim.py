#!/usr/bin/env python3
import json, math, numpy as np

def load_fit(p="outputs/phase10/tongue_master_wide.fit.json"):
    with open(p) as f: j=json.load(f)
    return dict(a=j["a"], b=j["b"], c=j["c"], x0=j.get("x0",2.8))

def Kmin(omega, fit):
    x = omega - fit["x0"]
    k = fit["a"] + fit["b"]*x + fit["c"]*x*x
    return float(max(0.0, k))

def make_module_freqs(n, center=2.8, span=0.24, pattern="block"):
    if pattern=="block":
        dw = np.concatenate([np.full(n//2, -span/2), np.full(n-n//2, span/2)])
    elif pattern=="alt":
        dw = np.array([(-1)**i for i in range(n)])*(span/2)
    elif pattern=="grad":
        dw = np.linspace(-span/2, span/2, n)
    else:
        raise ValueError(pattern)
    return center + dw

def build_modules(M=3, n=6, intra="ring"):
    N = M*n
    A = np.zeros((N,N), float)
    for m in range(M):
        idx = np.arange(m*n, (m+1)*n)
        if intra=="ring":
            for i in range(n):
                u, v = idx[i], idx[(i+1)%n]
                A[u,v]=A[v,u]=1.0
        elif intra=="chain":
            for i in range(n-1):
                u, v = idx[i], idx[i+1]
                A[u,v]=A[v,u]=1.0
        elif intra=="clique":
            for i in range(n):
                for j in range(i+1,n):
                    u,v = idx[i], idx[j]
                    A[u,v]=A[v,u]=1.0
        else:
            raise ValueError(intra)
    return A

def add_bridges(A, M, n, mode="random_k", k=1, seed=0):
    rng=np.random.default_rng(seed)
    if mode=="none" or k<=0: return A
    modules=[np.arange(m*n,(m+1)*n) for m in range(A.shape[0]//n)]
    if mode=="star":
        hub=0
        for m in range(1,len(modules)):
            for _ in range(k):
                u = rng.choice(modules[hub]); v = rng.choice(modules[m])
                A[u,v]=A[v,u]=1.0
    elif mode=="ring":
        for m in range(len(modules)):
            m2=(m+1)%len(modules)
            for _ in range(k):
                u = rng.choice(modules[m]); v = rng.choice(modules[m2])
                A[u,v]=A[v,u]=1.0
    elif mode=="random_k":
        for m in range(len(modules)):
            for m2 in range(m+1,len(modules)):
                for _ in range(k):
                    u = rng.choice(modules[m]); v = rng.choice(modules[m2])
                    A[u,v]=A[v,u]=1.0
    else:
        raise ValueError(mode)
    return A

def order_param(theta):
    return abs(np.exp(1j*theta).mean())

def run_sim(M=3, n=6,
            centers=(2.7,2.8,2.9),
            span=0.24,
            patterns=("block","grad","alt"),
            intra="ring",
            bridge_mode="random_k",
            bridges=1,
            alpha=1.0,
            steps=20000, dt=0.01, burn_in=2000, sample_every=10, tail=500,
            noise=0.0, seed=0,
            fit_path="outputs/phase10/tongue_master_wide.fit.json"):
    rng=np.random.default_rng(seed)
    N=M*n
    if len(centers)<M: centers = tuple(list(centers)+[centers[-1]]*(M-len(centers)))
    if len(patterns)<M: patterns = tuple(list(patterns)+[patterns[-1]]*(M-len(patterns)))
    w=np.zeros(N)
    for m in range(M):
        w[m*n:(m+1)*n] = make_module_freqs(n, centers[m], span, patterns[m])

    A=build_modules(M,n,intra)
    A=add_bridges(A,M,n,mode=bridge_mode,k=bridges,seed=seed)

    fit=load_fit(fit_path)
    Kij=np.zeros_like(A)
    edges=np.argwhere(A>0.0)
    for i,j in edges:
        kref=min(Kmin(w[i],fit),Kmin(w[j],fit))
        Kij[i,j]=alpha*kref

    # --- Always define lambda2 here (before simulation) ---
    if N>1:
        D = np.diag(Kij.sum(axis=1))
        L = D - Kij
        evals = np.sort(np.linalg.eigvalsh(L))
        lambda2 = float(evals[1]) if len(evals)>1 else 0.0
    else:
        lambda2 = 0.0
    # ------------------------------------------------------

    theta=rng.uniform(0,2*np.pi,N)
    Rs=[]; pair_vals=[]; tail_thetas=[]; kept=0
    sigma=math.sqrt(2*noise*dt) if noise>0 else 0.0
    for t in range(steps):
        dtheta=w.copy()
        for i in range(N):
            js=np.nonzero(A[i])[0]
            if js.size:
                dtheta[i]+=np.sum(Kij[i,js]*np.sin(theta[js]-theta[i]))
        if noise>0: dtheta+=rng.normal(0.0,1.0,N)*sigma
        theta=(theta+dt*dtheta)%(2*np.pi)

        if t>=burn_in and (t-burn_in)%sample_every==0:
            Rs.append(order_param(theta))
            if edges.size:
                csum=np.sum([math.cos(theta[j]-theta[i]) for i,j in edges])
                pair_vals.append(csum/len(edges))
            tail_thetas.append(theta.copy())
            kept+=1
            if kept>tail: tail_thetas.pop(0); kept-=1

    R_mean=float(np.mean(Rs)) if Rs else 0.0
    pair_coh=float(np.mean(pair_vals)) if pair_vals else 0.0
    K_edge=float(Kij[A>0.0].mean()) if (A>0.0).any() else 0.0

    if tail_thetas:
        TH=np.stack(tail_thetas,axis=0)
        THu=np.unwrap(TH,axis=0)
        last=THu[-1]
        phase_span=float(last.max()-last.min())
        z=np.exp(1j*TH)
        circ_var=float(1-abs(z.mean(axis=0)).mean())
        R_modules=[]
        for m in range(M):
            idx=np.arange(m*n,(m+1)*n)
            R_modules.append(float(abs(np.exp(1j*last[idx]).mean())))
        mu=[]
        for m in range(M):
            idx=np.arange(m*n,(m+1)*n)
            mu.append(np.angle(np.exp(1j*last[idx]).mean()))
        mu=np.unwrap(np.array(mu))
        inter_lag_var=float(np.var(mu))
    else:
        phase_span=circ_var=inter_lag_var=0.0
        R_modules=[0.0]*M

    return dict(
        R_mean=R_mean, pair_coh=pair_coh, K_edge=K_edge,
        phase_span=phase_span, circ_var=circ_var,
        R_modules=R_modules, inter_lag_var=inter_lag_var,
        lambda2=lambda2
    )

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=3)
    ap.add_argument("--n",type=int,default=6)
    ap.add_argument("--centers",nargs="+",type=float,default=[2.7,2.8,2.9])
    ap.add_argument("--span",type=float,default=0.24)
    ap.add_argument("--patterns",nargs="+",default=["block","grad","alt"])
    ap.add_argument("--intra",choices=["ring","chain","clique"],default="ring")
    ap.add_argument("--bridge_mode",choices=["none","ring","star","random_k"],default="random_k")
    ap.add_argument("--bridges",type=int,default=1)
    ap.add_argument("--alpha",type=float,default=1.0)
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--dt",type=float,default=0.01)
    ap.add_argument("--burn_in",type=int,default=2000)
    ap.add_argument("--sample_every",type=int,default=10)
    ap.add_argument("--tail",type=int,default=500)
    ap.add_argument("--noise",type=float,default=0.0)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--fit",default="outputs/phase10/tongue_master_wide.fit.json")
    args=ap.parse_args()
    res=run_sim(M=args.M,n=args.n,centers=tuple(args.centers),span=args.span,patterns=tuple(args.patterns),
                intra=args.intra,bridge_mode=args.bridge_mode,bridges=args.bridges,
                alpha=args.alpha,steps=args.steps,dt=args.dt,burn_in=args.burn_in,
                sample_every=args.sample_every,tail=args.tail,noise=args.noise,seed=args.seed,
                fit_path=args.fit)
    print(json.dumps(res))
