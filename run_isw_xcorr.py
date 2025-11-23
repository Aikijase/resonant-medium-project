import argparse, json, yaml, math, tomllib
from isw import cl_tg_proxy

def aic_bic(chi2,k,n): 
    AIC=chi2+2*k if chi2 is not None else None
    BIC=chi2+k*math.log(max(n,1)) if chi2 is not None else None
    return AIC,BIC

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="cfg",required=True)
    ap.add_argument("--bins",default="data/isw/bins.yaml")
    ap.add_argument("--out",dest="out",required=True)
    args=ap.parse_args()

    with open(args.cfg,"rb") as fh: P=tomllib.load(fh)["params"]
    bins=yaml.safe_load(open(args.bins))
    band=cl_tg_proxy(P,bins)
    n=sum(len(b["ell"]) for b in band)
    AIC,BIC=aic_bic(None,0,n)
    json.dump({"probe":"isw_xcorr","n":n,"k":0,"chi2":None,
               "AIC":{"value":AIC,"delta":None},
               "BIC":{"value":BIC,"delta":None},
               "bandpowers":band}, open(args.out,"w"), indent=2)
    print(f"Wrote {args.out}")
