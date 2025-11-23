\
    #!/usr/bin/env python3
    """
    AIC/BIC calculator for the POC sprint.

    Usage:
      python3 calc_aic_bic.py --json-glob "outputs/poc_*.json" \
          --log poc_master_log.csv \
          --N_SN 1701 --N_BAO 26 --N_prior 1 \
          --baseline "LCDM:curv=0:rd=fixed" "LCDM:curv=1:rd=fixed" "LCDM:curv=0:rd=prior" "LCDM:curv=1:rd=prior"

    Notes:
      - We assume each JSON has fields:
          {"model":"LCDM"|"RESN",
           "curvature_free":0|1,
           "rd_mode":"fixed"|"prior",
           "npar": int,
           "chi2":{"total": float, "sn": float, "bao": float, "rd_prior": float},
           "cmd": "the exact CLI you ran",
           "notes": "...",
           "outfile": "outputs/poc_lcdm_flat_fixed.json"
          }
      - This script computes AIC/BIC and delta vs the designated baseline matching (model family is ignored
        for baseline matching; we compare against the *LCDM* case with same curvature+rd_mode).
      - You must pass the correct N = N_SN + N_BAO + N_prior (integers). 
    """
    import argparse, glob, json, math, os, time
    import csv

    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--json-glob", required=True, help="Glob of JSON result files, e.g., 'outputs/poc_*.json'")
        p.add_argument("--log", default="poc_master_log.csv")
        p.add_argument("--N_SN", type=int, required=True)
        p.add_argument("--N_BAO", type=int, required=True)
        p.add_argument("--N_prior", type=int, default=0)
        # Baseline keys: we match deltas vs LCDM cases with same curvature & rd_mode
        # e.g., "LCDM:curv=0:rd=fixed", "LCDM:curv=1:rd=fixed", etc.
        p.add_argument("--baseline", nargs="+", required=True,
                       help="List of baseline spec strings (LCDM only). Format 'LCDM:curv=<0|1>:rd=<fixed|prior>'")
        return p.parse_args()

    def key_from(meta):
        return f"{meta['model']}:{'curv='+str(meta['curvature_free'])}:rd={meta['rd_mode']}"

    def main():
        args = parse_args()
        N = args.N_SN + args.N_BAO + args.N_prior
        lnN = math.log(N)

        files = sorted(glob.glob(args.json_glob))
        if not files:
            raise SystemExit(f"No files match {args.json_glob}")

        # Load all JSONs
        rows = []
        by_key = {}
        for fp in files:
            with open(fp, "r") as jf:
                meta = json.load(jf)
            # required fields sanity
            for fld in ["model", "curvature_free", "rd_mode", "npar", "chi2"]:
                if fld not in meta:
                    raise SystemExit(f"{fp} missing required field '{fld}'")
            chi2_total = float(meta["chi2"]["total"])
            k = int(meta["npar"])
            AIC = chi2_total + 2.0 * k
            BIC = chi2_total + k * lnN
            meta["AIC"] = AIC
            meta["BIC"] = BIC
            meta["N"] = N
            meta["outfile"] = meta.get("outfile", fp)
            rows.append((fp, meta))
            by_key[key_from(meta)] = meta

        # Build the delta comparisons vs LCDM baselines that match curv & rd_mode
        # Example: for RESN:curv=1:rd=prior -> compare to LCDM:curv=1:rd=prior
        baseline_specs = {}
        for spec in args.baseline:
            parts = spec.split(":")
            if len(parts) != 3 or parts[0] != "LCDM":
                raise SystemExit(f"Bad baseline spec '{spec}'. Use 'LCDM:curv=<0|1>:rd=<fixed|prior>'")
            baseline_specs[":".join(parts[1:])] = spec

        # Map from curv/rd to the LCDM baseline row
        lcdm_by_cond = {}
        for fp, meta in rows:
            if meta["model"] == "LCDM":
                cond = f"curv={meta['curvature_free']}:rd={meta['rd_mode']}"
                lcdm_by_cond[cond] = meta

        # Prepare to append/update the log csv
        log_exists = os.path.exists(args.log)
        log_fields = ["timestamp","model","curvature_free","rd_mode","npar","N","chi2_total","chi2_sn","chi2_bao","chi2_rd",
                      "AIC","BIC","deltaAIC_vs_baseline","deltaBIC_vs_baseline","cmd","notes","outfile"]

        # Build new log entries
        new_entries = []
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for fp, meta in rows:
            cond = f"curv={meta['curvature_free']}:rd={meta['rd_mode']}"
            baseline = lcdm_by_cond.get(cond)
            dAIC = None
            dBIC = None
            if baseline is not None:
                dAIC = meta["AIC"] - baseline["AIC"]
                dBIC = meta["BIC"] - baseline["BIC"]

            entry = {
                "timestamp": now,
                "model": meta["model"],
                "curvature_free": meta["curvature_free"],
                "rd_mode": meta["rd_mode"],
                "npar": meta["npar"],
                "N": meta["N"],
                "chi2_total": float(meta["chi2"]["total"]),
                "chi2_sn": float(meta["chi2"].get("sn", float("nan"))),
                "chi2_bao": float(meta["chi2"].get("bao", float("nan"))),
                "chi2_rd": float(meta["chi2"].get("rd_prior", float("nan"))),
                "AIC": meta["AIC"],
                "BIC": meta["BIC"],
                "deltaAIC_vs_baseline": dAIC if dAIC is not None else "",
                "deltaBIC_vs_baseline": dBIC if dBIC is not None else "",
                "cmd": meta.get("cmd",""),
                "notes": meta.get("notes",""),
                "outfile": meta.get("outfile", fp),
            }
            new_entries.append(entry)

        # Append to CSV (create header if missing)
        write_header = not log_exists
        with open(args.log, "a", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=log_fields)
            if write_header:
                w.writeheader()
            for e in new_entries:
                w.writerow(e)

        # Print a compact summary to stdout
        def fmt(x):
            return "nan" if x=="" else f"{x:.3f}" if isinstance(x, float) else str(x)
        print("model  curv  rd     npar   chi2_tot    AIC       BIC     dAIC_vs_LCDM  dBIC_vs_LCDM  outfile")
        for e in new_entries:
            print(f"{e['model']:5}  {e['curvature_free']:>4}  {e['rd_mode']:<6} {e['npar']:>4}  "
                  f"{fmt(e['chi2_total']):>9}  {fmt(e['AIC']):>8}  {fmt(e['BIC']):>8}  "
                  f"{fmt(e['deltaAIC_vs_baseline']):>13}  {fmt(e['deltaBIC_vs_baseline']):>13}  {e['outfile']}")

    if __name__ == "__main__":
        main()
