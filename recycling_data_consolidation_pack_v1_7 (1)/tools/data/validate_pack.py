#!/usr/bin/env python3
import sys, csv, os, hashlib, datetime

def sha256sum(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def basic_validate(csv_path, expect_header_len):
    with open(csv_path, newline='') as f:
        rdr = csv.reader(f)
        rows = [r for r in rdr if r and not r[0].startswith("#")]
    if not rows:
        return (0, False, "empty-or-only-comments")
    hdr, *data = rows
    if len(hdr) != expect_header_len:
        return (len(data), False, f"header_len_mismatch got={len(hdr)} expect={expect_header_len}")
    # z monotonic (if z in first col)
    try:
        zs = [float(r[0]) for r in data]
        mono = all(zs[i] <= zs[i+1] for i in range(len(zs)-1))
        if not mono:
            return (len(data), False, "z_not_monotonic")
    except Exception:
        pass
    return (len(data), True, "ok")

def main():
    if len(sys.argv) < 4:
        print("usage: validate_pack.py <templates_dir> <schema_dir> <master_log_csv>")
        sys.exit(2)
    tdir, sdir, mpath = sys.argv[1:4]
    templates = {
        "rho_bh_z.csv": 6,
        "smbh_population_history.csv": 8,
        "sfr_imf_bh_eff.csv": 7,
        "expansion_Hz.csv": 6,
        "omega_m_z.csv": 5,
        "eos_w_z.csv": 5,
    }
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    logf = open(mpath, "a", newline="")
    logw = csv.writer(logf)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    for name, hlen in templates.items():
        path = os.path.join(tdir, name)
        if not os.path.exists(path):
            logw.writerow([ts, "validate", "missing", name, ""])
            continue
        n, ok, msg = basic_validate(path, hlen)
        ch = sha256sum(path)
        status = "ok" if ok else f"fail:{msg}"
        logw.writerow([ts, f"validate:{name}", status, f"rows={n} sha256={ch[:12]}", "fix→re-run" if not ok else ""])
    logf.close()
    print("Validation complete. Logged to", mpath)

if __name__ == "__main__":
    main()
