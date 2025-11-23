def sync_index_at(omega2, Kphi, cfg):
    import re, json, subprocess, sys
    cmd = [sys.executable, "tools/phase10_phasecouple_demo.py",
           "--preset", cfg.preset, "--omega2", str(omega2),
           "--kv", cfg.kv, "--kx", cfg.kx, "--Kphi", str(Kphi),
           "--eps", cfg.eps, "--adapt_every", cfg.adapt_every,
           "--noise", cfg.noise, "--steps", str(cfg.steps), "--burn_in", str(cfg.burn_in)]
    if cfg.seed is not None:
        cmd += ["--seed1", str(cfg.seed), "--seed2", str(cfg.seed + 101)]
    if cfg.prefix:
        cmd += ["--prefix", cfg.prefix]
    p = subprocess.run(cmd, capture_output=True, text=True)
    p.check_returncode()
    out = p.stdout.strip()

    # Try direct JSON first
    try:
        obj = json.loads(out)
        return float(obj["metrics"]["sync_index"])
    except Exception:
        pass

    # Fallback: extract last JSON object from mixed stdout
    # This is tolerant of logs before/after JSON
    candidates = list(re.finditer(r"\{.*?\}", out, flags=re.DOTALL))
    for m in reversed(candidates):
        try:
            obj = json.loads(m.group(0))
            if "metrics" in obj and "sync_index" in obj["metrics"]:
                return float(obj["metrics"]["sync_index"])
        except Exception:
            continue

    # If we reach here, we couldn’t parse any JSON
    raise ValueError("Could not parse JSON metrics from subprocess stdout.")
