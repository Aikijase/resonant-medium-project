# tools/run_with_cfg.py
import json, subprocess as sp, sys, tempfile, pathlib, os

def run(cfg: dict, out_path: str):
    out_path = str(out_path)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cfg = dict(cfg)
    stem = os.path.splitext(out_path)[0]
    cfg["out"] = out_path
    cfg.setdefault("out_prefix", stem)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf); tf.flush()
        print(f"[run_with_cfg] Using config: {tf.name}")
        try:
            sp.run(["python3", "run_joint_bao_sn.py", tf.name], check=True)
        except sp.CalledProcessError as e:
            print("[run_with_cfg] Runner error:", e)
            print("[run_with_cfg] Config was:\n" + open(tf.name).read())
            raise

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]); out = sys.argv[2]
    run(cfg, out)
