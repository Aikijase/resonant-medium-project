#!/usr/bin/env python3
import re, sys, json, pathlib

def grab_last_json(text):
    # take the last {...} block that parses as JSON
    last = None
    for m in re.finditer(r'\{[\s\S]*?\}', text):
        blob = m.group(0)
        try:
            J = json.loads(blob)
            last = J
        except Exception:
            pass
    return last

def main():
    if len(sys.argv) < 3:
        print("usage: grab_last_json.py <stdin|path> <out.json>", file=sys.stderr)
        sys.exit(2)
    src, out = sys.argv[1], sys.argv[2]
    if src == "-":
        text = sys.stdin.read()
    else:
        text = pathlib.Path(src).read_text()
    J = grab_last_json(text)
    if not J or not isinstance(J, dict) or "chi2" not in J or "best_params" not in J:
        print("ERROR: no valid results JSON (needs keys: chi2, best_params)", file=sys.stderr)
        sys.exit(1)
    pathlib.Path(out).write_text(json.dumps(J, indent=2, sort_keys=True))
    print(f"Wrote {out}")
if __name__ == "__main__":
    main()
