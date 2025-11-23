#!/usr/bin/env bash
set -euo pipefail
f="tools/run_phase2.sh"
touch "$f"

# Add κκ if missing
grep -q "tools/run_phase4.sh" "$f" || echo 'python3 tools/run_phase4.sh' >> "$f"

# Add g×κ if missing
grep -q "tools/run_phase4_gk.sh" "$f" || echo 'python3 tools/run_phase4_gk.sh' >> "$f"

# Add Phase-4 orchestrator (summary + report patch) near the end
grep -q "tools/run_phase4_all.sh" "$f" || echo 'bash tools/run_phase4_all.sh' >> "$f"

echo "Wired Phase-4 into $f"
