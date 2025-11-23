#!/usr/bin/env bash
set -euo pipefail
slug="$1"
title="${2:-$slug}"
root="papers"
paper_dir="${root}/${slug}"

mkdir -p "${paper_dir}"/{manuscript,figures,data,scripts,refs,outputs,notes}
for d in manuscript figures data scripts refs outputs notes; do : > "${paper_dir}/${d}/.gitkeep"; done

cat > "${paper_dir}/README.md" <<EOF
# ${title}
Paper workspace created $(date +%F)
EOF

cat > "${paper_dir}/refs/references.bib" <<EOF
@article{example,
  author={Doe, J.},
  title={Example Reference},
  journal={Resonant Studies},
  year={2025}
}
EOF

echo "✅ Created paper folder: ${paper_dir}"
