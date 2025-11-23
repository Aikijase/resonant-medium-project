
#!/usr/bin/env bash
set -euo pipefail
# Set identity (edit these to your details)
git config --global user.name "Jason Watts"
git config --global user.email "jason@example.com"

# Initialize if needed
if [ ! -d .git ]; then
  git init
fi

git add -A
git commit -m "Add Phase 2 bundle: digitizer, collapse overlay, cosmic anchor scaffold"
