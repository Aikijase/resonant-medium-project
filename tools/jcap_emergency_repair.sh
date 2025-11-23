#!/usr/bin/env bash
set -euo pipefail
MAIN="paper-jcap/main.tex"
cp -f "$MAIN" "${MAIN}.bak"

# Restore common \new* macros that may have lost the backslash
# Start-of-line (optionally indented) cases:
sed -i -E 's/^(\s*)ewcommand/\1\\newcommand/'          "$MAIN"
sed -i -E 's/^(\s*)ewenvironment/\1\\newenvironment/'  "$MAIN"
sed -i -E 's/^(\s*)ewtheorem/\1\\newtheorem/'          "$MAIN"
sed -i -E 's/^(\s*)ewlength/\1\\newlength/'            "$MAIN"

# Very safe mid-line restoration if preceded by non-backslash and non-letter
# (prevents double-escaping already-correct \newcommand)
sed -i -E 's/([^\\A-Za-z])ewcommand/\1\\newcommand/g'         "$MAIN"
sed -i -E 's/([^\\A-Za-z])ewenvironment/\1\\newenvironment/g' "$MAIN"
sed -i -E 's/([^\\A-Za-z])ewtheorem/\1\\newtheorem/g'         "$MAIN"
sed -i -E 's/([^\\A-Za-z])ewlength/\1\\newlength/g'           "$MAIN"

# Clean any stray regex artifacts from earlier runs
sed -i 's/\\1[[:space:]]*//g' "$MAIN"

# Normalize the table header middle cell to a single math block
sed -i -E \
  -e 's/(&[[:space:]]*)Mean[[:space:]]*±[[:space:]]*SD([[:space:]]*&)/\1$Mean \\pm SD$\2/' \
  -e 's/(&[[:space:]]*)\$Mean[[:space:]]*\$\\pm[[:space:]]*\$[[:space:]]*SD\$(\s*&)/\1$Mean \\pm SD$\2/' \
  -e 's/(&[[:space:]]*)\$Mean[[:space:]]*\\pm[[:space:]]*SD\$(\s*&)/\1$Mean \\pm SD$\2/' \
  "$MAIN"

# Force-wrap any table middle cell containing \pm (numeric value) in math mode
awk '
function wrap_math(s){gsub(/^[ \t]+|[ \t]+$/, "", s); if(s ~ /^\$.*\$$/) return s; return "$" s "$"}
{
  line = $0
  if (line ~ /\\begin{tabular}/) inTab=1
  if (inTab==1 && line ~ /\\pm/) {
    n = split(line, parts, /&/)
    if (n >= 3) {
      mid = parts[2]
      gsub(/\r$/,"", mid)
      if (mid !~ /\$.*\$/) parts[2] = " " wrap_math(mid) " "
      line = parts[1]
      for(i=2;i<=n;i++) line = line "&" parts[i]
    }
  }
  print line
  if (line ~ /\\end{tabular}/) inTab=0
}' "$MAIN" > "${MAIN}.tmp" && mv "${MAIN}.tmp" "$MAIN"

echo "Emergency repair complete. Backup at ${MAIN}.bak"
