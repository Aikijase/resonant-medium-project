#!/usr/bin/env bash
set -e
printf "== Preflight ==\n"
grep -q '\\documentclass' main.tex
grep -q '\\begin{document}' main.tex
printf "OK\n"
