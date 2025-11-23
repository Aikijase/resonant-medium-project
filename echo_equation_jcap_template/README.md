
# Echo Equation — JCAP LaTeX Template (Lock → Verify → Style)

This bundle contains a plain, functional LaTeX template aligned with common JCAP submissions using `jcappub`.

## Files
- `main.tex` — paper scaffold with all sections, appendices, and metadata.
- `refs.bib` — starter BibTeX file (replace with your final references).
- `Makefile` — convenience targets for local compile (optional).
- `LICENSE` — CC0/public-domain dedication for this template only (your paper is yours).

## Compile Locally
1. Ensure TeX packages installed (including `jcappub`). On Ubuntu:
   ```bash
   sudo apt-get update
   sudo apt-get install -y texlive-full latexmk
   ```
2. Build PDF:
   ```bash
   latexmk -pdf main.tex
   ```

## Overleaf
- Create a new Overleaf project and upload `main.tex` and `refs.bib`. Overleaf has `jcappub` pre-installed.

## Notes
- Email is locked to: theexperimentalistlab@outlook.com
- Funding: None. Conflicts: None.
- Acknowledgement mentions ChatGPT-5 for drafting/organization only (adjust if preferred).
- Follow the “Lock → Verify → Style” protocol: get a compiling draft first, then we can do a styling pass (figures, tables, polish).



---

## Troubleshooting (jcappub.sty missing)

If `jcappub.sty` is missing on your machine, you now have **three** ways to compile:

1. **JCAP (preferred):** install the package and run
   ```bash
   latexmk -pdf main.tex
   ```

2. **Fallback (very similar look):**
   ```bash
   latexmk -pdf main_jheppub.tex
   ```

3. **Plain article (works anywhere):**
   ```bash
   latexmk -pdf main_article.tex
   ```

A small `jcappub.sty` shim is included that attempts to load the real JCAP style and otherwise falls back to `jheppub` so `main.tex` should compile even without system-wide `jcappub`.
