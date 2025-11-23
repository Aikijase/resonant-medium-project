

.PHONY: dist

.PHONY: build clean score preflight

build: preflight
	@/usr/bin/pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -draftmode main.tex >/tmp/tex1.log 2>&1 || true
	@bibtex main >/tmp/bib.log 2>&1 || true
	@/usr/bin/pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex >/tmp/tex2.log 2>&1 || true
	@/usr/bin/pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex >/tmp/tex3.log 2>&1 || true
	@$(MAKE) score

preflight:
	@printf "== Preflight ==\n"
	@test -s JHEP.bst || (echo "Missing JHEP.bst"; exit 1)
	@test -s echo_eq_refs_v1.bib || (echo "Missing echo_eq_refs_v1.bib"; exit 1)
	@grep -q '\\documentclass' main.tex || (echo "No \\documentclass in main.tex"; exit 1)
	@grep -q '\\begin{document}' main.tex || (echo "No \\begin{document} in main.tex"; exit 1)
	@echo "OK"

score:
	@echo "=== SCOREBOARD ==="
	@echo "-- undefined citations --"
	@grep -nE 'Citation .* undefined|There were undefined references' /tmp/tex3.log || echo "OK"
	@echo "-- bibitems --"
	@awk '/\\bibitem/{print NR": "$$0}' main.bbl | sed -n '1,200p' || echo "no bbl?"
	@echo "-- first errors (if any) --"
	@grep -nE '^!|LaTeX Error:|Undefined control sequence|Emergency stop|Fatal error' /tmp/tex1.log /tmp/tex2.log /tmp/tex3.log | sed -n '1,40p' || echo "no hard errors"
	@echo "-- pdf --"
	@ls -lh main.pdf 2>/dev/null || echo "no pdf"

clean:
	rm -f main.{aux,bbl,blg,log,out,toc,lof,lot,fls,fdb_latexmk}

dist:
	git archive --format=zip --output=paper-jcap-src.zip HEAD
	@echo Wrote paper-jcap-src.zip

.PHONY: dist-arxiv
dist-arxiv: build
	@test -s main.bbl || (echo "No main.bbl; run make build first" && exit 1)
	zip -9r paper-jcap-arxiv-src.zip \
	  main.tex sections JHEP.bst echo_eq_refs_v1.bib main.bbl Makefile latexmkrc .gitignore \
	  figures outputs/phase20 outputs/phase8 outputs/predictions -x "*.aux" "*.log" "*.out" "*.fls" "*.fdb_latexmk" "*.synctex.gz" "*.toc" "*.lof" "*.lot"
	@echo Wrote paper-jcap-arxiv-src.zip

.PHONY: camera
camera: build
	cp -f main.pdf paper-jcap_camera_ready.pdf
	@echo Wrote paper-jcap_camera_ready.pdf

.PHONY: verify
verify: clean build
	@echo "=== VERIFY ==="
	@echo "-- bibitems --"
	@awk '/\\bibitem/{print NR": "$0}' main.bbl | sed -n '1,200p' || echo "no bbl?"
	@echo "-- warnings --"
	@(grep -nE "Overfull|Underfull|Reference .* undefined|Label .* multiply" /tmp/tex3.log || echo "none")
	@echo "-- pdf --"
	@ls -lh main.pdf

.PHONY: verify-plus
verify-plus: clean build
	@echo "=== VERIFY+ ==="
	@echo "-- bibitems (keys) --"
	@sed -n 's/\\bibitem{\([^}]*\).*/\1/p' main.bbl | nl -ba | sed -n '1,200p' || echo "no bbl?"
	@echo "-- warnings --"
	@(grep -nE "Overfull|Underfull|Reference .* undefined|Label .* multiply" /tmp/tex3.log || echo "none")
	@echo "-- pdf --"
	@ls -lh main.pdf
