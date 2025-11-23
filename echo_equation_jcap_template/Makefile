
PDF=main.pdf

all: pdf

pdf:
	latexmk -pdf main.tex

jhep:
	latexmk -pdf main_jheppub.tex

article:
	latexmk -pdf main_article.tex

clean:
	latexmk -C
	rm -f main.bbl main.bcf main.run.xml

.PHONY: all pdf clean
