all:
	(cat preamble.tex && \
		pandoc -t latex \
		--variable=documentclass=report \
		--bibliography dissertation.bib \
		--filter pandoc-fignos \
		--biblatex \
		dissertation.md && \
		cat postamble.tex) > dissertation.tex
	pdflatex dissertation
	biber dissertation
	pdflatex dissertation
	git add .
	git commit -m "updated as of $(shell date)"
	git push
	open dissertation.pdf

clean:
	rm dissertation.bbl
	rm dissertation.blg
	rm dissertation.log
	rm dissertation.aux
	rm dissertation.toc
	rm dissertation.run.xml
	rm dissertation.bcf
	rm dissertation.out
