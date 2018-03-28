all:
	(cat preamble.tex && \
		pandoc -t latex \
		--variable=documentclass=report \
		--bibliography dissertation.bib \
		--filter pandoc-fignos \
		--filter pandoc-citeproc \
		dissertation.md && \
		echo "\\end{document}") > dissertation.tex
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
