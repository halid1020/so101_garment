# Build the teleoperation paper. Requires latexmk + a TeX distribution.
all:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f main.bbl

.PHONY: all clean
