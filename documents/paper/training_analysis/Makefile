# Build the cropped-tactile training analysis report.
all:
	latexmk -pdf -interaction=nonstopmode training_analysis.tex

clean:
	latexmk -C
	rm -f training_analysis.bbl

.PHONY: all clean
