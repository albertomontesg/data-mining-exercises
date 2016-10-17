
REPORT_FILE=report_hw2.tex
BUILD_NAME=report_hw2
DEFAULT_READER=Preview.app

# The first rule in a Makefile is the one executed by default ("make"). It
# should always be the "all" rule, so that "make" and "make all" are identical.
all: report

# MAIN LATEXMK RULE

# -pdf tells latexmk to generate PDF directly (instead of DVI).
# -pdflatex="" tells latexmk to call a specific backend with specific options.
# -use-make tells latexmk to call make for generating missing files.

# -interaction=nonstopmode keeps the pdflatex backend from stopping at a
# missing file reference and interactively asking you for an alternative.

report: $(REPORT_FILE)
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make -jobname=build/$(BUILD_NAME) $(REPORT_FILE)

read:
	# Only for macOS
	open -a $(DEFAULT_READER) build/$(BUILD_NAME).pdf

clean:
	find ./build -type f ! -name '$(BUILD_NAME).pdf' -delete
