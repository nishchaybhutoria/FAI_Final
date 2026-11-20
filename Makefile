# CS 329 final project: reproduction targets. Run `make help` for a summary.
#
# PYTHON  interpreter used to create the virtual environment (Python 3.11 or 3.12)
# VENV    virtual-environment directory (must be on a filesystem that supports symlinks)
# PY      interpreter used by every other target: the venv's Python if it exists, else PYTHON
# WORKERS optional number of worker processes for the simulation targets
# The report build fixes the PDF timestamp (SOURCE_DATE_EPOCH) so rebuilding gives an identical file.
PYTHON  ?= python3
VENV    ?= .venv
PY      ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(PYTHON))
WORKERS ?=
W := $(if $(WORKERS),--workers $(WORKERS),)
export OMP_NUM_THREADS := 1
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: help setup test demo verify reproduce reproduce-full figures report clean

help:
	@echo "make setup           create $(VENV) and install pinned dependencies"
	@echo "make test            run the test suite (about 30 s)"
	@echo "make demo            simulate one held-out stream under 8 methods; write results/screenshots/"
	@echo "make verify          quick check: rerun ~100 recorded runs exactly + recheck every report number"
	@echo "make reproduce       recompute all report numbers/tables from results/data, check them, redraw figures"
	@echo "make reproduce-full  rerun all studies with the original seeds (~1.9 CPU-hours) and check bit-identity"
	@echo "make figures         redraw every figure from results/data"
	@echo "make report          compile report/CS329_Final_Report.pdf (needs pdflatex and bibtex)"
	@echo "make clean           remove LaTeX by-products, regenerated reruns and caches"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q tests

demo:
	$(PY) experiments/demo.py

verify:
	$(PY) experiments/run_experiments.py --mode quick $(W)
	$(PY) experiments/reproduce_report.py --check

reproduce:
	$(PY) experiments/reproduce_report.py --check
	$(PY) experiments/generate_figures.py

reproduce-full:
	$(PY) experiments/run_experiments.py --mode full $(W)
	$(PY) experiments/reproduce_report.py --data results/rerun --check

figures:
	$(PY) experiments/generate_figures.py

report: export SOURCE_DATE_EPOCH := 1795132800
report: export FORCE_SOURCE_DATE := 1
report:
	cd report && pdflatex -interaction=nonstopmode -halt-on-error -jobname=CS329_Final_Report main.tex >/dev/null && \
	bibtex CS329_Final_Report >/dev/null && \
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=CS329_Final_Report main.tex >/dev/null && \
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=CS329_Final_Report main.tex >/dev/null
	@grep -E "Warning|Overfull|undefined" report/CS329_Final_Report.log || echo "report/CS329_Final_Report.pdf built without warnings"

clean:
	cd report && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.loa build.txt main.pdf
	rm -rf results/rerun results/check .pytest_cache .hypothesis
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
