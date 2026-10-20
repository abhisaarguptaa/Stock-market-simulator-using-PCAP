#!/usr/bin/env bash
# Builds both versions of the report:  bash report/build.sh
# Needs LaTeX:  sudo apt install -y texlive-latex-extra texlive-science texlive-publishers
set -euo pipefail
cd "$(dirname "$0")"
for f in main-ieee main-springer; do
  pdflatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null
  bibtex "$f" > /dev/null
  pdflatex -interaction=nonstopmode "$f" > /dev/null
  pdflatex -interaction=nonstopmode "$f" > /dev/null
done
cp main-ieee.pdf Report_IEEE.pdf
cp main-springer.pdf Report_Springer.pdf
rm -f main-*.aux main-*.bbl main-*.blg main-*.log main-*.out main-*.pdf main-*.out.log
echo "Built report/Report_IEEE.pdf and report/Report_Springer.pdf"
