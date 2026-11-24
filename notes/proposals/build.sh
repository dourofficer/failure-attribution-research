#!/bin/bash
# Build the documents of notes/proposals/: <name>.tex -> <name>.pdf (pdflatex + bibtex). Aux files go to build/.
# Usage: ./build.sh            every pps*.tex and case_studies.tex
#        ./build.sh <name>...  only the named documents (without .tex)
# pps00--pps05 share preamble.tex and refer to each other and to the survey through xr-hyper. For that, each one's labels are
# copied to build/<name>_labels.aux (its .aux minus the \bibcite lines) and the survey's to build/main_labels.aux, taken from
# ../datasets/build/main.aux; run ../datasets/build.sh first if that file is missing, otherwise survey references print as ??.
# Two rounds: every document is compiled on its own first, so that the labels exist before any other document reads them.
set -e
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
mkdir -p build
if [ $# -gt 0 ]; then DOCS="$*"; else DOCS="$(ls pps[0-9][0-9]_*.tex case_studies.tex | sed 's/\.tex$//')"; fi
TEX="pdflatex -interaction=nonstopmode -output-directory=build"
if [ -f ../datasets/build/main.aux ]; then grep -v '\\bibcite' ../datasets/build/main.aux > build/main_labels.aux
elif [ ! -f build/main_labels.aux ]; then echo "warning: ../datasets/build/main.aux is missing, survey references will print as ??"; fi
for N in $DOCS; do
  $TEX -halt-on-error $N.tex >/dev/null || { grep -A3 "^!" build/$N.log | head -30; exit 1; }
  BIBINPUTS=.: bibtex build/$N >/dev/null || true
  $TEX $N.tex >/dev/null
  grep -v '\\bibcite' build/$N.aux > build/${N}_labels.aux
done
for N in $DOCS; do
  $TEX $N.tex >/dev/null
  $TEX $N.tex >/dev/null
  cp build/$N.pdf $N.pdf
  echo "== $N: $(grep -E 'Output written' build/$N.log | sed 's/.*(//;s/).*//'), overfull boxes: $(grep -c Overfull build/$N.log || true)"
  grep -E "^!|undefined" build/$N.log | sort -u | head
done
