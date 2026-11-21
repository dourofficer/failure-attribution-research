#!/bin/bash
# Build <name>.tex -> <name>.pdf (pdflatex + bibtex). Aux files go to build/. Usage: ./build.sh [name], default proposal.
set -e
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
N=${1:-proposal}
mkdir -p build
TEX="pdflatex -interaction=nonstopmode -output-directory=build"
$TEX -halt-on-error $N.tex >/dev/null || { grep -A3 "^!" build/$N.log | head -30; exit 1; }
BIBINPUTS=.: bibtex build/$N >/dev/null || true
$TEX $N.tex >/dev/null
$TEX $N.tex >/dev/null
cp build/$N.pdf $N.pdf
grep -E "^!|undefined" build/$N.log | head
echo "Overfull boxes: $(grep -c Overfull build/$N.log || true)"
grep -E "Output written" build/$N.log
