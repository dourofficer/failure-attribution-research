#!/bin/bash
# Build the failure attribution methods literature survey (main.tex -> main.pdf).
set -e
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
mkdir -p build
TEX="pdflatex -interaction=nonstopmode -output-directory=build"
$TEX -halt-on-error main.tex >/dev/null || { grep -A4 "^!" build/main.log | head -35; exit 1; }
bibtex build/main >/dev/null || { tail -40 build/main.blg; exit 1; }
$TEX main.tex >/dev/null
$TEX main.tex >/dev/null
cp build/main.pdf main.pdf
echo "=== Errors ==="
grep -E "^!" build/main.log | head -20 || true
echo "=== Warnings (undefined citations/refs) ==="
grep -E "Citation.*undefined|Reference.*undefined|Multiply defined" build/main.log | head -20 || true
echo "=== Overfull hbox count ==="
grep -c "Overfull \\\\hbox" build/main.log || true
grep -E "Output written" build/main.log
