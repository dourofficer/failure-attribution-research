#!/bin/bash
# Build notes/workpackages/{wp1,wp2}.tex -> {wp1,wp2}.pdf (pdflatex, aux files in build/).
set -e
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
mkdir -p build
if [ $# -gt 0 ]; then DOCS="$*"; else DOCS="wp1 wp2"; fi
TEX="pdflatex -interaction=nonstopmode -output-directory=build"
for N in $DOCS; do
  $TEX -halt-on-error "$N.tex" >/dev/null || { grep -A5 "^!" "build/$N.log" | head -30; exit 1; }
  $TEX "$N.tex" >/dev/null
  cp "build/$N.pdf" "$N.pdf"
  echo "== $N: $(grep -E 'Output written' "build/$N.log" | sed 's/.*(//;s/).*//'), overfull boxes: $(grep -c Overfull "build/$N.log" || true)"
  grep -E "^!|undefined|Warning.*Reference" "build/$N.log" | sort -u | head
done
