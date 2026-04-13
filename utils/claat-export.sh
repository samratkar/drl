#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $(basename "$0") <markdown-file>" >&2
  exit 1
fi

src="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

if [ ! -f "$src" ]; then
  echo "Error: file not found: $src" >&2
  exit 1
fi

# extract the codelab id from frontmatter (line like "id: DRL-001")
codelab_id="$(grep -m1 '^id:' "$src" | sed 's/^id:[[:space:]]*//')"
if [ -z "$codelab_id" ]; then
  echo "Error: no 'id:' field found in frontmatter of $src" >&2
  exit 1
fi

outdir="$(dirname "$src")"

echo "Exporting $src (id: $codelab_id) -> $outdir/$codelab_id/"
claat export -o "$outdir" "$src"

index="$outdir/$codelab_id/index.html"
if [ ! -f "$index" ]; then
  echo "Error: expected output not found: $index" >&2
  exit 1
fi

# --- inject MathJax into <head> ---
mathjax_snippet='  <script>\
  MathJax = {\
    tex: {\
      inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]],\
      displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]]\
    }\
  };\
  </script>\
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>'

sed -i '' "/<link rel=\"stylesheet\" href=\"https:\/\/storage.googleapis.com\/claat-public\/codelab-elements.css\">/a\\
${mathjax_snippet}
" "$index"

# --- inject Mermaid + MathJax re-typeset before </body> ---
mermaid_snippet='  <script type="module">\
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";\
  mermaid.initialize({ startOnLoad: false, theme: "default" });\
\
  window.addEventListener("load", async function() {\
    if (typeof MathJax !== "undefined" \&\& MathJax.typesetPromise) {\
      MathJax.typesetPromise();\
    }\
    document.querySelectorAll("code.language-mermaid").forEach(function(code) {\
      var pre = code.parentElement;\
      var div = document.createElement("div");\
      div.className = "mermaid";\
      div.textContent = code.textContent;\
      pre.parentElement.replaceChild(div, pre);\
    });\
    await mermaid.run({ querySelector: ".mermaid" });\
  });\
  </script>'

sed -i '' "/<\/body>/i\\
${mermaid_snippet}
" "$index"

echo "Done. MathJax and Mermaid injected into $index"
