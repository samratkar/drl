#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: md2html.sh INPUT.md [OUTPUT.html]" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

input_path="${1:-}"
output_path="${2:-}"

if [[ $# -gt 2 ]]; then
  usage
fi

[[ -n "$input_path" ]] || usage
[[ -f "$input_path" ]] || { echo "Input file not found: $input_path" >&2; exit 1; }

case "$(printf '%s' "$input_path" | tr '[:upper:]' '[:lower:]')" in
  *.md) ;;
  *) echo "Input must be a Markdown file: $input_path" >&2; exit 1 ;;
esac

markdown_text="$(python - "$input_path" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
print(match.group(1).strip() if match else Path(sys.argv[1]).stem)
PY
)"

if [[ -z "$output_path" ]]; then
  output_path="${input_path%.*}.html"
fi

mkdir -p "$(dirname "$output_path")"

render_html() {
  local md_input="$1"
  local html_output="$2"
  local doc_title="$3"

  pandoc \
    --from=markdown+tex_math_dollars \
    --to=html5 \
    --standalone \
    --mathjax \
    --highlight-style=pygments \
    --metadata "title=$doc_title" \
    --output "$html_output" \
    "$md_input"

  python - "$html_output" "$doc_title" <<'PY'
from html import escape, unescape
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
doc_title = sys.argv[2]
html = path.read_text(encoding="utf-8")

html = re.sub(
    r'<pre class="mermaid"><code>(.*?)</code></pre>',
    lambda m: f'<pre class="mermaid">{unescape(m.group(1))}</pre>',
    html,
    flags=re.S,
)

html = re.sub(
    r"<title>.*?</title>",
    f"<title>{escape(doc_title)}</title>",
    html,
    count=1,
    flags=re.S,
)

injected_head = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
mermaid.initialize({ startOnLoad: true });
</script>
<style>
html {
  background: #f5f7fb;
}
body {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 40px 56px;
  background: #ffffff;
  color: #1f1f1f;
  font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.65;
  box-shadow: 0 8px 24px rgba(60, 64, 67, 0.12);
  border-radius: 16px;
}
header#title-block-header {
  display: none;
}
h1, h2, h3, h4 {
  font-family: Georgia, "Times New Roman", serif;
  font-weight: 700;
  line-height: 1.25;
  color: #9370db;
}
h1 {
  margin-top: 0;
  margin-bottom: 24px;
  font-size: 2.25rem;
  text-align: left;
  letter-spacing: -0.02em;
}
h2 {
  margin-top: 36px;
  margin-bottom: 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e0e3eb;
  font-size: 1.52rem;
}
h3 {
  margin-top: 28px;
  margin-bottom: 10px;
  font-size: 1.22rem;
}
h4 {
  margin-top: 22px;
  margin-bottom: 8px;
  font-size: 1rem;
  color: #3c4043;
}
p, li {
  text-align: left;
}
p {
  margin: 0.75rem 0 1rem;
}
ul, ol {
  margin-top: 0.35rem;
  margin-bottom: 1rem;
  padding-left: 1.5rem;
}
li {
  margin: 0.24rem 0;
}
hr {
  height: 1px;
  border: 0;
  background: #e0e3eb;
  margin: 24px 0;
}
a {
  color: #1a73e8;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
blockquote {
  border-left: 4px solid #1a73e8;
  color: #3c4043;
  background: #f8fbff;
  margin: 1rem 0;
  padding: 0.4rem 1rem;
  border-radius: 0 8px 8px 0;
}
code, pre code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.9em;
}
code {
  background: #f1f3f4;
  padding: 0.12rem 0.3rem;
  border-radius: 4px;
}
pre {
  background: #f8f9fa;
  border: 1px solid #e0e3eb;
  border-radius: 10px;
  padding: 14px 16px;
  line-height: 1.5;
  overflow-x: auto;
  white-space: pre;
}
pre > code {
  display: block;
  white-space: pre;
  word-break: normal;
  overflow-wrap: normal;
  tab-size: 4;
}
table {
  width: auto;
  max-width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin: 1.25rem auto;
  display: inline-table;
  table-layout: auto;
  font-size: 0.96em;
  border: 1px solid #d8bfd8;
  border-top: none;
  border-radius: 10px;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(147, 112, 219, 0.12);
}
th, td {
  border-right: 1px solid #e9d5ff;
  border-bottom: 1px solid #e9d5ff;
  padding: 10px 12px;
  vertical-align: top;
}
th {
  background: #ede9fe;
  color: #202124;
}
thead th {
  font-weight: 700;
  border-top: none !important;
}
tbody tr:nth-child(even) td {
  background: #fbfcfe;
}
tbody tr:hover td {
  background: #faf5ff;
}
tr > *:last-child {
  border-right: 0;
}
tbody tr:last-child td {
  border-bottom: 0;
}
thead tr:first-child th:first-child {
  border-top: none !important;
}
thead tr:first-child th {
  border-top: none !important;
}
.sourceCode,
pre code {
  font-variant-ligatures: none;
}
.math.display {
  overflow-x: auto;
  margin: 1rem 0 1.1rem;
  text-align: center;
}
.math.inline {
  white-space: normal;
}
.mermaid {
  text-align: center;
  margin: 1.2rem 0 1.35rem;
}
pre.mermaid {
  background: transparent;
  border: 0;
  padding: 0;
}
img, svg {
  display: block;
  margin: 1rem auto;
  max-width: 100%;
}
@page {
  margin: 0.6in;
}
@media print {
  html {
    background: white;
  }
  body {
    box-shadow: none;
    background: white;
    padding: 0;
    font-size: 12pt;
    border-radius: 0;
    max-width: none;
  }
  a {
    color: inherit;
  }
}
</style>
"""

html = html.replace("</head>", injected_head + "\n</head>", 1)
path.write_text(html, encoding="utf-8")
PY
}

render_html "$input_path" "$output_path" "$markdown_text"
echo "Rendered HTML written to: $output_path"
