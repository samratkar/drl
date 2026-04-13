param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$MarkdownFile
)

$ErrorActionPreference = 'Stop'

$src = [System.IO.Path]::GetFullPath($MarkdownFile)

if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
    Write-Error "Error: file not found: $src"
}

$srcItem = Get-Item -LiteralPath $src
$codelabId = $null

foreach ($line in Get-Content -LiteralPath $src) {
    if ($line -match '^id:\s*(.+)\s*$') {
        $codelabId = $Matches[1].Trim()
        break
    }
}

if ([string]::IsNullOrWhiteSpace($codelabId)) {
    Write-Error "Error: no 'id:' field found in frontmatter of $src"
}

$outdir = $srcItem.Directory.FullName
$targetDir = Join-Path $outdir $codelabId

Write-Host "Exporting $src (id: $codelabId) -> $targetDir\"
& claat export -o $outdir $src
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$index = Join-Path $targetDir 'index.html'
if (-not (Test-Path -LiteralPath $index -PathType Leaf)) {
    Write-Error "Error: expected output not found: $index"
}

$content = Get-Content -LiteralPath $index -Raw
$mathMarker = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
$mermaidMarker = 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs'
$styleTag = '<link rel="stylesheet" href="https://storage.googleapis.com/claat-public/codelab-elements.css">'

$mathjaxSnippet = @'
  <script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\(', '\)']],
      displayMath: [['$$', '$$'], ['\[', '\]']]
    }
  };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
'@

$mermaidSnippet = @'
  <script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: false, theme: "default" });

  window.addEventListener("load", async function() {
    if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
    document.querySelectorAll("code.language-mermaid").forEach(function(code) {
      var pre = code.parentElement;
      var div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      pre.parentElement.replaceChild(div, pre);
    });
    await mermaid.run({ querySelector: ".mermaid" });
  });
  </script>
'@

if ($content -notmatch [regex]::Escape($mathMarker)) {
    if ($content.Contains($styleTag)) {
        $content = $content.Replace($styleTag, $styleTag + [Environment]::NewLine + $mathjaxSnippet)
    }
    else {
        Write-Error "Error: stylesheet marker not found in $index"
    }
}

if ($content -notmatch [regex]::Escape($mermaidMarker)) {
    if ($content -match '</body>') {
        $content = $content -replace '</body>', ($mermaidSnippet + [Environment]::NewLine + '</body>')
    }
    else {
        Write-Error "Error: closing </body> tag not found in $index"
    }
}

Set-Content -LiteralPath $index -Value $content -NoNewline
Write-Host "Done. MathJax and Mermaid injected into $index"
