param(
    [string]$PythonExe = "",
    [string]$Output = "outputs\pilot_F18_F19_P16"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $PythonExe) {
    $bundled = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $bundled) {
        $PythonExe = $bundled
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExe = "python"
    } else {
        throw "Python was not found. Pass -PythonExe with a Python executable containing pandas, numpy, and Pillow."
    }
}

& $PythonExe (Join-Path $ProjectDir "analyze.py") `
    --config (Join-Path $ProjectDir "config.json") `
    --output (Join-Path $ProjectDir $Output)

if ($LASTEXITCODE -ne 0) {
    throw "Analysis failed with exit code $LASTEXITCODE."
}
