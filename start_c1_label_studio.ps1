$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LabelStudio = Join-Path $ProjectRoot ".venv-label313\Scripts\label-studio.exe"
$DataDir = Join-Path $ProjectRoot ".label-studio"

if (-not (Test-Path -LiteralPath $LabelStudio)) {
    throw "Label Studio is not installed at $LabelStudio"
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$env:LABEL_STUDIO_BASE_DATA_DIR = $DataDir

& $LabelStudio start --port 8080
