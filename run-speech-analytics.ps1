$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$localPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $localPython) {
    $pythonPath = $localPython
}
else {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if (-not $python) {
        $python = Get-Command python -ErrorAction SilentlyContinue
    }
    if ($python) {
        $pythonPath = $python.Source
    }
}
if (-not $pythonPath) {
    throw "Python 3.10 or later is required. Install Python, then run this script again."
}
& $pythonPath -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 or later is required. Install Python, then run this script again."
}

$envFile = Join-Path $root "server\.env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Configuration is missing. Copy server\.env.example to server\.env."
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"

Push-Location $root
try {
    & $pythonPath -m uvicorn app:app --app-dir (Join-Path $root "server") --host 127.0.0.1 --port 8000
    if ($LASTEXITCODE -ne 0) {
        throw "The speech analytics server exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
