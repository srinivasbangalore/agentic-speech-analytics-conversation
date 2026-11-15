$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$model = Join-Path $root "models\speech-analytics\qwen2.5-0.5b-instruct\qwen2.5-0.5b-instruct-q4_k_m.gguf"
$server = Join-Path $root "runtime\llama.cpp\llama-server.exe"

if (-not (Test-Path -LiteralPath $server)) { throw "llama-server.exe missing at $server" }
if (-not (Test-Path -LiteralPath $model)) {
    throw "Small local model missing. Run .venv\Scripts\python.exe server\download_models.py first."
}

$stream = [System.IO.File]::OpenRead($model)
$signature = New-Object byte[] 4
try { $null = $stream.Read($signature, 0, 4) } finally { $stream.Dispose() }
if ([System.Text.Encoding]::ASCII.GetString($signature) -ne "GGUF") { throw "Model is not a valid GGUF file: $model" }

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"
& $server `
  -m $model `
  -a qwen-local `
  -ngl 0 `
  -c 2048 `
  -np 1 `
  --host 127.0.0.1 `
  --port 8081
