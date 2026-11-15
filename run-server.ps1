# Muse Glimmer on this 16 GB AMD laptop.
# Official Q4 (~17 GB) will not load. This uses Unsloth UD-IQ2_XXS (~10.0 GB).
# Close other apps first. Expect slow CPU/iGPU inference.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$model = Join-Path $root "models\Muse-Glimmer-30B-UD-IQ2_XXS.gguf"
$server = Join-Path $root "runtime\llama.cpp\llama-server.exe"

if (-not (Test-Path $server)) { throw "llama-server.exe missing at $server" }
if (-not (Test-Path $model)) { throw "Model file missing. Wait for the download to finish: $model" }
$size = (Get-Item $model).Length
if ($size -lt 9GB) { throw "Model looks incomplete: $size bytes. Re-run the model download." }
$stream = [System.IO.File]::OpenRead($model)
$signature = New-Object byte[] 4
try { $null = $stream.Read($signature, 0, 4) } finally { $stream.Dispose() }
if ([System.Text.Encoding]::ASCII.GetString($signature) -ne "GGUF") { throw "Model is not a valid GGUF file: $model" }

# Short context keeps KV cache small enough for 16 GB RAM.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"
& $server `
  -m $model `
  -a muse-glimmer `
  --jinja `
  --chat-template-kwargs '{"reasoning_strength":"low"}' `
  -ngl 99 `
  -c 4096 `
  -np 1 `
  --host 127.0.0.1 `
  --port 8080
