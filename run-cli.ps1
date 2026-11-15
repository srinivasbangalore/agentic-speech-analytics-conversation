# Single-turn CLI chat with Muse Glimmer (IQ2_XXS, 16 GB laptop settings).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$model = Join-Path $root "models\Muse-Glimmer-30B-UD-IQ2_XXS.gguf"
$cli = Join-Path $root "bin\llama.cpp\llama-cli.exe"
$expected = 10746373152

if (-not (Test-Path $cli)) { throw "llama-cli.exe missing at $cli" }
if (-not (Test-Path $model)) { throw "Model file missing: $model" }
if ((Get-Item $model).Length -ne $expected) { throw "Model incomplete. Wait for the download." }

& $cli -m $model -ngl 99 -c 4096 --jinja -st @args
