# One-shot smoke test against llama-server (run-server.ps1 must already be running).
param(
  [string]$Prompt = "In one short sentence, what is Muse Glimmer?"
)

$body = @{
  model = "muse-glimmer"
  temperature = 0
  messages = @(@{ role = "user"; content = $Prompt })
} | ConvertTo-Json -Depth 6

$resp = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
Write-Host "--- reasoning ---"
Write-Host $resp.choices[0].message.reasoning_content
Write-Host "--- answer ---"
Write-Host $resp.choices[0].message.content
