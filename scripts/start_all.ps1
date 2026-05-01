$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\env.ps1"
Import-NimbusEnv -Root $Root

$Port = [int]$env:PORT
$QdrantPort = Get-PortFromUrl -Url $env:QDRANT_URL -DefaultPort ([int]$env:QDRANT_PORT)

Set-Location $Root

$qdrant = Get-NetTCPConnection -LocalPort $QdrantPort -State Listen -ErrorAction SilentlyContinue
if (!$qdrant) {
    & "$PSScriptRoot\start_qdrant.ps1"
    Start-Sleep -Seconds 3
}

$nimbus = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (!$nimbus) {
    $Python = Join-Path $Root ".conda\python.exe"
    if (!(Test-Path $Python)) {
        $Python = "python"
    }
    Start-Process -FilePath $Python -ArgumentList "$Root\app.py" -WorkingDirectory $Root -WindowStyle Hidden
}

Write-Host "Nimbus started on configured PORT=$Port"
Write-Host "Qdrant checked using configured QDRANT_URL"
