$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EscapedRoot = [WildcardPattern]::Escape($Root)
. "$PSScriptRoot\env.ps1"
Import-NimbusEnv -Root $Root

Get-CimInstance Win32_Process |
    Where-Object {
        ($_.Name -eq "python.exe" -and $_.CommandLine -like "*$EscapedRoot*app.py*") -or
        ($_.Name -eq "qdrant.exe" -and $_.CommandLine -like "*$EscapedRoot*qdrant*")
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force
    }

$runtime = $env:QDRANT_RUNTIME
if (!$runtime) {
    $runtime = ""
}
if ($runtime.ToLowerInvariant() -eq "docker" -and $env:QDRANT_CONTAINER_NAME) {
    docker stop $env:QDRANT_CONTAINER_NAME | Out-Null
}

Write-Host "Stopped Nimbus and Qdrant processes for $Root"
