$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\env.ps1"
Import-NimbusEnv -Root $Root

Set-Location $Root
$Python = Join-Path $Root ".conda\python.exe"
if (!(Test-Path $Python)) {
    $Python = "python"
}
& $Python "$Root\app.py"
