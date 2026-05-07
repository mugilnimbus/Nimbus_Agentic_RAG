param(
    [ValidateSet("start", "stop", "restart", "status", "start-qdrant", "stop-qdrant")]
    [string]$Action = "status",
    [switch]$Foreground,
    [switch]$KeepQdrant
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\env.ps1"
Import-NimbusEnv -Root $Root

$RuntimeDir = Join-Path $Root "data\runtime"
$PidFile = Join-Path $RuntimeDir "nimbus.pid"
$OutLog = Join-Path $RuntimeDir "nimbus.out.log"
$ErrLog = Join-Path $RuntimeDir "nimbus.err.log"

function Ensure-RuntimeDir {
    if (!(Test-Path $RuntimeDir)) {
        New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
    }
}

function Get-NimbusPort {
    if ($env:PORT) {
        return [int]$env:PORT
    }
    return 8000
}

function Get-QdrantPort {
    $default = 6333
    if ($env:QDRANT_PORT) {
        $default = [int]$env:QDRANT_PORT
    }
    return Get-PortFromUrl -Url $env:QDRANT_URL -DefaultPort $default
}

function Test-PortListening {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Get-PortOwners {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    $processIds = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($processId in $processIds) {
        if ($processId) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
        }
    }
}

function Test-HttpOk {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 2
    )
    try {
        Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-HttpOk {
    param(
        [string]$Url,
        [int]$Seconds = 20
    )
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpOk -Url $Url -TimeoutSeconds 2) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Get-PythonPath {
    $localPython = Join-Path $Root ".conda\python.exe"
    if (Test-Path $localPython) {
        return $localPython
    }
    return "python"
}

function Get-NimbusPidFromFile {
    if (!(Test-Path $PidFile)) {
        return $null
    }
    $raw = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    $pidValue = 0
    if ([int]::TryParse($raw, [ref]$pidValue)) {
        return $pidValue
    }
    return $null
}

function Get-NimbusProcesses {
    $rootLower = $Root.ToLowerInvariant()
    return Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "^python(\.exe)?$" -and
            (Test-IsNimbusProcess $_)
        }
}

function Test-IsNimbusProcess {
    param($Process)

    if (!$Process) {
        return $false
    }

    $name = [string]$Process.Name
    $commandLine = [string]$Process.CommandLine
    $executablePath = [string]$Process.ExecutablePath
    $rootLower = $Root.ToLowerInvariant()
    $localPython = (Join-Path $Root ".conda\python.exe").ToLowerInvariant()

    $isPython = $name -match "^python(\.exe)?$"
    $usesProjectRoot = $commandLine.ToLowerInvariant().Contains($rootLower)
    $runsApp = $commandLine.ToLowerInvariant().Contains("app.py")
    $usesProjectPython = $executablePath -and $executablePath.ToLowerInvariant() -eq $localPython

    return $isPython -and (($usesProjectRoot -and $runsApp) -or $usesProjectPython)
}

function Stop-ProcessAndWait {
    param(
        [int]$ProcessId,
        [string]$Label = "process"
    )

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (!$process) {
        return $false
    }

    Write-Host "Stopping $Label PID $ProcessId..."
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue

    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        if (!(Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }

    return !(Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Wait-PortFree {
    param(
        [int]$Port,
        [int]$Seconds = 10
    )

    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (!(Test-PortListening -Port $Port)) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    return !(Test-PortListening -Port $Port)
}

function Format-PortOwners {
    param([int]$Port)
    $owners = @(Get-PortOwners -Port $Port)
    if (!$owners.Count) {
        return "none"
    }
    return ($owners | ForEach-Object {
        $path = if ($_.ExecutablePath) { $_.ExecutablePath } else { $_.Name }
        "PID $($_.ProcessId) $path"
    }) -join "; "
}

function Start-Qdrant {
    $runtime = $env:QDRANT_RUNTIME
    if (!$runtime) {
        $runtime = ""
    }
    $runtime = $runtime.ToLowerInvariant()
    if ($runtime -ne "docker") {
        throw "QDRANT_RUNTIME must be docker. Current value: '$runtime'."
    }

    $container = $env:QDRANT_CONTAINER_NAME
    $image = $env:QDRANT_DOCKER_IMAGE
    $volume = $env:QDRANT_DOCKER_VOLUME
    $port = Get-QdrantPort
    if (!$container -or !$image -or !$volume) {
        throw "Docker Qdrant config is incomplete in .env."
    }
    if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is not available. Start Docker Desktop and try again."
    }

    $existing = docker ps -a --filter "name=^/$container$" --format "{{.Names}}"
    if (!$existing) {
        Write-Host "Creating Qdrant container '$container'..."
        docker volume create $volume | Out-Null
        docker run -d --name $container -p "${port}:6333" -p "6334:6334" -v "${volume}:/qdrant/storage" $image | Out-Null
    } else {
        $running = docker ps --filter "name=^/$container$" --filter "status=running" --format "{{.Names}}"
        if (!$running) {
            Write-Host "Starting Qdrant container '$container'..."
            docker start $container | Out-Null
        }
    }

    $qdrantUrl = $env:QDRANT_URL
    if (!$qdrantUrl) {
        $qdrantUrl = "http://127.0.0.1:$port"
    }
    if (!(Wait-HttpOk -Url "$qdrantUrl/collections" -Seconds 30)) {
        throw "Qdrant did not become ready at $qdrantUrl."
    }
    Write-Host "Qdrant is ready at $qdrantUrl"
}

function Stop-Qdrant {
    $runtime = $env:QDRANT_RUNTIME
    if (!$runtime) {
        $runtime = ""
    }
    $runtime = $runtime.ToLowerInvariant()
    if ($runtime -eq "docker" -and $env:QDRANT_CONTAINER_NAME) {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            $running = docker ps --filter "name=^/$($env:QDRANT_CONTAINER_NAME)$" --filter "status=running" --format "{{.Names}}"
            if ($running) {
                Write-Host "Stopping Qdrant container '$($env:QDRANT_CONTAINER_NAME)'..."
                docker stop $env:QDRANT_CONTAINER_NAME | Out-Null
            }
        }
    }
}

function Start-Nimbus {
    param([switch]$Foreground)

    $port = Get-NimbusPort
    if (Test-PortListening -Port $port) {
        Write-Host "Nimbus port $port is already in use; checking for a stale Nimbus process..."
        Stop-Nimbus
        if (!(Wait-PortFree -Port $port -Seconds 10)) {
            throw "Port $port is still in use by: $(Format-PortOwners -Port $port)"
        }
    }

    $python = Get-PythonPath
    Set-Location $Root
    if ($Foreground) {
        Write-Host "Starting Nimbus in foreground on port $port..."
        & $python "$Root\app.py"
        return
    }

    Ensure-RuntimeDir
    Write-Host "Starting Nimbus in background on port $port..."
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList "`"$Root\app.py`"" `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Path $PidFile -Value $process.Id

    $healthUrl = "http://127.0.0.1:$port/api/health"
    if (!(Wait-HttpOk -Url $healthUrl -Seconds 30)) {
        Write-Warning "Nimbus was started as PID $($process.Id), but health did not respond at $healthUrl."
        Write-Warning "Check logs: $OutLog and $ErrLog"
        return
    }
    Write-Host "Nimbus is ready at http://127.0.0.1:$port"
}

function Stop-Nimbus {
    $stopped = $false
    $port = Get-NimbusPort
    $pidValue = Get-NimbusPidFromFile
    if ($pidValue) {
        $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($process) {
            $stopped = (Stop-ProcessAndWait -ProcessId $pidValue -Label "Nimbus") -or $stopped
        }
        Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
    }

    foreach ($process in Get-NimbusProcesses) {
        $stopped = (Stop-ProcessAndWait -ProcessId $process.ProcessId -Label "Nimbus process") -or $stopped
    }

    foreach ($process in Get-PortOwners -Port $port) {
        if (Test-IsNimbusProcess $process) {
            $stopped = (Stop-ProcessAndWait -ProcessId $process.ProcessId -Label "Nimbus port owner") -or $stopped
        }
    }

    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
    Wait-PortFree -Port $port -Seconds 10 | Out-Null

    if (!$stopped) {
        Write-Host "Nimbus is not running."
    }
}

function Show-Status {
    $port = Get-NimbusPort
    $qdrantPort = Get-QdrantPort
    $qdrantUrl = $env:QDRANT_URL
    if (!$qdrantUrl) {
        $qdrantUrl = "http://127.0.0.1:$qdrantPort"
    }

    $nimbusHealth = Test-HttpOk -Url "http://127.0.0.1:$port/api/health"
    $qdrantHealth = Test-HttpOk -Url "$qdrantUrl/collections"
    $pidValue = Get-NimbusPidFromFile
    $portOwners = Format-PortOwners -Port $port

    Write-Host "Nimbus"
    Write-Host "  URL:    http://127.0.0.1:$port"
    Write-Host "  Health: $(if ($nimbusHealth) { 'ready' } else { 'not responding' })"
    Write-Host "  PID:    $(if ($pidValue) { $pidValue } else { 'unknown' })"
    Write-Host "  Port:   $portOwners"
    Write-Host "  Logs:   $OutLog"
    Write-Host ""
    Write-Host "Qdrant"
    Write-Host "  URL:    $qdrantUrl"
    Write-Host "  Health: $(if ($qdrantHealth) { 'ready' } else { 'not responding' })"
    $runtime = $env:QDRANT_RUNTIME
    if (!$runtime) {
        $runtime = ""
    }
    if ($runtime.ToLowerInvariant() -eq "docker" -and $env:QDRANT_CONTAINER_NAME -and (Get-Command docker -ErrorAction SilentlyContinue)) {
        $dockerStatus = docker ps -a --filter "name=^/$($env:QDRANT_CONTAINER_NAME)$" --format "{{.Names}} {{.Status}}"
        Write-Host "  Docker: $(if ($dockerStatus) { $dockerStatus } else { 'container missing' })"
    }
}

switch ($Action) {
    "start" {
        Start-Qdrant
        Start-Nimbus -Foreground:$Foreground
    }
    "stop" {
        Stop-Nimbus
        if (!$KeepQdrant) {
            Stop-Qdrant
        }
    }
    "restart" {
        Stop-Nimbus
        if (!$KeepQdrant) {
            Stop-Qdrant
        }
        Start-Sleep -Seconds 1
        Start-Qdrant
        Start-Nimbus -Foreground:$Foreground
    }
    "status" {
        Show-Status
    }
    "start-qdrant" {
        Start-Qdrant
    }
    "stop-qdrant" {
        Stop-Qdrant
    }
}
