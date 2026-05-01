$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\env.ps1"
Import-NimbusEnv -Root $Root

$runtime = $env:QDRANT_RUNTIME
if (!$runtime -or $runtime.ToLowerInvariant() -ne "docker") {
    throw "Only Docker Qdrant runtime is configured. Set QDRANT_RUNTIME=docker in .env."
}

$container = $env:QDRANT_CONTAINER_NAME
$image = $env:QDRANT_DOCKER_IMAGE
$volume = $env:QDRANT_DOCKER_VOLUME
$port = [int]$env:QDRANT_PORT
if (!$container -or !$image -or !$volume) {
    throw "Docker Qdrant config is incomplete in .env"
}

$existing = docker ps -a --filter "name=^/$container$" --format "{{.Names}}"
if (!$existing) {
    docker volume create $volume | Out-Null
    docker run -d --name $container -p "${port}:6333" -p "6334:6334" -v "${volume}:/qdrant/storage" $image | Out-Null
} else {
    docker start $container | Out-Null
}
