function Import-NimbusEnv {
    param([string]$Root)

    $envPath = Join-Path $Root ".env"
    if (!(Test-Path $envPath)) {
        throw "Missing secure config file: $envPath"
    }

    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if (!$line -or $line.StartsWith("#") -or !$line.Contains("=")) {
            return
        }
        $key, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim().Trim('"').Trim("'"), "Process")
    }
}

function Get-PortFromUrl {
    param(
        [string]$Url,
        [int]$DefaultPort
    )

    try {
        $uri = [Uri]$Url
        if ($uri.Port -gt 0) {
            return $uri.Port
        }
    } catch {
        return $DefaultPort
    }
    return $DefaultPort
}

