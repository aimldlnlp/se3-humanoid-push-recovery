param(
    [string]$HostName = "aimldl@140.113.149.94",
    [string]$RemoteRoot = "",
    [string]$EnvironmentName = "se3-wbc",
    [string]$KnownHostsFile = "$env:USERPROFILE\.ssh\known_hosts",
    [string]$IdentityFile = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
if (-not (Test-Path -LiteralPath $KnownHostsFile)) {
    throw "Verified SSH known_hosts file was not found: $KnownHostsFile"
}
$sshOptions += @("-o", "UserKnownHostsFile=$KnownHostsFile")
if (-not [string]::IsNullOrWhiteSpace($IdentityFile)) {
    if (-not (Test-Path -LiteralPath $IdentityFile)) {
        throw "SSH identity file was not found: $IdentityFile"
    }
    $sshOptions += @("-i", $IdentityFile)
}
if ([string]::IsNullOrWhiteSpace($RemoteRoot)) {
    throw "RemoteRoot must be provided explicitly for bootstrap."
}
$remoteVenv = "/home/aimldl/.venvs/$EnvironmentName"
$remotePython = "$remoteVenv/bin/python"

& ssh @sshOptions $HostName "mkdir -p '$RemoteRoot'"
if ($LASTEXITCODE -ne 0) { throw "Unable to create remote staging directory" }
& ssh @sshOptions $HostName "mkdir -p '/home/aimldl/.venvs'; if [ ! -x '$remotePython' ]; then python3 -m venv '$remoteVenv'; fi; '$remotePython' -m pip install --upgrade pip; '$remotePython' -m pip install -r '$RemoteRoot/requirements.txt'"
if ($LASTEXITCODE -ne 0) { throw "Remote environment bootstrap failed" }
Write-Host "Remote environment $EnvironmentName is ready"
