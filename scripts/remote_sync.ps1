param(
    [ValidateSet("Push", "Pull")]
    [string]$Direction = "Push",
    [string]$HostName = "aimldl@140.113.149.94",
    [string]$RemoteRoot = "",
    [string]$LocalResultsRoot = "",
    [string]$KnownHostsFile = "$env:USERPROFILE\.ssh\known_hosts",
    [string]$IdentityFile = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")

if ([string]::IsNullOrWhiteSpace($KnownHostsFile)) {
    throw "KnownHostsFile must point to a verified SSH known_hosts file."
}
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
    throw "RemoteRoot must be an isolated run directory."
}
if ([string]::IsNullOrWhiteSpace($LocalResultsRoot)) {
    $LocalResultsRoot = Join-Path $repoRoot "results\staging"
}

if ($Direction -eq "Push") {
    & ssh @sshOptions $HostName "mkdir -p '$RemoteRoot'"
    if ($LASTEXITCODE -ne 0) { throw "Unable to create remote staging directory" }
    $items = @("README.md", "pyproject.toml", "requirements.txt", "configs", "models", "src", "experiments", "scripts", "tests")
    foreach ($item in $items) {
        $source = Join-Path $repoRoot $item
        if (-not (Test-Path -LiteralPath $source)) { continue }
        & scp @sshOptions -r $source "$HostName`:$RemoteRoot/"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload $item" }
    }
    Write-Host "Pushed source to $HostName`:$RemoteRoot"
} else {
    New-Item -ItemType Directory -Force -Path $LocalResultsRoot | Out-Null
    & scp @sshOptions -r "$HostName`:$RemoteRoot/results/*" $LocalResultsRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull remote results" }
    Write-Host "Pulled results from $HostName`:$RemoteRoot/results into $LocalResultsRoot"
}
