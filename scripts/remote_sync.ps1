param(
    [ValidateSet("Push", "Pull")]
    [string]$Direction = "Push",
    [string]$HostName = "aimldl@140.113.149.94",
    [string]$RemoteRoot = "/home/aimldl/workspaces/se3-humanoid-push-recovery"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "UserKnownHostsFile=NUL", "-o", "StrictHostKeyChecking=no")
$remoteRootEscaped = $RemoteRoot

if ($Direction -eq "Push") {
    & ssh @sshOptions $HostName "mkdir -p '$remoteRootEscaped'"
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
    $localResults = Join-Path $repoRoot "results"
    New-Item -ItemType Directory -Force -Path $localResults | Out-Null
    & scp @sshOptions -r "$HostName`:$RemoteRoot/results" $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull remote results" }
    Write-Host "Pulled results from $HostName`:$RemoteRoot"
}
