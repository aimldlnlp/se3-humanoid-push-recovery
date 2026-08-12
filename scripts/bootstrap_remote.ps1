param(
    [string]$HostName = "aimldl@140.113.149.94",
    [string]$RemoteRoot = "/home/aimldl/workspaces/se3-humanoid-push-recovery",
    [string]$EnvironmentName = "se3-wbc"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "UserKnownHostsFile=NUL", "-o", "StrictHostKeyChecking=no")
$conda = "/home/aimldl/miniconda3/bin/conda"
$remotePython = "/home/aimldl/miniconda3/envs/$EnvironmentName/bin/python"

& ssh @sshOptions $HostName "mkdir -p '$RemoteRoot'"
if ($LASTEXITCODE -ne 0) { throw "Unable to create remote staging directory" }
& ssh @sshOptions $HostName "if [ ! -x '$remotePython' ]; then '$conda' create -y -n '$EnvironmentName' python=3.11; fi; '$remotePython' -m pip install -r '$RemoteRoot/requirements.txt'"
if ($LASTEXITCODE -ne 0) { throw "Remote environment bootstrap failed" }
Write-Host "Remote environment $EnvironmentName is ready"
