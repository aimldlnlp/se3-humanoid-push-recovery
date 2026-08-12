param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("phase1_simulation", "standing", "perturbed_standing", "single_push", "compare_baseline", "push_sweep", "robustness", "gpu_benchmark", "run_demo")]
    [string]$Experiment,
    [switch]$PullResults,
    [string]$HostName = "aimldl@140.113.149.94",
    [string]$RemoteRoot = "/home/aimldl/workspaces/se3-humanoid-push-recovery",
    [string]$EnvironmentName = "se3-wbc"
)

$ErrorActionPreference = "Stop"
$scriptRoot = $PSScriptRoot
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "UserKnownHostsFile=NUL", "-o", "StrictHostKeyChecking=no")

& "$scriptRoot\remote_sync.ps1" -Direction Push -HostName $HostName -RemoteRoot $RemoteRoot
if ($LASTEXITCODE -ne 0) { throw "Source sync failed" }

$python = "/home/aimldl/miniconda3/envs/$EnvironmentName/bin/python"
$entry = if ($Experiment -eq "run_demo") { "scripts/run_demo.py" } elseif ($Experiment -eq "gpu_benchmark") { "experiments/gpu_benchmark.py" } else { "experiments/$Experiment.py" }
$remoteCommand = "cd '$RemoteRoot' && mkdir -p results/logs results/data results/figures/png results/figures/pdf results/videos && '$python' '$entry' 2>&1 | tee 'results/logs/$Experiment.log'"
& ssh @sshOptions $HostName $remoteCommand
if ($LASTEXITCODE -ne 0) { throw "Remote experiment failed: $Experiment" }

if ($PullResults) {
    & "$scriptRoot\remote_sync.ps1" -Direction Pull -HostName $HostName -RemoteRoot $RemoteRoot
    if ($LASTEXITCODE -ne 0) { throw "Result sync failed" }
}
