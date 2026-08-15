param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("phase1_simulation", "standing", "perturbed_standing", "single_push", "compare_baseline", "push_calibration", "push_sweep", "robustness", "gpu_benchmark", "run_demo")]
    [string]$Experiment,
    [switch]$PullResults,
    [string]$HostName = "aimldl@140.113.149.94",
    [string]$RemoteRoot = "",
    [string]$EnvironmentName = "se3-wbc",
    [string]$SourceVersion = "",
    [string]$RunId = "",
    [string]$LocalResultsRoot = "",
    [string]$KnownHostsFile = "$env:USERPROFILE\.ssh\known_hosts",
    [string]$IdentityFile = ""
)

$ErrorActionPreference = "Stop"
$scriptRoot = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptRoot
$utc = Get-Date -AsUTC
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "run-$($utc.ToString('yyyyMMddTHHmmssZ'))"
}
if ($RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunId may contain only letters, numbers, dot, underscore, and hyphen."
}
if ([string]::IsNullOrWhiteSpace($SourceVersion)) {
    $SourceVersion = (& git -c "safe.directory=$repoRoot" rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($SourceVersion)) {
        throw "Unable to determine source version."
    }
}
if ([string]::IsNullOrWhiteSpace($RemoteRoot)) {
    $RemoteRoot = "/home/aimldl/workspaces/se3-humanoid-push-recovery-rerun-$RunId"
}
if ([string]::IsNullOrWhiteSpace($LocalResultsRoot)) {
    $LocalResultsRoot = Join-Path $repoRoot "results\staging\$RunId"
}
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

& "$scriptRoot\remote_sync.ps1" -Direction Push -HostName $HostName -RemoteRoot $RemoteRoot -KnownHostsFile $KnownHostsFile -IdentityFile $IdentityFile
if ($LASTEXITCODE -ne 0) { throw "Source sync failed" }

$python = "/home/aimldl/.venvs/$EnvironmentName/bin/python"
$entry = if ($Experiment -eq "run_demo") { "scripts/run_demo.py" } elseif ($Experiment -eq "gpu_benchmark") { "experiments/gpu_benchmark.py" } else { "experiments/$Experiment.py" }
$remoteCommand = "cd '$RemoteRoot' && mkdir -p results/logs results/data results/figures/png results/figures/pdf results/videos && SE3_SOURCE_VERSION='$SourceVersion' SE3_RUN_ID='$RunId' '$python' '$entry' 2>&1 | tee 'results/logs/$Experiment.log'"
& ssh @sshOptions $HostName $remoteCommand
if ($LASTEXITCODE -ne 0) { throw "Remote experiment failed: $Experiment" }

if ($PullResults) {
    & "$scriptRoot\remote_sync.ps1" -Direction Pull -HostName $HostName -RemoteRoot $RemoteRoot -LocalResultsRoot $LocalResultsRoot -KnownHostsFile $KnownHostsFile -IdentityFile $IdentityFile
    if ($LASTEXITCODE -ne 0) { throw "Result sync failed" }
}

Write-Host "Experiment: $Experiment"
Write-Host "Run ID: $RunId"
Write-Host "Source version: $SourceVersion"
Write-Host "Remote root: $RemoteRoot"
Write-Host "Local results root: $LocalResultsRoot"
