<#
.SYNOPSIS
    Resets local student-training artifacts and re-runs the full student sweep.

.DESCRIPTION
    DESTRUCTIVE: deletes logs\ and results\ (student run outputs). Teacher checkpoints
    (checkpoints\teachers\) and cached data splits (data\splits\*.json) are left
    untouched on purpose -teachers are trained once (CLAUDE.md) and splits must never
    be regenerated. If a teacher checkpoint is missing (e.g. fresh clone), it's trained
    first -that step is idempotent and skips automatically if already done.

    Runs: student=mobilenetv2_035 x {trashnet, rps_25} x {scratch, kd, rkd} x 3 seeds.

.USAGE
    Run from the repo root in PowerShell:
        .\scripts\reset_and_train_students.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "Clearing logs\ ..."
if (Test-Path "logs") {
    Get-ChildItem "logs" -File | Remove-Item -Force
} else {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "Clearing results\ (student runs only -teacher checkpoints untouched) ..."
if (Test-Path "results") {
    Remove-Item "results" -Recurse -Force
}
New-Item -ItemType Directory -Path "results" -Force | Out-Null

$logFile = "logs\student_sweep.log"

Write-Host "Ensuring teacher checkpoints exist (idempotent -skips if already trained) ..."
foreach ($dataset in @("trashnet", "rps_25")) {
    python -m qakd.train.teacher dataset=$dataset 2>&1 | Tee-Object -FilePath $logFile -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED training teacher for dataset=$dataset" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$datasets = @("trashnet", "rps_25")
$methods  = @("scratch", "kd", "rkd")
$seeds    = @(0, 1, 2)
$student  = "mobilenetv2_035"

foreach ($dataset in $datasets) {
    foreach ($method in $methods) {
        foreach ($seed in $seeds) {
            $line = "=== RUN: dataset=$dataset student=$student method=$method seed=$seed ==="
            Write-Host $line
            Add-Content -Path $logFile -Value $line
            python -m qakd.train dataset=$dataset student=$student method=$method seed=$seed 2>&1 |
                Tee-Object -FilePath $logFile -Append
            if ($LASTEXITCODE -ne 0) {
                Write-Host "FAILED: $line" -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
    }
}

Write-Host "Done -18 student runs written to results\. Launch the dashboard with:"
Write-Host "    streamlit run scripts\dashboard.py"
