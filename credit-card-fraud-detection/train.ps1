<#
.SYNOPSIS
    Trains and evaluates the Credit Card Fraud Detection model on Windows.

.DESCRIPTION
    Resolves its own project directory, reuses (or creates) the .venv used
    by run.ps1, and runs train_model.py followed by evaluate_model.py.

.PARAMETER SyntheticDemo
    Train on a clearly-labeled SYNTHETIC dataset instead of the real Kaggle
    dataset. For interface demonstration and testing ONLY — resulting
    metrics are not evidence of real fraud-detection performance.

.PARAMETER SkipInstall
    Skip installing requirements.txt before training.

.EXAMPLE
    .\train.ps1

.EXAMPLE
    .\train.ps1 -SyntheticDemo
#>

param(
    [switch]$SyntheticDemo,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
Write-Host "Project directory: $ProjectRoot" -ForegroundColor Cyan

function Get-PythonCommand {
    $candidates = @(
        @{ Cmd = "py"; Args = @("-3.11") },
        @{ Cmd = "py"; Args = @("-3") },
        @{ Cmd = "python"; Args = @() },
        @{ Cmd = "python3"; Args = @() }
    )
    foreach ($c in $candidates) {
        try {
            & $c.Cmd @($c.Args) "--version" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { return $c }
        } catch { continue }
    }
    return $null
}

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    $PythonCmd = Get-PythonCommand
    if (-not $PythonCmd) {
        Write-Host "ERROR: No compatible Python installation was found on PATH." -ForegroundColor Red
        exit 1
    }
    Write-Host "Creating virtual environment at $VenvPath ..." -ForegroundColor Cyan
    & $PythonCmd.Cmd @($PythonCmd.Args) "-m" "venv" $VenvPath
}

if (-not $SkipInstall) {
    Write-Host "Installing dependencies from requirements.txt ..." -ForegroundColor Cyan
    & $VenvPython "-m" "pip" "install" "--upgrade" "pip" | Out-Null
    & $VenvPython "-m" "pip" "install" "-r" (Join-Path $ProjectRoot "requirements.txt")
}

$DatasetPath = Join-Path $ProjectRoot "data\raw\creditcard.csv"

if (-not $SyntheticDemo -and -not (Test-Path $DatasetPath)) {
    Write-Host "ERROR: Real dataset not found at $DatasetPath" -ForegroundColor Red
    Write-Host "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" -ForegroundColor Red
    Write-Host "and place 'creditcard.csv' at the path above." -ForegroundColor Red
    Write-Host "`nAlternatively, run:  .\train.ps1 -SyntheticDemo" -ForegroundColor Yellow
    Write-Host "to exercise the application end-to-end with clearly-labeled synthetic data" -ForegroundColor Yellow
    Write-Host "(this does NOT produce real fraud-detection results)." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Training model ===" -ForegroundColor Cyan
if ($SyntheticDemo) {
    Write-Host "SYNTHETIC DEMO MODE — results will NOT reflect real fraud-detection performance." -ForegroundColor Yellow
    & $VenvPython (Join-Path $ProjectRoot "train_model.py") "--synthetic-demo"
} else {
    & $VenvPython (Join-Path $ProjectRoot "train_model.py")
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed. See the error output above." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=== Evaluating model on held-out test split ===" -ForegroundColor Cyan
& $VenvPython (Join-Path $ProjectRoot "evaluate_model.py")
exit $LASTEXITCODE
