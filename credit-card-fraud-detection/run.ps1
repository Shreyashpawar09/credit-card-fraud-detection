<#
.SYNOPSIS
    Launches the Credit Card Fraud Detection Flask application on Windows.

.DESCRIPTION
    - Resolves its own directory (works no matter where it's invoked from).
    - Detects a compatible Python installation (py launcher or python on PATH).
    - Creates a local .venv if one does not already exist.
    - Installs requirements.txt (unless -SkipInstall is passed).
    - Checks whether the real dataset and a trained model are present and
      explains clearly what is missing, without failing to start the app.
    - Starts the Flask application, letting errors surface normally.

.PARAMETER SkipInstall
    Skip the "pip install -r requirements.txt" step (useful once dependencies
    are already installed, to save time).

.EXAMPLE
    .\run.ps1

.EXAMPLE
    .\run.ps1 -SkipInstall
#>

param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. Resolve project directory (so this works regardless of caller's CWD)
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
Write-Host "Project directory: $ProjectRoot" -ForegroundColor Cyan

# If PowerShell's execution policy blocks this script, re-run it for this
# session only (no permanent, system-wide policy change is made or suggested):
#   powershell -ExecutionPolicy Bypass -File .\run.ps1

# ---------------------------------------------------------------------------
# 2. Detect a compatible Python installation
# ---------------------------------------------------------------------------
function Get-PythonCommand {
    $candidates = @(
        @{ Cmd = "py"; Args = @("-3.11") },
        @{ Cmd = "py"; Args = @("-3") },
        @{ Cmd = "python"; Args = @() },
        @{ Cmd = "python3"; Args = @() }
    )
    foreach ($c in $candidates) {
        try {
            $versionOutput = & $c.Cmd @($c.Args) "--version" 2>&1
            if ($LASTEXITCODE -eq 0) {
                return $c
            }
        } catch {
            continue
        }
    }
    return $null
}

$PythonCmd = Get-PythonCommand
if (-not $PythonCmd) {
    Write-Host "ERROR: No compatible Python installation was found on PATH." -ForegroundColor Red
    Write-Host "Install Python 3.11+ from https://www.python.org/downloads/ and try again." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python command: $($PythonCmd.Cmd) $($PythonCmd.Args -join ' ')" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 3. Create the virtual environment if missing
# ---------------------------------------------------------------------------
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at $VenvPath ..." -ForegroundColor Cyan
    & $PythonCmd.Cmd @($PythonCmd.Args) "-m" "venv" $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Using existing virtual environment at $VenvPath" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 4. Install dependencies (unless skipped)
# ---------------------------------------------------------------------------
if (-not $SkipInstall) {
    Write-Host "Installing dependencies from requirements.txt ..." -ForegroundColor Cyan
    & $VenvPython "-m" "pip" "install" "--upgrade" "pip" | Out-Null
    & $VenvPython "-m" "pip" "install" "-r" (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Dependency installation reported errors. Continuing, but the app may fail to start." -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping dependency installation (-SkipInstall)." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 5. Check dataset and model prerequisites (informational only)
# ---------------------------------------------------------------------------
$DatasetPath = Join-Path $ProjectRoot "data\raw\creditcard.csv"
$ModelPath = Join-Path $ProjectRoot "models\model.joblib"

if (Test-Path $DatasetPath) {
    Write-Host "Dataset found at: $DatasetPath" -ForegroundColor Green
} else {
    Write-Host "Dataset NOT found at: $DatasetPath" -ForegroundColor Yellow
    Write-Host "  Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" -ForegroundColor Yellow
    Write-Host "  and place 'creditcard.csv' at the path above, then run: .\train.ps1" -ForegroundColor Yellow
}

if (Test-Path $ModelPath) {
    Write-Host "Trained model found at: $ModelPath" -ForegroundColor Green
} else {
    Write-Host "No trained model found at: $ModelPath" -ForegroundColor Yellow
    Write-Host "  The app will still start, but predictions will be unavailable until you train a model." -ForegroundColor Yellow
    Write-Host "  Run: .\train.ps1            (trains on the real dataset above)" -ForegroundColor Yellow
    Write-Host "  or:  .\train.ps1 -SyntheticDemo   (trains on synthetic demo data only)" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 6. Start the Flask application (errors are not hidden)
# ---------------------------------------------------------------------------
Write-Host "`nStarting the Flask application on http://127.0.0.1:5000 ..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop.`n" -ForegroundColor Cyan
& $VenvPython (Join-Path $ProjectRoot "app.py")
exit $LASTEXITCODE
