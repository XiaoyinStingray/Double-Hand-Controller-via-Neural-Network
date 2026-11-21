# Create / refresh the local conda environment used for this project.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Fresh
#
# Defaults: clone the existing CUDA-enabled env "ComAi" into "hand" (fast, no
# torch re-download). Use -Fresh to build a clean python=3.9 env instead.

param(
    [string]$EnvName = "hand",
    [string]$SourceEnv = "ComAi",
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $envs = conda env list
    $exists = $envs -match "^\s*$EnvName\s"

    if ($exists -and $Fresh) {
        Write-Host "Removing existing env '$EnvName' ..."
        conda env remove -n $EnvName -y
        $exists = $false
    }

    if (-not $exists) {
        $hasSource = $envs -match "^\s*$SourceEnv\s"
        if ($hasSource) {
            Write-Host "Cloning '$SourceEnv' -> '$EnvName' (torch comes with it) ..."
            conda create --clone $SourceEnv -n $EnvName -y
        }
        else {
            Write-Host "Creating clean env '$EnvName' (python=3.9) + torch cu121 ..."
            conda create -n $EnvName python=3.9 -y
            conda run -n $EnvName python -m pip install torch==2.3.0 torchvision==0.18.0 `
                --index-url https://download.pytorch.org/whl/cu121
        }
    }
    else {
        Write-Host "Using existing env '$EnvName'."
    }

    Write-Host "Installing requirements (incl. desktop/demo deps) ..."
    conda run -n $EnvName python -m pip install -r requirements-app.txt

    Write-Host "Downloading model assets ..."
    conda run -n $EnvName python scripts/download_models.py

    Write-Host "Checking environment ..."
    conda run -n $EnvName python scripts/check_env.py --require-app

    Write-Host ""
    Write-Host "Done. Activate with:  conda activate $EnvName"
}
finally {
    Pop-Location
}
