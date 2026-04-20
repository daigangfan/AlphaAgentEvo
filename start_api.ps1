# Launch the backtest API server
# Usage: .\start_api.ps1

$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

Write-Host "============================================"
Write-Host "  AlphaAgentEvo Backtest Server"
Write-Host "============================================"

# Activate environment: prefer conda 'alphaevo', fall back to uv .venv
if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda activate alphaevo 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: alphaevo conda env not found, using current env"
    }
} elseif (Test-Path (Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1')) {
    & (Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1')
} else {
    Write-Host "Warning: no conda env or .venv found, using system Python"
}

Write-Host "Starting FastAPI server on port 8001..."
python -m uvicorn backtest.api_server:app `
    --host 0.0.0.0 `
    --port 8001 `
    --workers 1 `
    --log-level info
