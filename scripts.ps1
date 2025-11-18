# PowerShell script shortcuts for common tasks

# Install dependencies
function Install-Dependencies {
    Write-Host "Installing dependencies..." -ForegroundColor Green
    pip install -e .
}

# Initialize data
function Initialize-Data {
    Write-Host "Initializing historical data..." -ForegroundColor Green
    python -m app.services.data_update_service --init
}

# Update data
function Update-Data {
    Write-Host "Updating latest data..." -ForegroundColor Green
    python -m app.services.data_update_service --update
}

# Show data summary
function Show-DataSummary {
    Write-Host "Data summary:" -ForegroundColor Green
    python -m app.services.data_update_service --summary
}

# Run tests
function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Green
    pytest tests/ -v
}

# Format code
function Format-Code {
    Write-Host "Formatting code with black..." -ForegroundColor Green
    black .
}

# Run API
function Start-API {
    Write-Host "Starting FastAPI backend on http://localhost:8000" -ForegroundColor Green
    uvicorn app.api.main:app --reload --port 8000
}

# Run UI
function Start-UI {
    Write-Host "Starting Streamlit UI on http://localhost:8501" -ForegroundColor Green
    streamlit run streamlit_app.py
}

# Docker compose up
function Start-Docker {
    Write-Host "Starting Docker services..." -ForegroundColor Green
    docker-compose up --build
}

# Show help
function Show-Help {
    Write-Host "Market Forecasting - Available Commands" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Install-Dependencies  - Install Python dependencies" -ForegroundColor Yellow
    Write-Host "Initialize-Data       - Download historical 2025 data" -ForegroundColor Yellow
    Write-Host "Update-Data          - Fetch latest data" -ForegroundColor Yellow
    Write-Host "Show-DataSummary     - Display data availability" -ForegroundColor Yellow
    Write-Host "Run-Tests            - Execute test suite" -ForegroundColor Yellow
    Write-Host "Format-Code          - Format code with black" -ForegroundColor Yellow
    Write-Host "Start-API            - Run FastAPI backend" -ForegroundColor Yellow
    Write-Host "Start-UI             - Run Streamlit frontend" -ForegroundColor Yellow
    Write-Host "Start-Docker         - Run with Docker Compose" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Example usage:" -ForegroundColor Green
    Write-Host "  . .\scripts.ps1" -ForegroundColor White
    Write-Host "  Initialize-Data" -ForegroundColor White
    Write-Host "  Start-UI" -ForegroundColor White
}

# Export functions
Export-ModuleMember -Function Install-Dependencies, Initialize-Data, Update-Data, Show-DataSummary, Run-Tests, Format-Code, Start-API, Start-UI, Start-Docker, Show-Help

# Show help by default
Show-Help
