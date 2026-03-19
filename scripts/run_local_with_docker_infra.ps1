Param(
    [string]$BindAddress,
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BindAddress)) {
    $BindAddress = "0.0.0.0"
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Write-Host "Starting required Docker services (postgres, mongodb, redis, worker)..." -ForegroundColor Cyan

# Fail fast if Docker daemon isn't available (common on Windows when Docker Desktop isn't started).
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker daemon is not available." -ForegroundColor Red
    Write-Host "- Start Docker Desktop and wait until it says 'Running'" -ForegroundColor Yellow
    Write-Host "- Then re-run this script" -ForegroundColor Yellow
    exit 1
}

# Bring up only the infra + worker needed for Analysis Overview
& docker compose up -d postgres mongodb redis worker | Out-Host

Write-Host "Waiting for containers to become healthy..." -ForegroundColor Cyan
$maxSeconds = 60
$start = Get-Date

function Get-Health($name) {
    try {
        $json = & docker inspect $name | ConvertFrom-Json
        return $json[0].State.Health.Status
    } catch {
        return $null
    }
}

while ($true) {
    $pg = Get-Health "sales-strategist-postgres"
    $mg = Get-Health "sales-strategist-mongodb"
    $rd = Get-Health "sales-strategist-redis"

    if ($pg -eq "healthy" -and $mg -eq "healthy" -and $rd -eq "healthy") { break }

    if (((Get-Date) - $start).TotalSeconds -ge $maxSeconds) {
        Write-Host "Timed out waiting for healthy containers." -ForegroundColor Yellow
        & docker compose ps | Out-Host
        break
    }

    Start-Sleep -Seconds 2
}

Write-Host "Running Flask API locally against Docker-host ports..." -ForegroundColor Green
Write-Host "- API: http://localhost:$Port" -ForegroundColor DarkGray
Write-Host "- Dashboard: http://localhost:$Port/dashboard/analysis-overview" -ForegroundColor DarkGray

# Use .env values (already mapped to host ports like 5433/27018/6380)
# and force post-analysis sync mode to avoid Redis/Celery issues in the local process.
$env:POST_ANALYSIS_USE_CELERY = "0"

& $PythonExe -m flask --app src.api.app run --host $BindAddress --port $Port
