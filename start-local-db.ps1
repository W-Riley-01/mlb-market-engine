<#
.SYNOPSIS
  Check and start the local MLB PostgreSQL cluster (PostgreSQL 17, port 5435).

.DESCRIPTION
  Run this when the local database might not be running (after a reboot,
  logoff or sleep). It checks the usual causes of "can't connect", starts the
  server if needed, and optionally lists the databases and row counts.
  The password is only held in memory for the length of the run. Nothing
  secret is stored in this file.

  Keep this file in the root of the MLB_Analysis folder: it expects the data
  directory (pgdata-mlb) to sit in the folder ABOVE it. Edit the settings
  block below if anything lives somewhere else.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start-local-db.ps1
  powershell -ExecutionPolicy Bypass -File .\start-local-db.ps1 -Stop
#>
[CmdletBinding()]
param([switch]$Stop)

# ---- Settings (edit here if anything moves) ---------------------------------
$PgBin   = 'C:\Program Files\PostgreSQL\17\bin'
$Root    = Split-Path -Parent $PSScriptRoot      # folder that holds MLB_Analysis and pgdata-mlb
$DataDir = Join-Path $Root 'pgdata-mlb'
$LogFile = Join-Path $Root 'pgmlb-server.log'   # kept OUTSIDE the data dir on purpose
$Port    = 5435
$DbName  = 'mlb_engine'
$DbUser  = 'postgres'
# -----------------------------------------------------------------------------

function Ok($m)   { Write-Host "[ OK ] $m" -ForegroundColor Green }
function Info($m) { Write-Host "[ .. ] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Bad($m)  { Write-Host "[FAIL] $m" -ForegroundColor Red }
function Die($m)  { Bad $m; exit 1 }

$pgCtl     = Join-Path $PgBin 'pg_ctl.exe'
$pgIsReady = Join-Path $PgBin 'pg_isready.exe'
$psql      = Join-Path $PgBin 'psql.exe'

function Test-Ready {
    & $pgIsReady -h localhost -p $Port -q
    return ($LASTEXITCODE -eq 0)
}

# 1. PostgreSQL will not run from an elevated (administrator) window.
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
    Die "This window is elevated (Run as administrator). PostgreSQL refuses to run that way. Open a normal PowerShell and run this again."
}
Ok "Running as a normal user"

# 2. Binaries and data directory exist.
foreach ($f in @($pgCtl, $pgIsReady, $psql)) {
    if (-not (Test-Path $f)) { Die "Missing $f - is PostgreSQL 17 installed at $PgBin ?" }
}
if (-not (Test-Path (Join-Path $DataDir 'PG_VERSION'))) {
    Die "No data directory at $DataDir (expected 'pgdata-mlb' in the folder above this script). Edit the settings block at the top."
}
Ok "PostgreSQL binaries and data directory found"

# 3. Clean stop (do this before shutting Windows down, so the next start needs no crash recovery).
if ($Stop) {
    if (-not (Test-Ready)) { Info "Server is not running; nothing to stop."; exit 0 }
    & $pgCtl -D $DataDir stop -m fast
    if ($LASTEXITCODE -ne 0) { Die "pg_ctl could not stop the server." }
    Ok "Server stopped cleanly."
    exit 0
}

# 4. Already up? Otherwise check the port, then start.
if (Test-Ready) {
    Ok "Server already running and accepting connections on port $Port"
}
else {
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $owner = (Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue).ProcessName
        Die "Port $Port is held by '$owner' but is not answering as PostgreSQL. Stop that program or change `$Port in the settings."
    }

    $status = (& $pgCtl -D $DataDir status | Select-Object -First 1)
    Info "pg_ctl says: $status"
    Info "Starting server on port $Port (log: $LogFile) - this can take up to a minute after a crash..."
    & $pgCtl -D $DataDir -o "-p $Port" -l $LogFile -w -t 120 start

    if (($LASTEXITCODE -ne 0) -or (-not (Test-Ready))) {
        Bad "Server did not come up. Last lines of the log:"
        if (Test-Path $LogFile) { Get-Content $LogFile -Tail 20 } else { Warn "No log file at $LogFile" }
        Warn "If the log mentions postmaster.pid, make sure no postgres.exe for this data directory is running (Task Manager), then run this again."
        exit 1
    }
    Ok "Server started and accepting connections on port $Port"
    Info "Before shutting Windows down, run this script with -Stop for a clean shutdown."
}

# 5. Optional database check. The password is read once, kept in memory, and cleared at the end.
Write-Host ""
$sec = Read-Host "postgres password for the database check (press Enter to skip)" -AsSecureString
if ($sec.Length -eq 0) {
    Info "Skipped database check."
    Write-Host ""
    Ok "Ready: localhost:$Port  database=$DbName  user=$DbUser"
    exit 0
}

$failed = $false
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
try {
    $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)

    $dbs = & $psql -h localhost -p $Port -U $DbUser -d postgres -X -A -t -w -c "select datname || ' | ' || pg_size_pretty(pg_database_size(datname)) from pg_database where not datistemplate order by 1;"
    if ($LASTEXITCODE -ne 0) {
        Bad "Could not log in as '$DbUser' (wrong password?)."
        $failed = $true
    }
    else {
        Ok "Databases on port ${Port}:"
        $dbs | ForEach-Object { Write-Host "       $_" }

        if (-not ($dbs | Where-Object { $_ -like "$DbName |*" })) {
            Warn "Database '$DbName' was not found - check the list above."
        }
        else {
            $q = @'
select table_name || ' | ' || (xpath('/row/c/text()', query_to_xml(format('select count(*) as c from public.%I', table_name), false, true, '')))[1]::text from information_schema.tables where table_schema = 'public' and table_type = 'BASE TABLE' order by table_name;
'@
            $counts = & $psql -h localhost -p $Port -U $DbUser -d $DbName -X -A -t -w -c $q
            if (($LASTEXITCODE -ne 0) -or (-not $counts)) {
                Warn "Could not read row counts from '$DbName' (no tables?)."
            }
            else {
                Ok "Row counts in ${DbName}:"
                $counts | ForEach-Object { Write-Host "       $_" }
            }
        }
    }
}
finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

Write-Host ""
if ($failed) { exit 1 }
Ok "Ready: localhost:$Port  database=$DbName  user=$DbUser"
exit 0
