# setup_msvc_env.ps1
# Imports the MSVC (x64) build environment into the current PowerShell session by
# running vcvars64.bat and copying the resulting environment variables across.
# Usage:  . .\setup_msvc_env.ps1     (dot-source so the env changes persist)

$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at $vswhere - is Visual Studio installed?"
}

$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) { throw "No Visual Studio installation found." }

$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

Write-Host "Importing MSVC x64 environment from: $vcvars"

# Run vcvars64.bat in a child cmd, then dump the environment and re-import it.
$tmp = [System.IO.Path]::GetTempFileName()
cmd /c "`"$vcvars`" >nul 2>&1 && set" > $tmp

Get-Content $tmp | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}
Remove-Item $tmp

Write-Host "MSVC environment ready."
$cl = (Get-Command cl -ErrorAction SilentlyContinue)
if ($cl) { Write-Host "cl found at: $($cl.Source)" }
