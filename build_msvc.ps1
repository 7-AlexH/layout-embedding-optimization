# build_msvc.ps1
# One-shot configure + build helper for Windows / MSVC.
# Runs everything inside a single cmd invocation that first sets up the vcvars64
# environment, so the MSVC toolchain is on PATH for both configure and build.
#
# Usage:
#   .\build_msvc.ps1                 # configure + build msvc-release
#   .\build_msvc.ps1 -Preset msvc-debug
#   .\build_msvc.ps1 -Target optimize
#   .\build_msvc.ps1 -ConfigureOnly

param(
    [string]$Preset = "msvc-release",
    [string]$Target = "",
    [switch]$ConfigureOnly
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath  = & $vswhere -latest -products * -property installationPath
$vcvars  = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$buildArgs = "--build --preset $Preset"
if ($Target) { $buildArgs += " --target $Target" }

# Build the command sequence to run inside cmd. Native tool output (incl. stderr)
# stays inside cmd so PowerShell never misclassifies it as a terminating error.
$cmds = @(
    "call `"$vcvars`"",
    "cd /d `"$root`"",
    "cmake --preset $Preset"
)
if (-not $ConfigureOnly) {
    # NOTE: do not insert "if errorlevel 1 exit /b 1" here — cmd would parse the
    # following "&& cmake --build" as part of the if-body, silently skipping the
    # build when configure SUCCEEDS. && already short-circuits on failure.
    $cmds += "cmake $buildArgs"
}
$full = ($cmds -join " && ")

Write-Host ">>> $full"
cmd /c $full
exit $LASTEXITCODE
