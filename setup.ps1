<#
.SYNOPSIS
    Sets up STG player on a Windows machine: Python environment, the trainer
    source it renders with, the CUDA rasterizer and a desktop shortcut.
    Everything is created inside this repository; nothing outside it is
    changed except the shortcut.

.DESCRIPTION
    Result (all inside the repository folder):

        .venv\                               Python 3.12 virtual environment with the packages
        third_party\SpacetimeGaussians\      upstream trainer at commit 427abfc + upstream\stg_427abfc.patch
                                             (the player imports its camera and loader utilities)
        <Desktop>\STG player.lnk             shortcut that starts the player's menu

    The virtual environment is independent: it does not see the system
    Python's packages, and nothing is installed into the system Python.

    Every step checks whether it is already done, so the script can be run
    again after a failure; it continues where it stopped.

    Prerequisites (installed by hand, see INSTALL.md):
        NVIDIA driver, CUDA Toolkit 13.0, Visual Studio with "Desktop development with C++",
        Python 3.12, git. For VR additionally SteamVR.

.PARAMETER Python
    Python 3.12 executable. Default: found through the "py" launcher, then PATH.

.PARAMETER CudaHome
    CUDA Toolkit folder. Default: CUDA_PATH_V13_0, then the standard install location.

.PARAMETER CudaArch
    GPU architectures for the rasterizer. The default covers RTX 20xx to RTX 50xx.
    A single value such as "8.6" (RTX 30xx) compiles faster.

.PARAMETER RebuildExtensions
    Compile the rasterizer again even if it can be imported.

.PARAMETER NoShortcut
    Do not create the desktop shortcut.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
#>
param(
    [string]$Python = "",
    [string]$CudaHome = "",
    [string]$CudaArch = "7.5;8.0;8.6;8.9;9.0;12.0+PTX",
    [switch]$RebuildExtensions,
    [switch]$NoShortcut
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Repo ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$VenvPyw = Join-Path $Venv "Scripts\pythonw.exe"
$Third = Join-Path $Repo "third_party"
$Stg = Join-Path $Third "SpacetimeGaussians"
$Patch = Join-Path $Repo "upstream\stg_427abfc.patch"
$StgUrl = "https://github.com/oppo-us-research/SpacetimeGaussians.git"
$StgCommit = "427abfc"
$TorchIndex = "https://download.pytorch.org/whl/cu130"
$Extensions = @("gaussian_rasterization_ch3")
$ImportCheck = "import torch, diff_gaussian_rasterization_ch3"

function Step($n, $text) {
    Write-Host ""
    Write-Host "[$n/6] $text" -ForegroundColor Cyan
}

function Ok($text) { Write-Host "  OK $text" -ForegroundColor Green }

function Fail($text) {
    Write-Host "  ERROR $text" -ForegroundColor Red
    exit 1
}

function Run($exe, [string[]]$ArgList, $what) {
    & $exe @ArgList
    if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE)" }
}

function Test-Run($exe, [string[]]$ArgList) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $exe @ArgList *> $null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    return ($code -eq 0)
}

# ------------------------------------------------------------------ 1. prerequisites
Step 1 "Checking prerequisites"

if (-not $Python) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $ErrorActionPreference = "Continue"
            $p = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { $Python = "$p".Trim() }
        } catch { } finally { $ErrorActionPreference = "Stop" }
    }
}
if (-not $Python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $Python = $cmd.Source }
}
if (-not $Python) { Fail "Python 3.12 not found. Install it, or pass -Python <path to python.exe>." }
$ver = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($ver.Trim() -ne "3.12") { Fail "$Python is Python $ver; Python 3.12 is required." }
Ok "Python 3.12: $Python"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git not found on PATH." }
Ok "git found"

if (-not $CudaHome) {
    if ($env:CUDA_PATH_V13_0) { $CudaHome = $env:CUDA_PATH_V13_0 }
    else { $CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" }
}
$Nvcc = Join-Path $CudaHome "bin\nvcc.exe"
if (-not (Test-Path $Nvcc)) { Fail "nvcc not found at $Nvcc. Install CUDA Toolkit 13.0, or pass -CudaHome." }
Ok "CUDA Toolkit: $CudaHome"

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) { Fail "Visual Studio not found (no vswhere.exe)." }
$VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsPath) { Fail "Visual Studio has no C++ tools. Add the workload 'Desktop development with C++'." }
$VcVars = Join-Path $VsPath.Trim() "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $VcVars)) { Fail "vcvars64.bat not found at $VcVars." }
Ok "Visual Studio C++ tools: $($VsPath.Trim())"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { Ok "NVIDIA driver found" }
else { Write-Host "  ! nvidia-smi not found; playback needs an NVIDIA GPU and driver" -ForegroundColor Yellow }

# ------------------------------------------------------------------ 2. Python environment
Step 2 "Python environment (.venv)"
if (-not (Test-Path $VenvPy)) {
    Run $Python @("-m", "venv", $Venv) "Creating the virtual environment"
}
Run $VenvPy @("-m", "pip", "install", "--upgrade", "pip") "Upgrading pip"
Run $VenvPy @("-m", "pip", "install", "-r", (Join-Path $Repo "requirements.txt"),
              "--extra-index-url", $TorchIndex) "Installing packages"
$out = & $VenvPy -c "import torch, numpy, cv2; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, cv2.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "The packages cannot be imported." }
Ok "torch / CUDA / numpy / OpenCV: $out"
if ($out -notmatch " True ") {
    Write-Host "  ! torch does not see a CUDA GPU; check the NVIDIA driver" -ForegroundColor Yellow
}

# ------------------------------------------------------------------ 3. trainer source
Step 3 "Trainer source: SpacetimeGaussians $StgCommit + patch"
New-Item -ItemType Directory -Force $Third | Out-Null
if (-not (Test-Path (Join-Path $Stg ".git"))) {
    # Some upstream test files have very long paths; without longpaths the checkout fails on Windows.
    # autocrlf=false keeps the upstream LF line endings: the patch is LF, and with the
    # common Windows default core.autocrlf=true the checkout would be CRLF and the patch would not apply.
    Run git @("-c", "core.longpaths=true", "-c", "core.autocrlf=false", "clone", $StgUrl, $Stg) "Cloning the trainer"
}
Run git @("-C", $Stg, "config", "core.longpaths", "true") "Configuring the trainer repository"
Run git @("-C", $Stg, "config", "core.autocrlf", "false") "Configuring the trainer repository"
Run git @("-C", $Stg, "-c", "advice.detachedHead=false", "checkout", $StgCommit) "Checking out $StgCommit"
if (Test-Run git @("-C", $Stg, "-c", "core.autocrlf=false", "apply", "--check", $Patch)) {
    Run git @("-C", $Stg, "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn", $Patch) "Applying the patch"
    Ok "patch applied"
} elseif (Test-Run git @("-C", $Stg, "-c", "core.autocrlf=false", "apply", "--reverse", "--check", $Patch)) {
    Ok "patch already applied"
} else {
    Fail "The patch neither applies nor is already applied. The trainer folder has other changes; delete third_party\SpacetimeGaussians and run again."
}

# ------------------------------------------------------------------ 4. CUDA rasterizer
Step 4 "CUDA rasterizer"
if ((-not $RebuildExtensions) -and (Test-Run $VenvPy @("-c", $ImportCheck))) {
    Ok "already built"
} else {
    $Sub = Join-Path $Stg "thirdparty\gaussian_splatting\submodules"
    $Bat = Join-Path $env:TEMP "stg_player_build.bat"
    $lines = @(
        "@echo off",
        "call `"$VcVars`" >nul",
        "set `"CUDA_HOME=$CudaHome`"",
        "set `"CUDA_PATH=$CudaHome`"",
        "set `"PATH=$CudaHome\bin;%PATH%`"",
        "set `"TORCH_CUDA_ARCH_LIST=$CudaArch`"",
        "set `"DISTUTILS_USE_SDK=1`""
    )
    foreach ($m in $Extensions) {
        $lines += "`"$VenvPy`" -m pip install --no-build-isolation `"$Sub\$m`" || exit /b 1"
    }
    Set-Content -Path $Bat -Value $lines -Encoding ASCII
    Write-Host "  compiling $($Extensions -join ', ') for architectures $CudaArch (this takes a while)"
    & cmd.exe /c $Bat
    if ($LASTEXITCODE -ne 0) { Fail "Compiling the rasterizer failed; see the output above." }
    Remove-Item $Bat -ErrorAction SilentlyContinue
    if (-not (Test-Run $VenvPy @("-c", $ImportCheck))) { Fail "The rasterizer was built but cannot be imported." }
    Ok "built and importable"
}

# ------------------------------------------------------------------ 5. self-check
Step 5 "Self-check (renders a few Gaussians on the GPU)"
$old = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $VenvPy (Join-Path $Repo "player\selfcheck.py") 2>&1 | Where-Object { "$_" -match "^\[(OK|FAIL)\]" } | ForEach-Object { Write-Host "  $_" }
$code = $LASTEXITCODE
$ErrorActionPreference = $old
if ($code -ne 0) { Fail "The self-check failed; run .venv\Scripts\python.exe player\selfcheck.py to see the details." }
Ok "the rasterizer runs on this GPU"

# ------------------------------------------------------------------ 6. shortcut
Step 6 "Desktop shortcut"
if ($NoShortcut) {
    Ok "skipped (-NoShortcut)"
} else {
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "STG player.lnk"
    $sh = New-Object -ComObject WScript.Shell
    $s = $sh.CreateShortcut($lnk)
    $s.TargetPath = $VenvPyw
    $s.Arguments = "`"$(Join-Path $Repo 'player\launcher.py')`""
    $s.WorkingDirectory = Join-Path $Repo "player"
    $s.Save()
    Ok "created: $lnk"
}

Write-Host ""
Write-Host "Setup finished. Start the player with the desktop shortcut, or:" -ForegroundColor Green
Write-Host "  `"$VenvPyw`" `"$(Join-Path $Repo 'player\launcher.py')`""
