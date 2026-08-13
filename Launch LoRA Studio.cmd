@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LoRA Fine-tune Studio

if not exist "pyproject.toml" goto :wrong_folder
if not exist "uv.lock" goto :wrong_folder
if not exist "streamlit_app.py" goto :wrong_folder

set "UV_EXE="
for /f "delims=" %%I in ('where uv.exe 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"

if not defined UV_EXE (
    echo Installing uv for this Windows user...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 ^| iex"
    if errorlevel 1 goto :uv_failed
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
)

if not defined UV_EXE goto :uv_failed

echo Preparing Python 3.14, .venv, and project dependencies...
"%UV_EXE%" sync --locked --no-dev --python 3.14
if errorlevel 1 goto :sync_failed

if not exist ".venv\Scripts\pythonw.exe" goto :sync_failed

echo Preparing Python 3.13, .venv-unsloth, and native Unsloth dependencies...
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv-unsloth"
"%UV_EXE%" sync --project "unsloth-runtime" --locked --no-dev --python 3.13.13
if errorlevel 1 (
    set "UV_PROJECT_ENVIRONMENT="
    goto :unsloth_sync_failed
)
set "UV_PROJECT_ENVIRONMENT="

if not exist ".venv-unsloth\Scripts\python.exe" goto :unsloth_sync_failed
if not exist ".runs" mkdir ".runs"

echo Starting LoRA Fine-tune Studio...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root=(Get-Location).Path;" ^
  "$pythonw=Join-Path $root '.venv\Scripts\pythonw.exe';" ^
  "$stdout=Join-Path $root '.runs\streamlit.out.log';" ^
  "$stderr=Join-Path $root '.runs\streamlit.err.log';" ^
  "$token=[Environment]::GetEnvironmentVariable('HF_TOKEN','Process');" ^
  "if (-not $token) {$token=[Environment]::GetEnvironmentVariable('HF_TOKEN','User')};" ^
  "if ($token) {$env:HF_TOKEN=$token};" ^
  "$listeners=Get-NetTCPConnection -LocalPort 8504 -State Listen -ErrorAction SilentlyContinue;" ^
  "foreach ($listener in $listeners) {" ^
  "  $existing=Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $listener.OwningProcess);" ^
  "  if ($existing.CommandLine -notlike '*streamlit_app.py*') {Write-Error 'Port 8504 is used by another application.'; exit 1};" ^
  "  Stop-Process -Id $listener.OwningProcess -Force;" ^
  "  Wait-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue" ^
  "};" ^
  "$arguments='-m streamlit run streamlit_app.py --server.headless=true --server.port=8504';" ^
  "$process=Start-Process -FilePath $pythonw -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru;" ^
  "$deadline=(Get-Date).AddSeconds(90);" ^
  "do {" ^
  "  Start-Sleep -Milliseconds 500;" ^
  "  if ($process.HasExited) {Write-Error ('Streamlit stopped during startup. Read ' + $stderr); exit 1};" ^
  "  try {$response=Invoke-WebRequest 'http://localhost:8504/_stcore/health' -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -eq 200) {Start-Process 'http://localhost:8504'; exit 0}} catch {}" ^
  "} while ((Get-Date) -lt $deadline);" ^
  "Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue;" ^
  "Write-Error ('Streamlit did not become ready within 90 seconds. Read ' + $stderr); exit 1"

if errorlevel 1 goto :launch_failed
exit /b 0

:wrong_folder
echo Required project files were not found beside this launcher.
goto :failed

:uv_failed
echo uv could not be installed. Check the internet connection and PowerShell policy.
goto :failed

:sync_failed
echo Python or project dependencies could not be prepared.
goto :failed

:unsloth_sync_failed
echo The native Windows Unsloth runtime could not be prepared.
goto :failed

:launch_failed
echo Streamlit could not start. Review .runs\streamlit.err.log.
goto :failed

:failed
echo.
echo Launch failed. The window will remain open so you can read the error.
pause
exit /b 1
