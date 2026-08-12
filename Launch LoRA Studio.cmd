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

echo Preparing Python 3.12 and project dependencies...
"%UV_EXE%" sync --locked --no-dev
if errorlevel 1 goto :sync_failed

if not exist ".venv\Scripts\pythonw.exe" goto :sync_failed
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
  "$arguments='-m streamlit run streamlit_app.py --server.headless=true --server.port=8501';" ^
  "$process=Start-Process -FilePath $pythonw -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru;" ^
  "$deadline=(Get-Date).AddSeconds(90);" ^
  "do {" ^
  "  Start-Sleep -Milliseconds 500;" ^
  "  if ($process.HasExited) {Write-Error ('Streamlit stopped during startup. Read ' + $stderr); exit 1};" ^
  "  try {$response=Invoke-WebRequest 'http://localhost:8501/_stcore/health' -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -eq 200) {Start-Process 'http://localhost:8501'; exit 0}} catch {}" ^
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

:launch_failed
echo Streamlit could not start. Review .runs\streamlit.err.log.
goto :failed

:failed
echo.
echo Launch failed. The window will remain open so you can read the error.
pause
exit /b 1
