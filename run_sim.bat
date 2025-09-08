@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "OMNI_KIT_ACCEPT_EULA=1"
set "OMNI_KIT_FAST_SHUTDOWN=1"
set "ISAACLAB_APPS_DIR=D:\My_Project\Sim\IsaacLab\apps"

if defined ISAACSIM_PYTHON_EXE (set "PYEXE=%ISAACSIM_PYTHON_EXE%") else (set "PYEXE=python")

if "%~1"=="" (
  rem switched to --num_envs
  set "ARGS=--robot go2 --num_envs 1 --app python"
) else (
  set "ARGS=%*"
)

echo [INFO] Python: %PYEXE%
echo [INFO] Args: %ARGS%
%PYEXE% main.py %ARGS%
exit /b %ERRORLEVEL%