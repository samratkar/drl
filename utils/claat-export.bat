@echo off
setlocal

if "%~1"=="" (
  >&2 echo Usage: %~nx0 ^<markdown-file^>
  exit /b 1
)

set "SRC=%~f1"
if not exist "%SRC%" (
  >&2 echo Error: file not found: %SRC%
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0claat-export.ps1" "%SRC%"

set "EXITCODE=%ERRORLEVEL%"
exit /b %EXITCODE%
