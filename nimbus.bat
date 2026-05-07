@echo off
setlocal

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=status"

if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\nimbusctl.ps1" status
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\nimbusctl.ps1" %*
)
