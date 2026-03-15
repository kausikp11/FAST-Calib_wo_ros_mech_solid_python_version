@echo off
setlocal

cd /d "%~dp0"

if exist "dist\FAST-Calib-GUI-windows.exe" (
  start "" "dist\FAST-Calib-GUI-windows.exe"
  exit /b 0
)

echo Packaged Windows GUI not found: dist\FAST-Calib-GUI-windows.exe
exit /b 1
