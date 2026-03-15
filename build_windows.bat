@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv-win" (
  py -3.10 -m venv .venv-win
)

call ".venv-win\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements-build.txt

pyinstaller --noconfirm --onefile --windowed --name FAST-Calib-GUI-windows gui_app.py

echo.
echo Build finished.
echo Executable:
echo %CD%\dist\FAST-Calib-GUI-windows.exe

endlocal
