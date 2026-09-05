@echo off
REM ========================================
REM  NOTICE: This functionality has been
REM  integrated into install-audiobook.bat
REM  This standalone script is kept for
REM  emergency pydantic updates only.
REM ========================================
echo ========================================
echo  Chatterbox TTS - Pydantic Updater
echo ========================================
echo.
echo This script will update the pydantic library in your
echo existing Chatterbox TTS virtual environment to version 2.10.6
echo to resolve potential compatibility issues.
echo.
echo NOTE: This functionality is now integrated into install-audiobook.bat
echo This standalone script is for emergency updates only.
echo.

echo Checking for virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure this script is in the chatterbox repository root
    echo and that you have run install-audiobook.bat at least once.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Uninstalling existing pydantic (if any)...
pip uninstall pydantic -y

echo.
echo Installing pydantic version 2.10.6...
pip install pydantic==2.10.6

echo.
echo Verifying pydantic version...
pip show pydantic | findstr /C:"Version: 2.10.6"
if %errorlevel% neq 0 (
    echo ERROR: Pydantic 2.10.6 installation failed or was not confirmed.
    echo Please check the output above for errors.
    echo You may need to run install-audiobook.bat again.
) else (
    echo INFO: Pydantic successfully updated to version 2.10.6.
)

echo.
echo Deactivating virtual environment...
deactivate

echo.
echo ========================================
echo         Update Process Complete
echo ========================================
echo.
echo You can now try running launch_audiobook.bat again.
pause 