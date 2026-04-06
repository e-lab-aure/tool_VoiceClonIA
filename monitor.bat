@echo off
:: ============================================================
:: VoiceClonIA — Monitor
:: Lance le tableau de bord console de surveillance.
:: ============================================================

setlocal

if not exist "%~dp0venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel introuvable.
    echo Lancez setup.bat d'abord.
    pause
    exit /b 1
)

call "%~dp0venv\Scripts\activate.bat"
python "%~dp0monitor.py"

endlocal
