@echo off
title DBD Survivor Detector — Stop All
echo.
echo ================================================================
echo   STOPPING ALL DBD SURVIVOR DETECTOR PROCESSES
echo ================================================================
echo.

REM Python-Prozesse beenden (Overlay, Training, Label-Tool, etc.)
echo [1/3] Beende alle Python-Prozesse...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM python3.12.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1

REM Falls der Flask-Server auf Port 8765 noch offen ist
echo [2/3] Beende Port 8765 Handler...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM OpenCV-Fenster (falls hängend)
echo [3/3] Beende OpenCV Fenster...
taskkill /F /IM yolo.exe /T >nul 2>&1

echo.
echo ================================================================
echo   ALLES BEENDET.
echo ================================================================
echo.

REM Status
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find "python.exe" >nul
if errorlevel 1 (
    echo   [OK] Keine Python-Prozesse mehr aktiv.
) else (
    echo   [!] Einige Python-Prozesse laufen noch. Manuell im Task-Manager beenden.
)

echo.
timeout /t 3 >nul
