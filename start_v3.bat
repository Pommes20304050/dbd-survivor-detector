@echo off
title DBD Survivor Detector — v3 Live System
cd /d "%~dp0"

echo.
echo ================================================================
echo   DBD SURVIVOR DETECTOR — v3 Live Overlay System
echo ================================================================
echo.
echo   - Transparentes Overlay auf DBD
echo   - Dashboard: http://localhost:8765
echo   - Modell: models/best.pt
echo.
echo ================================================================
echo.

python overlay\overlay_server.py

echo.
echo System beendet.
pause
