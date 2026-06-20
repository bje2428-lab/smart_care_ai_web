@echo off
cd /d "%~dp0"

echo =========================================
echo Smart Care AI API Server
echo main.py / port 8000
echo =========================================

venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause