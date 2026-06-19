@echo off
echo Smart Care AI Backend Start

cd /d %~dp0

py -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause