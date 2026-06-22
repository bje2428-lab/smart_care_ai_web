@echo off
echo =========================================
echo Smart Care AI API Server
echo main.py / port 8000
echo =========================================

cd /d %~dp0

if not exist venv\Scripts\activate (
    echo [ERROR] venv가 없습니다.
    echo 먼저 아래 명령어를 실행하세요:
    echo py -m venv venv
    pause
    exit /b
)

call venv\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause