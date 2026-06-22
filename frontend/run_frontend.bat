@echo off
echo =========================================
echo Smart Care AI Frontend Server
echo React / Vite / port 5173
echo =========================================

cd /d %~dp0

if not exist package.json (
    echo [ERROR] package.json 파일을 찾을 수 없습니다.
    echo 이 bat 파일은 frontend 폴더 안에 있어야 합니다.
    echo 현재 위치:
    cd
    pause
    exit /b
)

if not exist node_modules (
    echo [INFO] node_modules가 없습니다.
    echo npm install을 먼저 실행합니다.
    echo.
    npm install
)

echo.
echo =========================================
echo [INFO] Frontend URL
echo 내 컴퓨터 접속:
echo http://localhost:5173
echo.
echo 팀원 접속:
echo http://192.168.0.24:5173
echo.
echo [INFO] Backend API URL
echo http://192.168.0.24:8000
echo =========================================
echo.

echo [INFO] Vite 서버를 5173 포트로 실행합니다.
echo.

npm run dev -- --host 0.0.0.0 --port 5173 --strictPort

pause