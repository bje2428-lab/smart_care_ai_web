from pathlib import Path
from datetime import datetime
import sys
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# =========================================================
# 경로 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"


for path in [ROOT_DIR, BACKEND_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =========================================================
# Router import
# =========================================================

from FallDashboard import (
    router as fall_router,
    startup_fall_dashboard,
    shutdown_fall_dashboard,
    health as fall_health_status,
)
from AbnormalDashboard import router as abnormal_router


# =========================================================
# FastAPI App 생성
# =========================================================

app = FastAPI(
    title="Smart Care AI - Fall & Abnormal Detection API",
    description="""
스마트 돌봄을 위한 독거노인 안전 관리 AI 협업 관제 플랫폼 API입니다.

기능:
- mmWave CSV 기반 낙상 감지
- 이상행동 / 돌봄 상태 예측
- 위험도 계산
- 보호자 알림 판단
- 낙상 알림 MongoDB 저장
- 이상행동 기록 JSON 저장
""",
    version="1.0.0",
)


# =========================================================
# CORS 설정
# React / Flutter Web / 팀원 PC 접속 허용
# =========================================================
# 주의:
# allow_credentials=True일 때 allow_origins=["*"]를 쓰면
# 브라우저에서 CORS 오류가 날 수 있음.
# 그래서 실제 사용할 주소를 직접 적고,
# 192.168.x.x 대역은 allow_origin_regex로 허용함.
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # 내 PC React / Flutter Web 로컬
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",

        # 내 PC 프론트 주소
        "http://192.168.0.24",
        "http://192.168.0.24:5173",
        "http://192.168.0.24:3000",
        "http://192.168.0.24:5000",

        # 팀원 PC 주소
        "http://192.168.0.58",
        "http://192.168.0.58:5173",
        "http://192.168.0.58:3000",
        "http://192.168.0.58:5000",
    ],
    # Flutter Web은 실행할 때 포트가 랜덤으로 잡힐 수 있어서
    # localhost, 127.0.0.1, 192.168.x.x의 모든 포트를 허용
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Router 연결
# =========================================================

app.include_router(fall_router)
app.include_router(abnormal_router)


# =========================================================
# 전체 에러 확인용 핸들러
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("=========================================")
    print("[SERVER ERROR]")
    print(f"URL: {request.url}")
    print(f"METHOD: {request.method}")
    print("-----------------------------------------")
    traceback.print_exc()
    print("=========================================")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "서버 내부 오류가 발생했습니다.",
            "path": str(request.url.path),
            "error": str(exc),
        },
    )


# =========================================================
# 서버 시작 / 종료
# =========================================================

@app.on_event("startup")
def startup_event():
    print("=========================================")
    print("Smart Care AI API Server Started")
    print(f"ROOT_DIR    : {ROOT_DIR}")
    print(f"BACKEND_DIR : {BACKEND_DIR}")
    print(f"MODEL_DIR   : {MODEL_DIR}")
    print("=========================================")

    startup_fall_dashboard()

    print("[ROUTER] FallDashboard 연결 완료")
    print("[ROUTER] AbnormalDashboard 연결 완료")


@app.on_event("shutdown")
def shutdown_event():
    shutdown_fall_dashboard()
    print("[SERVER] 종료 완료")


# =========================================================
# 공통 기본 API
# 프론트 서버 상태 카드에서도 이 응답을 사용할 수 있게
# 모델 / MongoDB 상태를 같이 내려줌
# =========================================================

@app.get("/")
def root():
    fall_status = fall_health_status()

    return {
        "service": "Smart Care AI Fall & Abnormal Detection API",
        "status": "running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "model_dir": str(MODEL_DIR),

        # 프론트 서버 상태 카드용
        "api": "fall-running",
        "model_exists": fall_status.get("model_exists", False),
        "meta_exists": fall_status.get("meta_exists", False),
        "mongo_connected": fall_status.get("mongo_connected", False),
        "model_path": fall_status.get("model_path"),
        "meta_path": fall_status.get("meta_path"),
        "rf_dir_exists": fall_status.get("rf_dir_exists", False),

        "apis": {
            "fall": {
                "health": "/health",
                "predict": "/predict",
                "events": "/events",
                "stats": "/stats",
            },
            "abnormal": {
                "health": "/abnormal/health",
                "features": "/abnormal/features",
                "predict": "/abnormal/predict",
                "history": "/abnormal/history",
                "stats": "/abnormal/stats",
                "alerts": "/abnormal/alerts",
                "dashboard": "/abnormal/dashboard",
            },
        },
    }


@app.get("/api-info")
def api_info():
    fall_status = fall_health_status()

    return {
        "service": "Smart Care AI API",
        "version": "1.0.0",
        "backend_status": "running",
        "model_exists": fall_status.get("model_exists", False),
        "mongo_connected": fall_status.get("mongo_connected", False),
        "frontend_origin_examples": [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://192.168.0.24:5173",
            "http://192.168.0.58:5173",
        ],
        "backend_url_examples": [
            "http://localhost:8000",
            "http://192.168.0.24:8000",
        ],
        "team_notice": "팀원 PC에서는 localhost:8000이 아니라 백엔드가 켜진 PC의 IP 주소를 사용해야 합니다.",
    }