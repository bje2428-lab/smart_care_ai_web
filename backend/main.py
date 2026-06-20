from pathlib import Path
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from FallDashboard import (
    router as fall_router,
    startup_fall_dashboard,
    shutdown_fall_dashboard,
)
from AbnormalDashboard import router as abnormal_router


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"


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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# =========================================================

@app.get("/")
def root():
    return {
        "service": "Smart Care AI Fall & Abnormal Detection API",
        "status": "running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "model_dir": str(MODEL_DIR),
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