from pathlib import Path
from datetime import datetime
import os
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
RF_DIR = ROOT_DIR / "rf"
MODEL_DIR = ROOT_DIR / "models"

for path in [ROOT_DIR, BACKEND_DIR, RF_DIR, MODEL_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =========================================================
# 환경변수
# =========================================================

def parse_cors_origins():
    raw = os.getenv("CORS_ORIGINS", "*").strip()

    if raw == "*":
        return ["*"]

    return [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]


CORS_ORIGINS = parse_cors_origins()


# =========================================================
# FastAPI App 생성
# =========================================================

app = FastAPI(
    title=os.getenv("APP_TITLE", "Smart Care AI API"),
    description=os.getenv(
        "APP_DESCRIPTION",
        "스마트 돌봄을 위한 독거노인 안전 관리 AI 협업 관제 플랫폼 API",
    ),
    version=os.getenv("APP_VERSION", "1.0.0"),
)


# =========================================================
# CORS 설정
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 기본 Health API
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "api": "running",
        "message": "Smart Care AI API is running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "rf_dir": str(RF_DIR),
        "model_dir": str(MODEL_DIR),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": "running",
        "message": "FastAPI server is connected",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "rf_dir": str(RF_DIR),
        "model_dir": str(MODEL_DIR),
        "rf_dir_exists": RF_DIR.exists(),
        "model_dir_exists": MODEL_DIR.exists(),
        "cors_origins": CORS_ORIGINS,
    }


# =========================================================
# Router import
# =========================================================

fall_router = None
startup_fall_dashboard = None
shutdown_fall_dashboard = None
fall_import_error = None

abnormal_router = None
abnormal_import_error = None

vital_router = None
startup_vital_signal = None
shutdown_vital_signal = None
vital_import_error = None

integrated_router = None
integrated_import_error = None


try:
    from FallDashboard import (
        router as fall_router,
        startup_fall_dashboard,
        shutdown_fall_dashboard,
    )

    print("[ROUTER] FallDashboard import 성공")

except Exception as e:
    fall_import_error = str(e)
    print("[ROUTER] FallDashboard import 실패")
    traceback.print_exc()


try:
    from AbnormalDashboard import router as abnormal_router

    print("[ROUTER] AbnormalDashboard import 성공")

except Exception as e:
    abnormal_import_error = str(e)
    print("[ROUTER] AbnormalDashboard import 실패 - 이상행동 API 비활성화")
    traceback.print_exc()


try:
    from vital_signal import (
        router as vital_router,
        startup_vital_signal,
        shutdown_vital_signal,
    )

    print("[ROUTER] VitalSignal import 성공")

except Exception as e:
    vital_import_error = str(e)
    print("[ROUTER] VitalSignal import 실패 - 생체신호 API 비활성화")
    traceback.print_exc()


try:
    from IntegratedDashboard import router as integrated_router

    print("[ROUTER] IntegratedDashboard import 성공")

except Exception as e:
    integrated_import_error = str(e)
    print("[ROUTER] IntegratedDashboard import 실패 - 통합 관제 API 비활성화")
    traceback.print_exc()


# =========================================================
# Router 연결
# =========================================================

if fall_router is not None:
    app.include_router(fall_router)
    print("[ROUTER] FallDashboard 연결 완료")
else:
    print("[ROUTER] FallDashboard 연결 실패")

if abnormal_router is not None:
    app.include_router(abnormal_router)
    print("[ROUTER] AbnormalDashboard 연결 완료")
else:
    print("[ROUTER] AbnormalDashboard 연결 안 함")

if vital_router is not None:
    app.include_router(vital_router)
    print("[ROUTER] VitalSignal 연결 완료")
else:
    print("[ROUTER] VitalSignal 연결 안 함")

if integrated_router is not None:
    app.include_router(integrated_router)
    print("[ROUTER] IntegratedDashboard 연결 완료")
else:
    print("[ROUTER] IntegratedDashboard 연결 안 함")


# =========================================================
# API 정보 확인
# =========================================================

@app.get("/api-info")
def api_info():
    return {
        "service": "Smart Care AI API",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "backend_status": "running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "paths": {
            "root": "/",
            "health": "/health",

            # 낙상 API
            "fall_health": "/fall/health",
            "fall_predict": "/predict",
            "fall_events": "/events",
            "fall_stats": "/stats",

            # 이상행동 API
            "abnormal_health": "/abnormal/health",
            "abnormal_events": "/abnormal/events",

            # 바이탈 API
            "vital_health": "/vital/health",
            "vital_features": "/vital/features",
            "vital_predict": "/vital/predict",
            "vital_predict_file": "/vital/predict-file",
            "vital_history": "/vital/history",
            "vital_stats": "/vital/stats",
            "vital_reset": "/vital/reset",

            # 통합 관제 API
            "integrated_health": "/integrated/health",
            "integrated_upload": "/integrated/simulation/upload",
            "integrated_status": "/integrated/simulation/{session_id}/status",
            "integrated_next": "/integrated/simulation/{session_id}/next",
            "integrated_reset": "/integrated/simulation/{session_id}/reset",
        },

        "routers": {
            "fall": {
                "loaded": fall_router is not None,
                "import_error": fall_import_error,
            },
            "abnormal": {
                "loaded": abnormal_router is not None,
                "import_error": abnormal_import_error,
            },
            "vital_signal": {
                "loaded": vital_router is not None,
                "import_error": vital_import_error,
            },
            "integrated": {
                "loaded": integrated_router is not None,
                "import_error": integrated_import_error,
            },
        },

        "directories": {
            "root_dir": str(ROOT_DIR),
            "backend_dir": str(BACKEND_DIR),
            "rf_dir": str(RF_DIR),
            "model_dir": str(MODEL_DIR),
            "rf_dir_exists": RF_DIR.exists(),
            "model_dir_exists": MODEL_DIR.exists(),
        },

        "cors_origins": CORS_ORIGINS,
    }


# =========================================================
# 서버 시작 / 종료
# =========================================================

@app.on_event("startup")
def startup_event():
    print("=========================================")
    print("Smart Care AI API Server Started")
    print(f"ROOT_DIR    : {ROOT_DIR}")
    print(f"BACKEND_DIR : {BACKEND_DIR}")
    print(f"RF_DIR      : {RF_DIR}")
    print(f"MODEL_DIR   : {MODEL_DIR}")
    print(f"CORS        : {CORS_ORIGINS}")
    print("=========================================")

    if startup_fall_dashboard is not None:
        try:
            startup_fall_dashboard()
            print("[STARTUP] FallDashboard 시작 완료")
        except Exception:
            print("[STARTUP] FallDashboard 시작 오류")
            traceback.print_exc()

    if startup_vital_signal is not None:
        try:
            startup_vital_signal()
            print("[STARTUP] VitalSignal 시작 완료")
        except Exception:
            print("[STARTUP] VitalSignal 시작 오류")
            traceback.print_exc()

    print("[SERVER] startup 완료")


@app.on_event("shutdown")
def shutdown_event():
    if shutdown_fall_dashboard is not None:
        try:
            shutdown_fall_dashboard()
            print("[SHUTDOWN] FallDashboard 종료 완료")
        except Exception:
            print("[SHUTDOWN] FallDashboard 종료 오류")
            traceback.print_exc()

    if shutdown_vital_signal is not None:
        try:
            shutdown_vital_signal()
            print("[SHUTDOWN] VitalSignal 종료 완료")
        except Exception:
            print("[SHUTDOWN] VitalSignal 종료 오류")
            traceback.print_exc()

    print("[SERVER] 종료 완료")


# =========================================================
# 전체 에러 핸들러
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
            "status": "Error",
            "message": "서버 내부 오류가 발생했습니다.",
            "path": str(request.url.path),
            "error": str(exc),
        },
    )