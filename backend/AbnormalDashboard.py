from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import io
import json

import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel


# =========================================================
# 프로젝트 경로 설정
# backend/AbnormalDashboard.py 기준
# ROOT_DIR    = C:\smart_care_ai_web
# BACKEND_DIR = C:\smart_care_ai_web\backend
# MODEL_DIR   = C:\smart_care_ai_web\models
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"


# =========================================================
# 이상행동 모델 파일 경로
# pkl 파일은 전부 C:\smart_care_ai_web\models 안에 둠
# =========================================================

ELDERCARE_MODEL_PATH = MODEL_DIR / "eldercare_xgb_model.pkl"
ELDERCARE_LABEL_ENCODER_PATH = MODEL_DIR / "eldercare_label_encoder.pkl"
ELDERCARE_REASON_MODEL_PATH = MODEL_DIR / "eldercare_reason_model.pkl"
ELDERCARE_REASON_ENCODER_PATH = MODEL_DIR / "eldercare_reason_encoder.pkl"
ELDERCARE_FEATURES_PATH = MODEL_DIR / "eldercare_features.pkl"


# =========================================================
# 이상행동 기록 저장 파일
# MongoDB 없이 바로 동작하게 JSON 파일로 저장
# =========================================================

ABNORMAL_HISTORY_PATH = BACKEND_DIR / "abnormal_events_store.json"


# =========================================================
# APIRouter
# FallDashboard.py에서 include_router로 연결함
#
# 최종 주소:
# GET  /abnormal/health
# GET  /abnormal/features
# POST /abnormal/predict
# POST /abnormal/simulation/upload
# POST /abnormal/simulation/next
# GET  /abnormal/history
# GET  /abnormal/alerts
# GET  /abnormal/dashboard
# =========================================================

router = APIRouter(prefix="/abnormal", tags=["Abnormal Behavior"])


# =========================================================
# 전역 모델 객체
# =========================================================

model = None
label_encoder = None
reason_model = None
reason_encoder = None
features = None


# =========================================================
# 시뮬레이션 상태
# =========================================================

simulation_df = None
simulation_index = 0
simulation_filename = None


# =========================================================
# 요청 스키마
# 프론트에서는 아래 형태로 보냄:
# {
#   "sensor": {
#     "feature1": 0,
#     "feature2": 0
#   },
#   "source": "web"
# }
# =========================================================

class AbnormalPredictRequest(BaseModel):
    sensor: Dict[str, Any]
    source: Optional[str] = "web"
    actual_state: Optional[str] = None
    actual_reason: Optional[str] = None


# =========================================================
# 모델 로딩
# =========================================================

def load_eldercare_models():
    global model
    global label_encoder
    global reason_model
    global reason_encoder
    global features

    if model is not None:
        return

    required_files = [
        ELDERCARE_MODEL_PATH,
        ELDERCARE_LABEL_ENCODER_PATH,
        ELDERCARE_REASON_MODEL_PATH,
        ELDERCARE_REASON_ENCODER_PATH,
        ELDERCARE_FEATURES_PATH,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "이상행동 모델 파일이 없습니다.\n"
            "아래 파일들을 C:\\smart_care_ai_web\\models 폴더에 넣어주세요.\n\n"
            + "\n".join(missing_files)
        )

    model = joblib.load(ELDERCARE_MODEL_PATH)
    label_encoder = joblib.load(ELDERCARE_LABEL_ENCODER_PATH)
    reason_model = joblib.load(ELDERCARE_REASON_MODEL_PATH)
    reason_encoder = joblib.load(ELDERCARE_REASON_ENCODER_PATH)
    features = joblib.load(ELDERCARE_FEATURES_PATH)

    features = list(features)

    print("[ABNORMAL MODEL] 이상행동 모델 로드 완료")
    print(f"[ABNORMAL MODEL] feature 개수: {len(features)}")
    print(f"[ABNORMAL MODEL] 모델 경로: {ELDERCARE_MODEL_PATH}")


# =========================================================
# 위험 점수 / 보호자 알림
# =========================================================

def risk_score_by_state(state):
    if state == "위험":
        return 92
    if state == "주의":
        return 65
    if state == "외출":
        return 38
    if state == "식사":
        return 30
    if state == "수면":
        return 22
    return 18


def load_abnormal_history():
    if not ABNORMAL_HISTORY_PATH.exists():
        return []

    try:
        with open(ABNORMAL_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_abnormal_history(history):
    with open(ABNORMAL_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_recent_warning_count(limit=2):
    """
    최근 기록 중 연속된 '주의' 개수 확인.
    현재 예측이 '주의'일 때, 최근 2개도 '주의'면 총 3회 연속으로 판단.
    """
    history = load_abnormal_history()
    recent = list(reversed(history))[:limit]

    count = 0

    for item in recent:
        if item.get("state") == "주의":
            count += 1
        else:
            break

    return count


def guardian_alert_by_state(state):
    if state == "위험":
        return {
            "guardian_alert": True,
            "guardian_status": "보호자 알림 발송",
            "guardian_message": "위험 상황으로 판단되어 보호자에게 즉시 알림이 발송됩니다.",
        }

    if state == "주의":
        recent_warning_count = get_recent_warning_count(limit=2)

        if recent_warning_count >= 2:
            return {
                "guardian_alert": True,
                "guardian_status": "보호자 확인 필요",
                "guardian_message": "주의 상태가 3회 연속 감지되어 보호자 확인이 필요합니다.",
            }

        return {
            "guardian_alert": False,
            "guardian_status": "주의 관찰 중",
            "guardian_message": "주의 상태가 감지되었습니다. 3회 연속 감지 시 보호자 알림이 발송됩니다.",
        }

    return {
        "guardian_alert": False,
        "guardian_status": "알림 없음",
        "guardian_message": "현재 즉시 보호자 알림이 필요한 상태는 아닙니다.",
    }


# =========================================================
# 입력값 전처리
# =========================================================

def build_input_dataframe(input_dict: dict):
    load_eldercare_models()

    input_df = pd.DataFrame([input_dict])

    # 모델이 원하는 feature가 없으면 0으로 채움
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # 숫자 변환
    for col in features:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_df = input_df.fillna(0)

    # 학습 당시 feature 순서로 정렬
    input_df = input_df[features]

    return input_df


# =========================================================
# 예측 실행
# =========================================================

def run_prediction(
    input_dict,
    source="web",
    actual_state=None,
    actual_reason=None,
):
    load_eldercare_models()

    input_df = build_input_dataframe(input_dict)

    # 상태 예측
    pred_encoded = model.predict(input_df)
    state = str(label_encoder.inverse_transform(pred_encoded)[0])

    # 사유 예측
    reason_encoded = reason_model.predict(input_df)
    reason = str(reason_encoder.inverse_transform(reason_encoded)[0])

    # 위험 점수
    risk_score = risk_score_by_state(state)

    # 보호자 알림 여부
    guardian = guardian_alert_by_state(state)

    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "state": state,
        "risk_level": state,
        "risk_score": risk_score,
        "reason": reason,
        "actual_state": actual_state,
        "actual_reason": actual_reason,
        **guardian,
        "sensor": input_dict,
    }

    history = load_abnormal_history()
    history.append(result)

    # 최근 300개만 저장
    history = history[-300:]

    save_abnormal_history(history)

    return result


# =========================================================
# 기본 상태 API
# =========================================================

@router.get(
    "/",
    summary="이상행동 API 기본 상태 확인",
    description="이상행동 관제 API가 실행 중인지 확인합니다.",
)
def root():
    return {
        "message": "이상행동 관제 API 실행중",
        "service": "독거노인 돌봄 상태 예측 API",
        "model_dir": str(MODEL_DIR),
        "history_path": str(ABNORMAL_HISTORY_PATH),
    }


@router.get(
    "/health",
    summary="이상행동 서버 및 모델 상태 확인",
    description="AI 모델 파일 로드 상태와 feature 정보를 확인합니다.",
)
def health():
    try:
        load_eldercare_models()

        return {
            "status": "ok",
            "api": "abnormal-running",
            "model": "loaded",
            "feature_count": len(features),
            "simulation_file": simulation_filename,
            "simulation_index": simulation_index,
            "model_path": str(ELDERCARE_MODEL_PATH),
            "features_path": str(ELDERCARE_FEATURES_PATH),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        return {
            "status": "error",
            "api": "abnormal-running",
            "model": "not-loaded",
            "message": str(e),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


@router.get(
    "/features",
    summary="이상행동 모델 입력 feature 조회",
    description="이상행동 모델이 필요로 하는 입력 컬럼 목록을 반환합니다.",
)
def get_features():
    load_eldercare_models()

    return {
        "feature_count": len(features),
        "features": features,
    }


# =========================================================
# 예측 API
# =========================================================

@router.post(
    "/predict",
    summary="돌봄 상태 예측",
    description="""
센서 데이터를 입력받아 독거노인의 현재 돌봄 상태를 예측합니다.

예측 상태:
- 수면
- 식사
- 외출
- 주의
- 위험
- 기타

예측 후 기록 파일에 저장되며,
위험 또는 주의 상태일 경우 보호자 알림 여부도 함께 반환합니다.
""",
)
def predict(request: AbnormalPredictRequest):
    try:
        return run_prediction(
            input_dict=request.sensor,
            source=request.source,
            actual_state=request.actual_state,
            actual_reason=request.actual_reason,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이상행동 예측 실패: {e}")


# =========================================================
# 시뮬레이션 파일 업로드
# =========================================================

@router.post(
    "/simulation/upload",
    summary="시뮬레이션 파일 업로드",
    description="""
CSV 또는 Excel 파일을 업로드합니다.

업로드된 파일은 실제 센서 스트림처럼 한 줄씩 재생됩니다.
파일에는 모델 입력에 필요한 센서 컬럼들이 포함되어야 합니다.

지원 형식:
- .csv
- .xlsx
- .xls
""",
)
async def upload_simulation_file(file: UploadFile = File(...)):
    global simulation_df
    global simulation_index
    global simulation_filename

    load_eldercare_models()

    content = await file.read()
    filename = file.filename or ""

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="CSV 또는 Excel 파일만 업로드 가능합니다.",
            )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 읽기 실패: {e}")

    missing = [col for col in features if col not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "필수 센서 컬럼이 없습니다.",
                "missing_columns": missing,
            },
        )

    simulation_df = df.copy()
    simulation_index = 0
    simulation_filename = filename

    return {
        "message": "시뮬레이션 파일 업로드 완료",
        "filename": filename,
        "rows": len(simulation_df),
        "columns": list(simulation_df.columns),
        "start_index": simulation_index,
    }


@router.get(
    "/simulation/status",
    summary="시뮬레이션 상태 조회",
    description="현재 업로드된 시뮬레이션 파일의 상태를 조회합니다.",
)
def simulation_status():
    if simulation_df is None:
        return {
            "loaded": False,
            "message": "업로드된 시뮬레이션 파일이 없습니다.",
        }

    return {
        "loaded": True,
        "filename": simulation_filename,
        "rows": len(simulation_df),
        "current_index": simulation_index,
        "remaining": max(len(simulation_df) - simulation_index, 0),
    }


@router.post(
    "/simulation/reset",
    summary="시뮬레이션 처음부터 다시 시작",
    description="업로드된 시뮬레이션 파일의 재생 위치를 0번 행으로 초기화합니다.",
)
def simulation_reset():
    global simulation_index

    if simulation_df is None:
        raise HTTPException(
            status_code=400,
            detail="업로드된 시뮬레이션 파일이 없습니다.",
        )

    simulation_index = 0

    return {
        "message": "시뮬레이션 인덱스 초기화 완료",
        "current_index": simulation_index,
    }


@router.post(
    "/simulation/next",
    summary="다음 센서 데이터 실행",
    description="""
업로드된 CSV/Excel 파일에서 다음 행을 읽어 AI 예측을 수행합니다.

동작 과정:
1. 현재 행의 센서값을 읽음
2. AI 모델에 입력
3. 돌봄 상태 예측
4. 위험지수 계산
5. 보호자 알림 여부 판단
6. 기록 저장
7. 다음 행으로 이동
""",
)
def simulation_next():
    global simulation_index

    load_eldercare_models()

    if simulation_df is None:
        raise HTTPException(
            status_code=400,
            detail="업로드된 시뮬레이션 파일이 없습니다.",
        )

    if simulation_index >= len(simulation_df):
        raise HTTPException(
            status_code=400,
            detail="시뮬레이션 데이터가 끝났습니다. /abnormal/simulation/reset을 실행하세요.",
        )

    row = simulation_df.iloc[simulation_index]

    input_dict = {}

    for col in features:
        value = row[col]

        if pd.isna(value):
            value = 0

        input_dict[col] = float(value)

    actual_state = None
    actual_reason = None

    if "Estimation" in simulation_df.columns:
        actual_state = str(row["Estimation"])

    if "Reason" in simulation_df.columns:
        actual_reason = str(row["Reason"])

    current_index = simulation_index
    simulation_index += 1

    result = run_prediction(
        input_dict=input_dict,
        source="simulation_file",
        actual_state=actual_state,
        actual_reason=actual_reason,
    )

    result["simulation"] = {
        "filename": simulation_filename,
        "row_index": current_index,
        "next_index": simulation_index,
        "total_rows": len(simulation_df),
    }

    return result


# =========================================================
# 기록 조회 API
# =========================================================

@router.get(
    "/history",
    summary="돌봄 상태 예측 이력 조회",
    description="최근 이상행동 예측 이력을 조회합니다.",
)
def get_history(limit: int = 20):
    history = load_abnormal_history()
    history = list(reversed(history))

    limit = max(1, min(limit, 100))

    return {
        "count": min(limit, len(history)),
        "items": history[:limit],
    }


@router.delete(
    "/history",
    summary="이상행동 이력 삭제",
    description="저장된 이상행동 예측 이력을 모두 삭제합니다.",
)
def delete_history():
    save_abnormal_history([])

    return {
        "deleted": True,
        "message": "이상행동 기록을 삭제했습니다.",
    }


@router.get(
    "/alerts",
    summary="위험 및 보호자 알림 조회",
    description="""
실제로 보호자 알림이 발생한 이력만 조회합니다.

표시 대상:
- 위험 상태
- 주의 상태 3회 연속 발생
""",
)
def get_alerts(limit: int = 20):
    history = load_abnormal_history()

    alerts = [
        item for item in history
        if item.get("guardian_alert") is True
    ]

    alerts = list(reversed(alerts))

    limit = max(1, min(limit, 100))

    return {
        "count": min(limit, len(alerts)),
        "items": alerts[:limit],
    }


@router.get(
    "/dashboard",
    summary="대시보드 통합 데이터 조회",
    description="최신 예측 결과와 최근 이력을 한 번에 조회합니다.",
)
def dashboard():
    history = load_abnormal_history()

    latest = history[-1] if history else None
    recent = list(reversed(history))[:10]

    return {
        "latest": latest,
        "recent": recent,
    }


@router.get(
    "/stats",
    summary="이상행동 통계 조회",
    description="이상행동 대시보드 카드에 표시할 통계를 조회합니다.",
)
def stats():
    history = load_abnormal_history()

    total = len(history)
    danger_count = len([x for x in history if x.get("state") == "위험"])
    warning_count = len([x for x in history if x.get("state") == "주의"])
    outing_count = len([x for x in history if x.get("state") == "외출"])
    meal_count = len([x for x in history if x.get("state") == "식사"])
    sleep_count = len([x for x in history if x.get("state") == "수면"])
    guardian_alert_count = len([x for x in history if x.get("guardian_alert") is True])

    latest = history[-1] if history else None

    return {
        "total": total,
        "danger_count": danger_count,
        "warning_count": warning_count,
        "outing_count": outing_count,
        "meal_count": meal_count,
        "sleep_count": sleep_count,
        "guardian_alert_count": guardian_alert_count,
        "latest": latest,
    }