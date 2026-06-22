from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import io
import json

import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"


ELDERCARE_MODEL_PATH = MODEL_DIR / "eldercare_xgb_model.pkl"
ELDERCARE_LABEL_ENCODER_PATH = MODEL_DIR / "eldercare_label_encoder.pkl"
ELDERCARE_REASON_MODEL_PATH = MODEL_DIR / "eldercare_reason_model.pkl"
ELDERCARE_REASON_ENCODER_PATH = MODEL_DIR / "eldercare_reason_encoder.pkl"
ELDERCARE_FEATURES_PATH = MODEL_DIR / "eldercare_features.pkl"


ABNORMAL_HISTORY_PATH = BACKEND_DIR / "abnormal_events_store.json"


DEFAULT_FEATURES = [
    "Temperature",
    "Humidity",
    "Illuminance",
    "Activity_IR",
    "CO2",
    "TVOC",
    "HeartRate",
    "BreathRate",
    "SPO2",
    "SkinTemperature",
    "SleepPhase",
    "SleepScore",
    "WalkingSteps",
    "StressIndex",
    "ActivityIntensity",
    "CaloricExpenditure",
    "Button",
    "Shout",
]


SAVE_HISTORY_STATES = {"위험", "주의"}


router = APIRouter(prefix="/abnormal", tags=["Abnormal Behavior"])


model = None
label_encoder = None
reason_model = None
reason_encoder = None
features = None


simulation_df = None
simulation_index = 0
simulation_filename = None


# 저장은 위험/주의만 하지만,
# 주의 3회 연속 판단은 정상/수면/식사/외출까지 포함한 실제 예측 흐름을 봐야 함
recent_prediction_states = []


class AbnormalPredictRequest(BaseModel):
    sensor: Dict[str, Any]
    source: Optional[str] = "web"
    actual_state: Optional[str] = None
    actual_reason: Optional[str] = None


def get_model_file_status():
    required_files = [
        ELDERCARE_MODEL_PATH,
        ELDERCARE_LABEL_ENCODER_PATH,
        ELDERCARE_REASON_MODEL_PATH,
        ELDERCARE_REASON_ENCODER_PATH,
        ELDERCARE_FEATURES_PATH,
    ]

    existing_files = [str(path) for path in required_files if path.exists()]
    missing_files = [str(path) for path in required_files if not path.exists()]

    return {
        "all_exists": len(missing_files) == 0,
        "existing_files": existing_files,
        "missing_files": missing_files,
    }


def load_feature_list():
    global features

    if features is not None:
        return features

    if ELDERCARE_FEATURES_PATH.exists():
        try:
            loaded = joblib.load(ELDERCARE_FEATURES_PATH)
            features = list(loaded)
            return features
        except Exception:
            pass

    features = list(DEFAULT_FEATURES)
    return features


def load_eldercare_models():
    global model
    global label_encoder
    global reason_model
    global reason_encoder
    global features

    if model is not None:
        return

    file_status = get_model_file_status()

    if not file_status["all_exists"]:
        raise FileNotFoundError(
            "이상행동 모델 파일이 없습니다.\n"
            "아래 파일들을 C:\\smart_care_ai_web\\models 폴더에 넣어주세요.\n\n"
            + "\n".join(file_status["missing_files"])
        )

    model = joblib.load(ELDERCARE_MODEL_PATH)
    label_encoder = joblib.load(ELDERCARE_LABEL_ENCODER_PATH)
    reason_model = joblib.load(ELDERCARE_REASON_MODEL_PATH)
    reason_encoder = joblib.load(ELDERCARE_REASON_ENCODER_PATH)
    features = list(joblib.load(ELDERCARE_FEATURES_PATH))

    print("[ABNORMAL MODEL] 이상행동 모델 로드 완료")
    print(f"[ABNORMAL MODEL] feature 개수: {len(features)}")
    print(f"[ABNORMAL MODEL] 모델 경로: {ELDERCARE_MODEL_PATH}")


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


def is_save_history_state(state: str) -> bool:
    return state in SAVE_HISTORY_STATES


def load_abnormal_history():
    if not ABNORMAL_HISTORY_PATH.exists():
        return []

    try:
        with open(ABNORMAL_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        # 예전 파일에 외출/식사/수면/정상 기록이 섞여 있어도 화면에는 위험/주의만 보여줌
        return [
            item for item in data
            if item.get("state") in SAVE_HISTORY_STATES
        ]

    except Exception:
        return []


def save_abnormal_history(history):
    filtered_history = [
        item for item in history
        if item.get("state") in SAVE_HISTORY_STATES
    ]

    with open(ABNORMAL_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_history, f, ensure_ascii=False, indent=2)


def get_recent_warning_count_from_memory(limit=2):
    count = 0

    for state in reversed(recent_prediction_states[-limit:]):
        if state == "주의":
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
        recent_warning_count = get_recent_warning_count_from_memory(limit=2)

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


def build_input_dataframe(input_dict: dict):
    load_eldercare_models()

    input_df = pd.DataFrame([input_dict])

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    for col in features:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_df = input_df.fillna(0)
    input_df = input_df[features]

    return input_df


def run_prediction(
    input_dict,
    source="web",
    actual_state=None,
    actual_reason=None,
):
    global recent_prediction_states

    load_eldercare_models()

    input_df = build_input_dataframe(input_dict)

    pred_encoded = model.predict(input_df)
    state = str(label_encoder.inverse_transform(pred_encoded)[0])

    reason_encoded = reason_model.predict(input_df)
    reason = str(reason_encoder.inverse_transform(reason_encoded)[0])

    risk_score = risk_score_by_state(state)
    guardian = guardian_alert_by_state(state)

    should_save = is_save_history_state(state)

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
        "saved_to_history": should_save,
        "save_policy": "위험/주의만 저장, 그 외 상태는 최신 결과로만 표시",
    }

    # 실제 예측 흐름은 항상 기억
    # 그래야 주의 3회 연속 판단이 정상/수면/식사/외출로 끊길 수 있음
    recent_prediction_states.append(state)
    recent_prediction_states = recent_prediction_states[-50:]

    # 위험 / 주의만 기록 저장
    if should_save:
        history = load_abnormal_history()
        history.append(result)

        # 최근 위험/주의 기록 300개만 저장
        history = history[-300:]

        save_abnormal_history(history)

    return result


@router.get("/")
def root():
    return {
        "message": "이상행동 관제 API 실행중",
        "service": "독거노인 돌봄 상태 예측 API",
        "model_dir": str(MODEL_DIR),
        "history_path": str(ABNORMAL_HISTORY_PATH),
        "save_policy": "위험/주의만 저장",
    }


@router.get("/health")
def health():
    file_status = get_model_file_status()

    try:
        if file_status["all_exists"]:
            load_eldercare_models()
            model_state = "loaded"
            status = "ok"
            message = "이상행동 모델이 정상 로드되었습니다."
        else:
            model_state = "not-loaded"
            status = "warning"
            message = "이상행동 모델 파일이 없어 예측은 불가능하지만 API는 실행 중입니다."

        feature_list = load_feature_list()

        return {
            "status": status,
            "api": "abnormal-running",
            "model": model_state,
            "message": message,
            "feature_count": len(feature_list),
            "simulation_file": simulation_filename,
            "simulation_index": simulation_index,
            "model_dir": str(MODEL_DIR),
            "model_path": str(ELDERCARE_MODEL_PATH),
            "features_path": str(ELDERCARE_FEATURES_PATH),
            "missing_files": file_status["missing_files"],
            "save_policy": "위험/주의만 저장",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        return {
            "status": "error",
            "api": "abnormal-running",
            "model": "not-loaded",
            "message": str(e),
            "missing_files": file_status["missing_files"],
            "save_policy": "위험/주의만 저장",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


@router.get("/features")
def get_features():
    feature_list = load_feature_list()
    file_status = get_model_file_status()

    return {
        "feature_count": len(feature_list),
        "features": feature_list,
        "model_loaded": model is not None,
        "model_files_ready": file_status["all_exists"],
        "missing_files": file_status["missing_files"],
    }


@router.post("/predict")
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


@router.post("/simulation/upload")
async def upload_simulation_file(file: UploadFile = File(...)):
    global simulation_df
    global simulation_index
    global simulation_filename
    global recent_prediction_states

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
    recent_prediction_states = []

    return {
        "message": "시뮬레이션 파일 업로드 완료",
        "filename": filename,
        "rows": len(simulation_df),
        "columns": list(simulation_df.columns),
        "start_index": simulation_index,
        "save_policy": "위험/주의만 저장",
    }


@router.get("/simulation/status")
def simulation_status():
    if simulation_df is None:
        return {
            "loaded": False,
            "message": "업로드된 시뮬레이션 파일이 없습니다.",
            "save_policy": "위험/주의만 저장",
        }

    return {
        "loaded": True,
        "filename": simulation_filename,
        "rows": len(simulation_df),
        "current_index": simulation_index,
        "remaining": max(len(simulation_df) - simulation_index, 0),
        "auto_interval_seconds": 10,
        "save_policy": "위험/주의만 저장",
    }


@router.post("/simulation/reset")
def simulation_reset():
    global simulation_index
    global recent_prediction_states

    if simulation_df is None:
        raise HTTPException(
            status_code=400,
            detail="업로드된 시뮬레이션 파일이 없습니다.",
        )

    simulation_index = 0
    recent_prediction_states = []

    return {
        "message": "시뮬레이션 인덱스 초기화 완료",
        "current_index": simulation_index,
        "save_policy": "위험/주의만 저장",
    }


@router.post("/simulation/next")
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
        "remaining": max(len(simulation_df) - simulation_index, 0),
    }

    return result


@router.get("/history")
def get_history(limit: int = 20):
    history = load_abnormal_history()
    history = list(reversed(history))

    limit = max(1, min(limit, 100))

    return {
        "count": min(limit, len(history)),
        "items": history[:limit],
        "save_policy": "위험/주의만 저장",
    }


@router.delete("/history")
def delete_history():
    global recent_prediction_states

    save_abnormal_history([])
    recent_prediction_states = []

    return {
        "deleted": True,
        "message": "위험/주의 이상행동 기록을 삭제했습니다.",
        "save_policy": "위험/주의만 저장",
    }


@router.get("/alerts")
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
        "save_policy": "위험/주의만 저장",
    }


@router.get("/dashboard")
def dashboard():
    history = load_abnormal_history()

    latest = history[-1] if history else None
    recent = list(reversed(history))[:10]

    return {
        "latest": latest,
        "recent": recent,
        "save_policy": "위험/주의만 저장",
    }


@router.get("/stats")
def stats():
    history = load_abnormal_history()

    total = len(history)
    danger_count = len([x for x in history if x.get("state") == "위험"])
    warning_count = len([x for x in history if x.get("state") == "주의"])
    guardian_alert_count = len([x for x in history if x.get("guardian_alert") is True])

    latest = history[-1] if history else None

    return {
        "total": total,
        "danger_count": danger_count,
        "warning_count": warning_count,

        # 저장 정책상 아래 상태들은 기록에 저장하지 않으므로 0
        "outing_count": 0,
        "meal_count": 0,
        "sleep_count": 0,

        "guardian_alert_count": guardian_alert_count,
        "latest": latest,
        "save_policy": "위험/주의만 저장",
    }