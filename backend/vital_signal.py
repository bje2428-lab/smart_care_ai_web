from pathlib import Path
from datetime import datetime
import json
import sys
import traceback
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel


# =========================================================
# 경로 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
HISTORY_DIR = DATA_DIR / "vital_history"
HISTORY_FILE = HISTORY_DIR / "vital_history.json"

for path in [ROOT_DIR, BACKEND_DIR, MODEL_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =========================================================
# Router
# main.py에서 app.include_router(vital_router)로 연결
#
# GET  /vital/health
# GET  /vital/features
# POST /vital/predict
# POST /vital_predict       이전 호환용
# GET  /vital/history
# GET  /vital/stats
# =========================================================

router = APIRouter(tags=["Vital Signal"])


# =========================================================
# 모델 후보 경로
# 있으면 사용하고, 없으면 rule-based로 동작
# =========================================================

MODEL_CANDIDATES = [
    MODEL_DIR / "vital_signal_model.pkl",
    MODEL_DIR / "vital_model.pkl",
    MODEL_DIR / "vital_anomaly_model.pkl",
    MODEL_DIR / "respiration_anomaly_model.pkl",
]


vital_model = None
loaded_model_path = None
model_load_error = None


# =========================================================
# 입력 스키마
# =========================================================

class VitalInput(BaseModel):
    heart_rate: Optional[float] = None
    respiration_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    movement: Optional[float] = None
    signal_quality: Optional[float] = None


# =========================================================
# 공통 유틸
# =========================================================

def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def ensure_history_file():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_history() -> List[Dict[str, Any]]:
    ensure_history_file()

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_history_item(item: Dict[str, Any]):
    ensure_history_file()

    history = load_history()
    history.insert(0, item)

    history = history[:500]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# =========================================================
# 모델 로딩
# =========================================================

def startup_vital_signal():
    """
    main.py startup에서 호출됨.
    모델 파일이 없어도 서버는 죽지 않음.
    """
    global vital_model
    global loaded_model_path
    global model_load_error

    ensure_history_file()

    model_path = find_model_path()

    if model_path is None:
        vital_model = None
        loaded_model_path = None
        model_load_error = None
        print("[VITAL] 생체신호 pkl 모델 없음. rule-based 방식으로 동작합니다.")
        return

    try:
        vital_model = joblib.load(model_path)
        loaded_model_path = str(model_path)
        model_load_error = None
        print(f"[VITAL] 모델 로드 성공: {model_path}")

    except Exception as e:
        vital_model = None
        loaded_model_path = None
        model_load_error = str(e)
        print(f"[VITAL] 모델 로드 실패: {e}")
        traceback.print_exc()


def shutdown_vital_signal():
    print("[VITAL] shutdown 완료")


# =========================================================
# 상태 확인
# main.py에서 health as vital_health_status로 import함
# =========================================================

def health():
    return {
        "status": "ok",
        "api": "vital-running",
        "message": "Vital Signal API is running",

        "model_loaded": vital_model is not None,
        "model_path": loaded_model_path,
        "model_candidates": [str(path) for path in MODEL_CANDIDATES],
        "model_load_error": model_load_error,

        "tensorflow_available": False,
        "tensorflow_version": None,
        "tensorflow_note": "현재 vital_signal.py는 TensorFlow 없이 동작하도록 구성되어 있습니다.",

        "history_file": str(HISTORY_FILE),
        "history_file_exists": HISTORY_FILE.exists(),

        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "model_dir": str(MODEL_DIR),
    }


# =========================================================
# Feature 처리
# =========================================================

def normalize_vital_input(data: Dict[str, Any]) -> Dict[str, float]:
    """
    다양한 이름으로 들어와도 최대한 맞춰서 사용.
    """
    heart_rate = to_float(
        data.get("heart_rate")
        or data.get("heartRate")
        or data.get("hr")
        or data.get("bpm")
    )

    respiration_rate = to_float(
        data.get("respiration_rate")
        or data.get("respirationRate")
        or data.get("breathing_rate")
        or data.get("breath_rate")
        or data.get("rr")
        or data.get("resp")
    )

    spo2 = to_float(
        data.get("spo2")
        or data.get("SpO2")
        or data.get("oxygen")
        or data.get("oxygen_saturation")
    )

    temperature = to_float(
        data.get("temperature")
        or data.get("temp")
        or data.get("body_temperature")
    )

    movement = to_float(
        data.get("movement")
        or data.get("motion")
        or data.get("activity")
    )

    signal_quality = to_float(
        data.get("signal_quality")
        or data.get("quality")
        or data.get("snr")
    )

    return {
        "heart_rate": heart_rate,
        "respiration_rate": respiration_rate,
        "spo2": spo2,
        "temperature": temperature,
        "movement": movement,
        "signal_quality": signal_quality,
    }


def extract_features_from_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    """
    CSV 업로드용.
    숫자 컬럼을 기준으로 평균/최소/최대/표준편차를 만들고,
    가능한 경우 심박/호흡/산소포화도 컬럼을 찾아서 대표값으로 사용.
    """
    if df.empty:
        raise ValueError("CSV 파일에 데이터가 없습니다.")

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        raise ValueError("CSV 안에 숫자형 데이터가 없습니다.")

    features = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        if series.empty:
            continue

        safe_col = str(col).strip().replace(" ", "_")

        features[f"{safe_col}_mean"] = float(series.mean())
        features[f"{safe_col}_min"] = float(series.min())
        features[f"{safe_col}_max"] = float(series.max())
        features[f"{safe_col}_std"] = float(series.std()) if len(series) > 1 else 0.0
        features[f"{safe_col}_range"] = float(series.max() - series.min())

    lower_map = {str(col).lower(): col for col in numeric_df.columns}

    def find_column(candidates):
        for name in candidates:
            for lower_name, original_col in lower_map.items():
                if name in lower_name:
                    return original_col
        return None

    hr_col = find_column(["heart", "hr", "bpm"])
    rr_col = find_column(["resp", "breath", "rr"])
    spo2_col = find_column(["spo2", "oxygen"])
    temp_col = find_column(["temp"])
    move_col = find_column(["move", "motion", "activity"])

    result = {
        "heart_rate": 0.0,
        "respiration_rate": 0.0,
        "spo2": 0.0,
        "temperature": 0.0,
        "movement": 0.0,
        "signal_quality": 0.0,
    }

    if hr_col is not None:
        result["heart_rate"] = float(numeric_df[hr_col].dropna().mean())

    if rr_col is not None:
        result["respiration_rate"] = float(numeric_df[rr_col].dropna().mean())

    if spo2_col is not None:
        result["spo2"] = float(numeric_df[spo2_col].dropna().mean())

    if temp_col is not None:
        result["temperature"] = float(numeric_df[temp_col].dropna().mean())

    if move_col is not None:
        result["movement"] = float(numeric_df[move_col].dropna().mean())

    result.update(features)

    return result


# =========================================================
# Rule-based 판단
# 모델 없을 때 사용
# =========================================================

def rule_based_vital_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    heart_rate = to_float(features.get("heart_rate"))
    respiration_rate = to_float(features.get("respiration_rate"))
    spo2 = to_float(features.get("spo2"))
    temperature = to_float(features.get("temperature"))
    movement = to_float(features.get("movement"))
    signal_quality = to_float(features.get("signal_quality"))

    reasons = []
    score = 0

    # 심박수 기준
    if heart_rate > 0:
        if heart_rate < 45:
            score += 35
            reasons.append("심박수가 낮습니다.")
        elif heart_rate > 120:
            score += 35
            reasons.append("심박수가 높습니다.")
        elif heart_rate < 55 or heart_rate > 100:
            score += 15
            reasons.append("심박수가 주의 범위입니다.")

    # 호흡수 기준
    if respiration_rate > 0:
        if respiration_rate < 8:
            score += 40
            reasons.append("호흡수가 매우 낮습니다.")
        elif respiration_rate > 30:
            score += 40
            reasons.append("호흡수가 매우 높습니다.")
        elif respiration_rate < 12 or respiration_rate > 24:
            score += 20
            reasons.append("호흡수가 주의 범위입니다.")

    # 산소포화도 기준
    if spo2 > 0:
        if spo2 < 90:
            score += 45
            reasons.append("산소포화도가 위험 범위입니다.")
        elif spo2 < 95:
            score += 25
            reasons.append("산소포화도가 낮은 편입니다.")

    # 체온 기준
    if temperature > 0:
        if temperature < 35.0:
            score += 30
            reasons.append("체온이 낮습니다.")
        elif temperature >= 38.0:
            score += 30
            reasons.append("체온이 높습니다.")
        elif temperature >= 37.5:
            score += 15
            reasons.append("미열 가능성이 있습니다.")

    # 움직임 기준
    if movement > 0:
        if movement < 0.02:
            score += 10
            reasons.append("움직임이 매우 적습니다.")

    # 신호 품질
    if signal_quality > 0 and signal_quality < 0.3:
        score += 10
        reasons.append("센서 신호 품질이 낮습니다.")

    score = min(score, 100)

    if score >= 70:
        status = "Danger"
        state = "위험"
        alert = True
        message = "생체신호 이상 위험이 감지되었습니다."
    elif score >= 35:
        status = "Warning"
        state = "주의"
        alert = False
        message = "생체신호 주의 상태입니다."
    else:
        status = "Normal"
        state = "정상"
        alert = False
        message = "생체신호가 정상 범위로 판단됩니다."

    if not reasons:
        reasons.append("위험 기준에 해당하는 생체신호 이상이 감지되지 않았습니다.")

    return {
        "status": status,
        "state": state,
        "alert": alert,
        "risk_score": score,
        "message": message,
        "reasons": reasons,
    }


def model_based_prediction(features: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    pkl 모델이 있을 경우 사용.
    실패하면 None 반환해서 rule-based로 fallback.
    """
    if vital_model is None:
        return None

    try:
        feature_df = pd.DataFrame([features])
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if hasattr(vital_model, "feature_names_in_"):
            feature_names = list(vital_model.feature_names_in_)

            for col in feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0

            feature_df = feature_df[feature_names]

        pred = vital_model.predict(feature_df)[0]

        prob = None

        if hasattr(vital_model, "predict_proba"):
            proba = vital_model.predict_proba(feature_df)[0]
            prob = float(max(proba))

        pred_text = str(pred)

        if pred_text.lower() in ["1", "danger", "abnormal", "anomaly", "warning"]:
            status = "Warning"
            state = "주의"
            risk_score = int((prob or 0.65) * 100)
            alert = risk_score >= 70
        else:
            status = "Normal"
            state = "정상"
            risk_score = int((prob or 0.2) * 100)
            alert = False

        return {
            "status": status,
            "state": state,
            "alert": alert,
            "risk_score": risk_score,
            "model_prediction": pred_text,
            "model_probability": prob,
            "message": "생체신호 모델 예측 결과입니다.",
            "reasons": ["pkl 모델을 사용해 생체신호 상태를 예측했습니다."],
        }

    except Exception as e:
        print(f"[VITAL MODEL PREDICT ERROR] {e}")
        traceback.print_exc()
        return None


def run_prediction(features: Dict[str, float], source: str = "manual") -> Dict[str, Any]:
    model_result = model_based_prediction(features)

    if model_result is not None:
        result = model_result
        mode = "model"
    else:
        result = rule_based_vital_prediction(features)
        mode = "rule_based"

    final_result = {
        "success": True,
        "time": now_text(),
        "source": source,
        "mode": mode,

        "status": result["status"],
        "state": result["state"],
        "alert": result["alert"],
        "risk_score": result["risk_score"],
        "message": result["message"],
        "reasons": result.get("reasons", []),

        "guardian_alert": result["alert"],
        "guardian_message": (
            "생체신호 위험 상태로 판단되어 보호자 확인이 필요합니다."
            if result["alert"]
            else "보호자 즉시 알림 기준에는 도달하지 않았습니다."
        ),

        "features": features,

        "model_loaded": vital_model is not None,
        "model_path": loaded_model_path,
    }

    save_history_item(final_result)

    return final_result


# =========================================================
# API
# =========================================================

@router.get("/vital/health")
def vital_health():
    return health()


@router.get("/vital/features")
def vital_features():
    return {
        "status": "ok",
        "required_or_supported_features": [
            "heart_rate",
            "respiration_rate",
            "spo2",
            "temperature",
            "movement",
            "signal_quality",
        ],
        "description": {
            "heart_rate": "심박수 bpm",
            "respiration_rate": "호흡수 회/분",
            "spo2": "산소포화도 %",
            "temperature": "체온 ℃",
            "movement": "움직임 정도",
            "signal_quality": "센서 신호 품질",
        },
        "csv_note": "CSV 업로드 시 숫자 컬럼을 자동으로 통계 feature로 변환합니다.",
    }


@router.post("/vital/predict")
async def vital_predict(
    data: Optional[VitalInput] = None,
    file: Optional[UploadFile] = File(default=None),
):
    try:
        if file is not None:
            if not file.filename.lower().endswith(".csv"):
                return {
                    "success": False,
                    "status": "Error",
                    "message": "CSV 파일만 업로드할 수 있습니다.",
                    "time": now_text(),
                }

            df = pd.read_csv(file.file)
            features = extract_features_from_dataframe(df)
            return run_prediction(features, source=file.filename)

        if data is None:
            return {
                "success": False,
                "status": "Error",
                "message": "예측할 생체신호 데이터가 없습니다.",
                "time": now_text(),
            }

        features = normalize_vital_input(data.dict())
        return run_prediction(features, source="manual")

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "status": "Error",
            "message": f"생체신호 예측 중 오류가 발생했습니다: {e}",
            "time": now_text(),
        }


@router.post("/vital_predict")
async def old_vital_predict(data: VitalInput):
    """
    이전 코드 호환용 API
    """
    try:
        features = normalize_vital_input(data.dict())
        return run_prediction(features, source="old_vital_predict")

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "status": "Error",
            "message": f"생체신호 예측 중 오류가 발생했습니다: {e}",
            "time": now_text(),
        }


@router.get("/vital/history")
def vital_history(limit: int = 30):
    history = load_history()
    limit = max(1, min(limit, 100))

    return {
        "status": "ok",
        "count": min(len(history), limit),
        "total": len(history),
        "history": history[:limit],
    }


@router.get("/vital/stats")
def vital_stats():
    history = load_history()

    total = len(history)
    normal_count = sum(1 for item in history if item.get("status") == "Normal")
    warning_count = sum(1 for item in history if item.get("status") == "Warning")
    danger_count = sum(1 for item in history if item.get("status") == "Danger")
    alert_count = sum(1 for item in history if item.get("alert") is True)

    return {
        "status": "ok",
        "total": total,
        "normal_count": normal_count,
        "warning_count": warning_count,
        "danger_count": danger_count,
        "alert_count": alert_count,
        "model_loaded": vital_model is not None,
        "model_path": loaded_model_path,
    }