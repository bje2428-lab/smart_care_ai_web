from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional
from uuid import uuid4
import math
import os
import traceback
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile


router = APIRouter(prefix="/integrated", tags=["Integrated Dashboard"])

CODE_VERSION = "integrated_v6_ESTIMATION_DB_SAVE_LOCK"

SESSIONS: Dict[str, Dict[str, Any]] = {}
SAVED_LOGS = []

DEFAULT_FPS = 10
DEFAULT_FALL_WINDOW_FRAMES = 10

# =========================================================
# 핵심 수정
# 이상행동 CSV는 TimeStamp를 보지 않고 무조건 1행 = 1초로 읽는다.
# =========================================================
ABNORMAL_ROW_SECONDS = 1

VITAL_SAMPLE_INTERVAL_SECONDS = 0.003
VITAL_GENERATED_TIME_COL = "_vital_generated_time"

RAW_VITAL_TIME_COLS = [
    "Time_Seconds",
    "time_seconds",
    "time",
    "seconds",
    "sec",
    VITAL_GENERATED_TIME_COL,
]

RAW_VITAL_SIGNAL_COLS = [
    "VitalSignal",
    "vital_signal",
    "signal",
    "value",
]

RAW_VITAL_CONDITION_COLS = [
    "Condition",
    "condition",
    "label",
    "state",
]

VITAL_FEATURE_NAMES = [
    "mean",
    "std",
    "peak_to_peak",
    "zero_crossings",
    "fft_mean",
    "fft_max",
    "fft_std",
]

FEATURE_COLUMN_ALIASES = {
    "mean": ["mean", "signal_mean", "vital_mean"],
    "std": ["std", "signal_std", "vital_std"],
    "peak_to_peak": ["peak_to_peak", "peak2peak", "p2p", "ptp", "range"],
    "zero_crossings": ["zero_crossings", "zero_crossing", "zc"],
    "fft_mean": ["fft_mean", "fft_avg"],
    "fft_max": ["fft_max", "fft_peak"],
    "fft_std": ["fft_std"],
}

ABNORMAL_ALLOWED_STATES = ["기타", "수면", "식사", "외출", "주의", "위험"]

ABNORMAL_RISK_SCORE_MAP = {
    "위험": 92,
    "주의": 65,
    "외출": 38,
    "식사": 30,
    "수면": 22,
    "기타": 18,
}

ABNORMAL_NUMERIC_STATE_MAP = {
    "0": "기타",
    "1": "수면",
    "2": "식사",
    "3": "외출",
    "4": "주의",
    "5": "위험",
}

ABNORMAL_STATE_ALIASES = {
    "위험": [
        "위험",
        "danger",
        "emergency",
        "critical",
        "highrisk",
        "high_risk",
        "risk_high",
    ],
    "주의": [
        "주의",
        "warning",
        "caution",
        "abnormal",
        "care",
        "watch",
        "wandering",
        "inactive",
        "inactivity",
        "no_activity",
        "long_stay",
    ],
    "외출": [
        "외출",
        "outing",
        "outside",
        "outdoor",
        "goout",
        "go_out",
        "leave",
        "leaving",
        "out",
    ],
    "식사": [
        "식사",
        "meal",
        "eat",
        "eating",
        "food",
        "breakfast",
        "lunch",
        "dinner",
    ],
    "수면": [
        "수면",
        "sleep",
        "sleeping",
        "rest",
        "resting",
        "bed",
        "lying",
        "lie",
    ],
    "기타": [
        "기타",
        "정상",
        "일상",
        "normal",
        "etc",
        "daily",
        "ordinary",
        "unknown",
        "none",
    ],
}

ABNORMAL_ESTIMATION_COLUMN_CANDIDATES = [
    "Estimation", "estimation", "ESTIMATION",
    "State", "state", "STATE",
    "Status", "status", "STATUS",
    "Label", "label", "LABEL",
    "Class", "class", "CLASS",
    "Target", "target", "TARGET",
    "Activity", "activity", "ACTIVITY",
    "Behavior", "behavior", "BEHAVIOR",
    "Action", "action", "ACTION",
    "Result", "result", "RESULT",
    "Prediction", "prediction", "PREDICTION",
    "Pred", "pred", "PRED",
    "생활상태", "상태", "예측상태", "추정상태", "판정", "결과", "라벨", "행동", "활동",
]

ABNORMAL_REASON_COLUMN_CANDIDATES = [
    "Reason",
    "reason",
    "REASON",
    "Information",
    "information",
    "Description",
    "description",
    "Detail",
    "detail",
    "Cause",
    "cause",
]

ABNORMAL_TEXT_COLUMN_CANDIDATES = [
    "Estimation",
    "estimation",
    "state",
    "State",
    "label",
    "Label",
    "activity",
    "Activity",
    "behavior",
    "Behavior",
    "action",
    "Action",
    "Reason",
    "reason",
    "Information",
    "information",
    "Description",
    "description",
    "Detail",
    "detail",
]

ABNORMAL_VITAL_ALIASES = {
    "HeartRate": [
        "HeartRate",
        "heart_rate",
        "heartRate",
        "hr",
        "bpm",
        "심박",
        "심박수",
        "분당심박",
        "분당_심박",
    ],
    "BreathRate": [
        "BreathRate",
        "breath_rate",
        "breathRate",
        "respiratory_rate",
        "respiratoryRate",
        "rr",
        "호흡",
        "호흡수",
        "분당호흡",
        "분당_호흡",
    ],
    "SPO2": [
        "SPO2",
        "SpO2",
        "spo2",
        "oxygen",
        "oxygen_saturation",
        "blood_oxygen",
        "산소포화도",
        "혈중산소농도",
    ],
    "SkinTemperature": [
        "SkinTemperature",
        "skin_temperature",
        "skinTemperature",
        "temperature",
        "temp",
        "body_temp",
        "체온",
        "피부온도",
        "피부온도변화",
    ],
    "StressIndex": [
        "StressIndex",
        "stress_index",
        "stressIndex",
        "stress_score",
        "StressScore",
        "stress",
        "스트레스",
        "스트레스지수",
        "스트레스점수",
    ],
}

ELDERCARE_FEATURE_ALIASES = {
    "Temperature": ["Temperature", "temperature", "temp", "실내온도", "온도"],
    "Humidity": ["Humidity", "humidity", "humid", "습도"],
    "Illuminance": ["Illuminance", "illuminance", "illumination", "light", "lux", "조도"],
    "Activity_IR": ["Activity_IR", "activity_ir", "ir", "pir", "motion", "움직임", "활동감지"],
    "CO2": ["CO2", "co2", "이산화탄소"],
    "TVOC": ["TVOC", "tvoc", "voc", "휘발성유기화합물"],
    "HeartRate": ABNORMAL_VITAL_ALIASES["HeartRate"],
    "BreathRate": ABNORMAL_VITAL_ALIASES["BreathRate"],
    "SPO2": ABNORMAL_VITAL_ALIASES["SPO2"],
    "SkinTemperature": ABNORMAL_VITAL_ALIASES["SkinTemperature"],
    "SleepPhase": ["SleepPhase", "sleep_phase", "sleepPhase", "수면단계", "수면상태"],
    "SleepScore": ["SleepScore", "sleep_score", "sleepScore", "수면점수"],
    "WalkingSteps": ["WalkingSteps", "walking_steps", "steps", "step_count", "걸음수", "보행수"],
    "StressIndex": ABNORMAL_VITAL_ALIASES["StressIndex"],
    "ActivityIntensity": ["ActivityIntensity", "activity_intensity", "activityIntensity", "intensity", "활동강도"],
    "CaloricExpenditure": ["CaloricExpenditure", "caloric_expenditure", "calories", "calorie", "kcal", "소모칼로리", "칼로리"],
    "Button": ["Button", "button", "emergency_button", "panic_button", "응급버튼", "버튼"],
    "Shout": ["Shout", "shout", "voice", "scream", "응급음성", "고함", "비명"],
}


# =========================================================
# 경로 / 모델
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"


def resolve_model_path(filename: str) -> Path:
    """
    프로젝트의 models 폴더를 우선 사용한다.
    단, 테스트나 파일 위치가 살짝 다른 경우에도 죽지 않도록 몇 가지 후보를 같이 확인한다.
    """
    candidates = [
        MODEL_DIR / filename,
        BACKEND_DIR / "models" / filename,
        BACKEND_DIR / filename,
        ROOT_DIR / filename,
        Path.cwd() / "models" / filename,
        Path.cwd() / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return MODEL_DIR / filename


FALL_MODEL_PATH = resolve_model_path("mmwave_rf_smote_model.pkl")
FALL_META_PATH = resolve_model_path("mmwave_rf_smote_model_meta.json")

ELDERCARE_MODEL_PATH = resolve_model_path("eldercare_xgb_model.pkl")
ELDERCARE_REASON_MODEL_PATH = resolve_model_path("eldercare_reason_model.pkl")
ELDERCARE_LABEL_ENCODER_PATH = resolve_model_path("eldercare_label_encoder.pkl")
ELDERCARE_REASON_ENCODER_PATH = resolve_model_path("eldercare_reason_encoder.pkl")
ELDERCARE_FEATURES_PATH = resolve_model_path("eldercare_features.pkl")

fall_model = None
fall_meta = None
fall_feature_columns = []
fall_threshold = 0.4

eldercare_model = None
eldercare_reason_model = None
eldercare_label_encoder = None
eldercare_reason_encoder = None
eldercare_features = []

recent_abnormal_states = []


# =========================================================
# 모델 로딩
# =========================================================

def load_fall_model():
    global fall_model
    global fall_meta
    global fall_feature_columns
    global fall_threshold

    if fall_model is not None:
        return

    if not FALL_MODEL_PATH.exists():
        raise FileNotFoundError(f"낙상 모델 파일이 없습니다: {FALL_MODEL_PATH}")

    if not FALL_META_PATH.exists():
        raise FileNotFoundError(f"낙상 메타 파일이 없습니다: {FALL_META_PATH}")

    fall_model = joblib.load(FALL_MODEL_PATH)

    with open(FALL_META_PATH, "r", encoding="utf-8") as f:
        fall_meta = json.load(f)

    fall_feature_columns = list(
        fall_meta.get("feature_columns")
        or fall_meta.get("feature_names")
        or []
    )

    if not fall_feature_columns:
        raise ValueError("낙상 메타 파일에 feature_columns가 없습니다.")

    meta_threshold = float(
        fall_meta.get("fall_alert_threshold")
        or fall_meta.get("best_threshold")
        or fall_meta.get("default_threshold")
        or 0.5
    )

    # 화면의 Fall Alert는 이전 기준처럼 70% 이상일 때만 확정한다.
    # meta best_threshold(예: 0.4)는 모델 평가용 기준으로만 참고하고,
    # 통합 관제 알림은 과잉 감지를 줄이기 위해 0.70을 기본으로 둔다.
    fall_threshold = float(os.getenv("FALL_ALERT_THRESHOLD", max(0.70, meta_threshold)))

    print("[INTEGRATED FALL MODEL] 로드 완료")
    print(f"[INTEGRATED FALL MODEL] threshold: {fall_threshold}")
    print(f"[INTEGRATED FALL MODEL] feature_count: {len(fall_feature_columns)}")


def load_eldercare_models():
    global eldercare_model
    global eldercare_reason_model
    global eldercare_label_encoder
    global eldercare_reason_encoder
    global eldercare_features

    if eldercare_model is not None:
        return

    required_files = [
        ELDERCARE_MODEL_PATH,
        ELDERCARE_REASON_MODEL_PATH,
        ELDERCARE_LABEL_ENCODER_PATH,
        ELDERCARE_REASON_ENCODER_PATH,
        ELDERCARE_FEATURES_PATH,
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(f"이상행동 모델 파일이 없습니다: {missing}")

    eldercare_model = joblib.load(ELDERCARE_MODEL_PATH)
    eldercare_reason_model = joblib.load(ELDERCARE_REASON_MODEL_PATH)
    eldercare_label_encoder = joblib.load(ELDERCARE_LABEL_ENCODER_PATH)
    eldercare_reason_encoder = joblib.load(ELDERCARE_REASON_ENCODER_PATH)
    eldercare_features = list(joblib.load(ELDERCARE_FEATURES_PATH))

    if not eldercare_features:
        raise ValueError("eldercare_features.pkl에 feature 목록이 없습니다.")

    print("[INTEGRATED ELDERCARE MODEL] 로드 완료")
    print(f"[INTEGRATED ELDERCARE MODEL] feature_count: {len(eldercare_features)}")


def get_model_file_status():
    files = {
        "fall_model": FALL_MODEL_PATH,
        "fall_meta": FALL_META_PATH,
        "eldercare_model": ELDERCARE_MODEL_PATH,
        "eldercare_reason_model": ELDERCARE_REASON_MODEL_PATH,
        "eldercare_label_encoder": ELDERCARE_LABEL_ENCODER_PATH,
        "eldercare_reason_encoder": ELDERCARE_REASON_ENCODER_PATH,
        "eldercare_features": ELDERCARE_FEATURES_PATH,
    }

    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
        }
        for name, path in files.items()
    }


# =========================================================
# vital_signal.py optional
# =========================================================

try:
    import vital_signal as vital_module
    VITAL_MODULE_IMPORT_ERROR = None
except Exception as e:
    vital_module = None
    VITAL_MODULE_IMPORT_ERROR = str(e)


# =========================================================
# MongoDB optional
# =========================================================

mongo_client = None
mongo_collection = None
mongo_error = None

try:
    from pymongo import MongoClient

    MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "smart_care_ai")
    MONGO_COLLECTION_NAME = os.getenv(
        "MONGO_INTEGRATED_COLLECTION",
        "integrated_detection_events",
    )

    mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=1500)
    mongo_client.admin.command("ping")
    mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]

    print("[DB] IntegratedDashboard MongoDB 연결 성공")

except Exception as e:
    mongo_error = str(e)
    mongo_client = None
    mongo_collection = None
    print("[DB] IntegratedDashboard MongoDB 연결 실패 - 메모리 로그 사용")


# =========================================================
# 공통 유틸
# =========================================================

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def clamp(value, min_value=0, max_value=100):
    try:
        value = float(value)
    except Exception:
        value = 0

    return max(min_value, min(max_value, value))


def clean_text(value):
    return str(value or "").replace("\ufeff", "").strip()


def compact_text(value):
    return (
        clean_text(value)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def normalize_column_name(value):
    return (
        str(value or "")
        .replace("\ufeff", "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def is_valid_text(value):
    text = clean_text(value)
    lowered = text.lower()

    if lowered in ["", "-", "nan", "none", "null"]:
        return False

    return True


def find_col(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None

    column_map = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)

        if key in column_map:
            return column_map[key]

    return None


def numeric_series(df: pd.DataFrame, candidates):
    col = find_col(df, candidates)

    if col is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(df[col], errors="coerce").dropna()


def text_value(df: pd.DataFrame, candidates, default="-"):
    col = find_col(df, candidates)

    if col is None or df.empty:
        return default

    values = df[col].dropna().astype(str)

    if values.empty:
        return default

    value = values.iloc[-1].replace("\ufeff", "").strip()

    if not value:
        return default

    return value


def mean_value(df: pd.DataFrame, candidates):
    series = numeric_series(df, candidates)

    if series.empty:
        return None

    return float(series.mean())


def round_optional(value, decimals=1):
    if value is None:
        return None

    try:
        return round(float(value), decimals)
    except Exception:
        return value


def json_safe(data):
    if isinstance(data, dict):
        return {key: json_safe(value) for key, value in data.items()}

    if isinstance(data, list):
        return [json_safe(value) for value in data]

    if isinstance(data, tuple):
        return [json_safe(value) for value in data]

    if isinstance(data, np.integer):
        return int(data)

    if isinstance(data, np.floating):
        value = float(data)
        if math.isnan(value):
            return None
        return value

    if isinstance(data, float) and math.isnan(data):
        return None

    return data


async def read_upload_file(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(BytesIO(content))
        else:
            try:
                df = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(content), encoding="cp949")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename} 파일을 읽을 수 없습니다: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename} 파일이 비어 있습니다.",
        )

    df.columns = [
        str(col).replace("\ufeff", "").strip()
        for col in df.columns
    ]

    return df


# =========================================================
# 시간 계산 / 슬라이싱
# =========================================================

def get_frame_range(df: Optional[pd.DataFrame]):
    if df is None or df.empty:
        return 0, 0, 0

    frame_col = find_col(df, ["frame", "Frame", "frame_id", "frameId"])

    if frame_col:
        frames = pd.to_numeric(df[frame_col], errors="coerce").dropna()

        if frames.empty:
            return 0, len(df) - 1, len(df)

        min_frame = int(frames.min())
        max_frame = int(frames.max())

        return min_frame, max_frame, max_frame - min_frame + 1

    return 0, len(df) - 1, len(df)


def resolve_step_seconds(abnormal_df, fps, fall_window_frames):
    """
    핵심:
    이상행동 CSV가 있으면 TimeStamp를 절대 기준으로 쓰지 않는다.
    사용자가 원하는 기준은 1행 = 1초다.
    """
    if abnormal_df is not None and not abnormal_df.empty:
        return ABNORMAL_ROW_SECONDS

    return fall_window_frames / fps


def total_steps_by_window(fall_df, abnormal_df, vital_df, fps, window_frames, step_seconds):
    total_steps = 0

    if abnormal_df is not None and not abnormal_df.empty:
        # 핵심:
        # 이상행동은 무조건 한 행씩 읽는다.
        # 600행이면 600 step = 600초다.
        total_steps = max(total_steps, len(abnormal_df))

    if fall_df is not None and not fall_df.empty:
        _, _, frame_count = get_frame_range(fall_df)
        duration_seconds = frame_count / fps
        fall_steps = max(1, math.ceil(duration_seconds / step_seconds))
        total_steps = max(total_steps, fall_steps)

    if vital_df is not None and not vital_df.empty:
        time_col = find_col(vital_df, RAW_VITAL_TIME_COLS)

        if time_col:
            times = pd.to_numeric(vital_df[time_col], errors="coerce").dropna()

            if not times.empty:
                duration = max(
                    VITAL_SAMPLE_INTERVAL_SECONDS,
                    float(times.max() - times.min()) + VITAL_SAMPLE_INTERVAL_SECONDS,
                )
                vital_steps = max(1, math.ceil(duration / step_seconds))
                total_steps = max(total_steps, vital_steps)
            else:
                duration = len(vital_df) * VITAL_SAMPLE_INTERVAL_SECONDS
                vital_steps = max(1, math.ceil(duration / step_seconds))
                total_steps = max(total_steps, vital_steps)

        else:
            duration = len(vital_df) * VITAL_SAMPLE_INTERVAL_SECONDS
            vital_steps = max(1, math.ceil(duration / step_seconds))
            total_steps = max(total_steps, vital_steps)

    return max(1, int(total_steps))


def slice_by_time_window(df, step, fps, window_frames, mode, step_seconds):
    if df is None or df.empty:
        return pd.DataFrame()

    current_second = step * step_seconds

    if mode == "abnormal":
        # 핵심:
        # TimeStamp가 10행씩 같은 초로 반복되어도 절대 묶지 않는다.
        # 오직 row index 기준으로 1행씩 읽는다.
        return df.iloc[step:step + 1].copy()

    if mode == "fall":
        frame_col = find_col(df, ["frame", "Frame", "frame_id", "frameId"])

        if frame_col:
            frames = pd.to_numeric(df[frame_col], errors="coerce")
            valid_frames = frames.dropna()

            if valid_frames.empty:
                return pd.DataFrame()

            first_frame = float(valid_frames.min())
            start_frame = first_frame + current_second * fps
            end_frame = start_frame + step_seconds * fps

            return df[(frames >= start_frame) & (frames < end_frame)].copy()

        start_row = int(current_second * fps)
        end_row = int((current_second + step_seconds) * fps)

        return df.iloc[start_row:end_row].copy()

    if mode == "vital":
        time_col = find_col(df, RAW_VITAL_TIME_COLS)

        if time_col:
            times = pd.to_numeric(df[time_col], errors="coerce")
            valid_times = times.dropna()

            if valid_times.empty:
                return pd.DataFrame()

            first_time = float(valid_times.min())
            start_time = first_time + current_second
            end_time = start_time + step_seconds

            return df[(times >= start_time) & (times < end_time)].copy()

        start_row = int(math.floor(current_second / VITAL_SAMPLE_INTERVAL_SECONDS))
        end_row = int(math.ceil((current_second + step_seconds) / VITAL_SAMPLE_INTERVAL_SECONDS))

        chunk = df.iloc[start_row:end_row].copy()

        if not chunk.empty:
            generated_times = np.arange(start_row, start_row + len(chunk), dtype=float)
            chunk[VITAL_GENERATED_TIME_COL] = np.round(
                generated_times * VITAL_SAMPLE_INTERVAL_SECONDS,
                6,
            )

        return chunk

    return pd.DataFrame()


# =========================================================
# 낙상 분석
# =========================================================

def center_points(df: pd.DataFrame):
    x_col = find_col(df, ["x"])
    y_col = find_col(df, ["y"])
    z_col = find_col(df, ["z"])
    frame_col = find_col(df, ["frame", "Frame", "frame_id", "frameId"])

    if not x_col or not y_col or not z_col:
        return pd.DataFrame(columns=["x", "y", "z"])

    temp = df[[x_col, y_col, z_col]].copy()
    temp.columns = ["x", "y", "z"]
    temp = temp.apply(pd.to_numeric, errors="coerce").dropna()

    if temp.empty:
        return temp

    if frame_col:
        frames = pd.to_numeric(df.loc[temp.index, frame_col], errors="coerce")
        temp["frame"] = frames
        temp = temp.dropna()

        return temp.groupby("frame")[["x", "y", "z"]].mean().reset_index(drop=True)

    return temp[["x", "y", "z"]]


def movement_after_value(df: pd.DataFrame):
    centers = center_points(df)

    if len(centers) < 2:
        return 0.0

    diffs = centers[["x", "y", "z"]].diff().dropna()
    movement = np.sqrt((diffs ** 2).sum(axis=1))

    if len(movement) == 0:
        return 0.0

    start = int(len(movement) * 0.6)

    return float(movement.iloc[start:].mean())


def stat_features(series: pd.Series, prefix: str):
    series = pd.to_numeric(series, errors="coerce").dropna()

    if series.empty:
        return {
            f"{prefix}_mean": 0,
            f"{prefix}_std": 0,
            f"{prefix}_min": 0,
            f"{prefix}_max": 0,
            f"{prefix}_median": 0,
            f"{prefix}_q25": 0,
            f"{prefix}_q75": 0,
            f"{prefix}_range": 0,
        }

    return {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_std": float(series.std(ddof=0)),
        f"{prefix}_min": float(series.min()),
        f"{prefix}_max": float(series.max()),
        f"{prefix}_median": float(series.median()),
        f"{prefix}_q25": float(series.quantile(0.25)),
        f"{prefix}_q75": float(series.quantile(0.75)),
        f"{prefix}_range": float(series.max() - series.min()),
    }


def infer_fall_action(chunk, height_drop, speed_max, movement_after, fall_already_detected=False):
    behavior_label = text_value(
        chunk,
        ["behavior_label", "behavior", "action", "activity", "state", "label"],
        default="-",
    )

    scenario = text_value(
        chunk,
        ["scenario", "scene", "phase"],
        default="-",
    )

    description = text_value(
        chunk,
        ["description", "desc", "actual_reason", "reason"],
        default="-",
    )

    text = f"{behavior_label} {scenario} {description}".lower()

    has_fall_word = any(
        word in text
        for word in [
            "fall_forward",
            "fall_backward",
            "fall_left",
            "fall_right",
            "fall_alert",
            "fall alert",
            "낙상",
        ]
    )

    has_post_fall_word = any(
        word in text
        for word in [
            "post_fall",
            "after_fall",
            "fall_no_movement",
            "낙상 후",
            "낙상후",
        ]
    )

    has_real_fall_motion = (
        (height_drop >= 0.45 and speed_max >= 0.60)
        or height_drop >= 0.65
        or (has_fall_word and height_drop >= 0.35 and speed_max >= 0.35)
    )

    if has_post_fall_word and fall_already_detected:
        action = "낙상 후 움직임 적음"
        direction = "바닥에 머문 상태"
        cause_guess = "이전 구간에서 낙상이 감지된 뒤 움직임이 거의 없습니다."

    elif has_post_fall_word:
        action = "움직임 적음"
        direction = "방향 정보 없음"
        cause_guess = "움직임은 적지만 앞선 낙상 감지가 없어 낙상 후 상태로 보지 않습니다."

    elif "fall_forward" in text and has_real_fall_motion:
        action = "걷다가 전방 낙상"
        direction = "전방 방향 추정"
        cause_guess = "이동 중 몸의 높이가 급격히 낮아지고 속도 변화가 크게 나타났습니다."

    elif has_fall_word and has_real_fall_motion:
        action = "낙상 발생"
        direction = "전방 또는 측면 방향 추정"
        cause_guess = "낙상 라벨과 센서의 높이 변화, 속도 변화가 함께 나타났습니다."

    elif "walking" in text or "walk" in text or "보행" in text or "걷" in text:
        if has_real_fall_motion:
            action = "걷다가 낙상"
            direction = "전방 또는 측면 방향 추정"
            cause_guess = "보행 중 높이 변화와 속도 변화가 함께 나타났습니다."
        else:
            action = "보행"
            direction = "이동 방향 정보 부족"
            cause_guess = "보행 상태이지만 낙상으로 볼 만큼의 높이 급감은 없습니다."

    elif "sit" in text or "앉" in text:
        action = "빠른 자세변화"
        direction = "하방 이동"
        cause_guess = "앉는 동작과 낙상이 유사할 수 있으나 낙상 기준을 별도로 확인해야 합니다."

    elif height_drop >= 0.7 and speed_max >= 0.8:
        action = "이동 중 낙상"
        direction = "전방 또는 측면 방향 추정"
        cause_guess = "높이 하강과 속도 변화가 동시에 나타났습니다."

    elif height_drop >= 0.45 and speed_max >= 0.60:
        action = "자세 변화 중 낙상 의심"
        direction = "하방 이동"
        cause_guess = "상체 높이 변화와 속도 변화가 함께 나타났습니다."

    else:
        action = "낙상 가능성 낮음"
        direction = "방향 정보 부족"
        cause_guess = "낙상 기준을 넘는 변화가 충분하지 않습니다."

    causes = []

    if height_drop >= 0.70:
        causes.append("높이가 급격히 낮아졌습니다.")
    elif height_drop >= 0.45:
        causes.append("상체 높이 변화가 크게 나타났습니다.")

    if speed_max >= 1.2:
        causes.append("순간 속도가 매우 크게 증가했습니다.")
    elif speed_max >= 0.8:
        causes.append("중간 수준 이상의 속도 변화가 있습니다.")

    if movement_after <= 0.2 and has_real_fall_motion:
        causes.append("동작 이후 움직임이 거의 없어 낙상 후 움직임 감소로 볼 수 있습니다.")
    elif movement_after <= 0.5 and has_real_fall_motion:
        causes.append("동작 이후 이동이 줄었습니다.")

    if description != "-":
        causes.append(description)

    if not causes:
        causes.append("낙상 기준을 넘는 변화는 크지 않습니다.")

    return action, direction, cause_guess, " ".join(causes), scenario, description


def extract_fall_features_for_model(chunk: pd.DataFrame):
    load_fall_model()

    features = {}

    frame_col = find_col(chunk, ["frame", "Frame", "frame_id", "frameId"])
    det_col = find_col(chunk, ["DetObj#", "detobj", "detobj#", "detObj", "point_count"])
    x_col = find_col(chunk, ["x"])
    y_col = find_col(chunk, ["y"])
    z_col = find_col(chunk, ["z"])
    v_col = find_col(chunk, ["v", "velocity", "speed"])
    snr_col = find_col(chunk, ["snr", "SNR"])
    noise_col = find_col(chunk, ["noise", "Noise"])

    features["total_points"] = int(len(chunk))

    if frame_col:
        frames = pd.to_numeric(chunk[frame_col], errors="coerce").dropna()
        features["frame_count"] = int(frames.nunique()) if not frames.empty else 0
    else:
        features["frame_count"] = int(len(chunk))

    det_series = (
        pd.to_numeric(chunk[det_col], errors="coerce")
        if det_col
        else pd.Series([len(chunk)])
    )
    features.update(stat_features(det_series, "detobj"))

    for col, name in [
        (x_col, "x"),
        (y_col, "y"),
        (z_col, "z"),
        (v_col, "v"),
        (snr_col, "snr"),
        (noise_col, "noise"),
    ]:
        if col:
            features.update(stat_features(chunk[col], name))
        else:
            features.update(stat_features(pd.Series(dtype=float), name))

    x = pd.to_numeric(chunk[x_col], errors="coerce") if x_col else pd.Series(dtype=float)
    y = pd.to_numeric(chunk[y_col], errors="coerce") if y_col else pd.Series(dtype=float)
    z = pd.to_numeric(chunk[z_col], errors="coerce") if z_col else pd.Series(dtype=float)
    v = pd.to_numeric(chunk[v_col], errors="coerce") if v_col else pd.Series(dtype=float)

    abs_v = v.abs() if not v.empty else pd.Series(dtype=float)
    features.update(stat_features(abs_v, "abs_v"))

    if not x.empty and not y.empty and not z.empty:
        radial_dist = np.sqrt(
            (x.fillna(0) ** 2)
            + (y.fillna(0) ** 2)
            + (z.fillna(0) ** 2)
        )
    else:
        radial_dist = pd.Series(dtype=float)

    features.update(stat_features(radial_dist, "radial_dist"))

    centers = center_points(chunk)

    if len(centers) >= 2:
        diffs = centers[["x", "y", "z"]].diff().dropna()
        center_move = np.sqrt((diffs ** 2).sum(axis=1))
        z_center = centers["z"].reset_index(drop=True)
        z_diff = z_center.diff().dropna()
        abs_z_diff = z_diff.abs()

        features.update(stat_features(pd.Series(center_move), "center_move"))
        features.update(stat_features(pd.Series(z_diff), "z_diff"))
        features.update(stat_features(pd.Series(abs_z_diff), "abs_z_diff"))

        features["z_center_drop"] = float(z_center.max() - z_center.min())
        features["z_center_first_to_min_drop"] = float(z_center.iloc[0] - z_center.min())
        features["z_center_peak_to_last_drop"] = float(z_center.max() - z_center.iloc[-1])
        features["z_center_last_minus_first"] = float(z_center.iloc[-1] - z_center.iloc[0])

        tail_start = int(len(centers) * 0.8)
        tail_centers = centers.iloc[tail_start:]
        tail_move_start = int(len(center_move) * 0.8)
        tail_move = center_move.iloc[tail_move_start:]

        features["tail_movement_mean"] = float(tail_move.mean()) if len(tail_move) else 0
        features["tail_movement_max"] = float(tail_move.max()) if len(tail_move) else 0
        features["tail_z_mean"] = float(tail_centers["z"].mean()) if len(tail_centers) else 0
        features["tail_z_min"] = float(tail_centers["z"].min()) if len(tail_centers) else 0
        features["tail_z_max"] = float(tail_centers["z"].max()) if len(tail_centers) else 0

    else:
        features.update(stat_features(pd.Series(dtype=float), "center_move"))
        features.update(stat_features(pd.Series(dtype=float), "z_diff"))
        features.update(stat_features(pd.Series(dtype=float), "abs_z_diff"))

        features["z_center_drop"] = 0
        features["z_center_first_to_min_drop"] = 0
        features["z_center_peak_to_last_drop"] = 0
        features["z_center_last_minus_first"] = 0
        features["tail_movement_mean"] = 0
        features["tail_movement_max"] = 0
        features["tail_z_mean"] = 0
        features["tail_z_min"] = 0
        features["tail_z_max"] = 0

    speed_max = float(abs_v.max()) if not abs_v.empty else 0
    height_drop = float(z.max() - z.min()) if not z.empty else 0
    movement_after = movement_after_value(chunk)

    features["speed_max"] = speed_max
    features["height_drop"] = height_drop
    features["movement_after"] = movement_after

    rule_score = 0

    if height_drop >= 0.70:
        rule_score += 45
    elif height_drop >= 0.55:
        rule_score += 36
    elif height_drop >= 0.40:
        rule_score += 24

    if speed_max >= 1.20:
        rule_score += 35
    elif speed_max >= 0.90:
        rule_score += 28
    elif speed_max >= 0.60:
        rule_score += 16

    if movement_after <= 0.20 and height_drop >= 0.40:
        rule_score += 20
    elif movement_after <= 0.40 and height_drop >= 0.40:
        rule_score += 13

    features["rule_fall_score"] = int(clamp(rule_score))
    features["rule_fall_candidate"] = 1 if rule_score >= 50 else 0

    row = {col: features.get(col, 0) for col in fall_feature_columns}
    input_df = pd.DataFrame([row], columns=fall_feature_columns)
    input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return input_df, features


def get_fall_probability(model_obj, input_df):
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(input_df)[0]

        if hasattr(model_obj, "classes_"):
            classes = list(model_obj.classes_)

            if 1 in classes:
                fall_idx = classes.index(1)
            elif "Fall" in classes:
                fall_idx = classes.index("Fall")
            else:
                fall_idx = min(1, len(proba) - 1)
        else:
            fall_idx = min(1, len(proba) - 1)

        return float(proba[fall_idx])

    pred = model_obj.predict(input_df)[0]

    if str(pred).lower() in ["1", "fall", "fall alert"]:
        return 1.0

    return 0.0


def predict_fall_chunk(chunk: pd.DataFrame, fall_already_detected=False):
    if chunk.empty:
        return {
            "module": "fall",
            "title": "낙상 감지",
            "state": "대기",
            "level": "idle",
            "risk_score": 0,
            "reason": "해당 구간의 낙상 데이터가 없습니다.",
            "fall_action": "-",
            "fall_direction": "-",
            "fall_cause": "-",
            "cause_guess": "-",
            "scenario": "-",
            "description": "-",
            "features": {
                "fall_prob": 0,
                "speed_max": 0,
                "speed_mean": 0,
                "height_drop": 0,
                "movement_after": 0,
                "z_mean": 0,
                "z_min": 0,
                "z_max": 0,
                "model_mode": "waiting",
            },
        }

    try:
        input_df, extracted = extract_fall_features_for_model(chunk)
        fall_prob = get_fall_probability(fall_model, input_df)
        risk_score = int(clamp(round(fall_prob * 100)))

        speed_max = float(extracted.get("speed_max", 0))
        height_drop = float(extracted.get("height_drop", 0))
        movement_after = float(extracted.get("movement_after", 0))

        z = numeric_series(chunk, ["z", "height"])
        v = numeric_series(chunk, ["v", "velocity", "speed"])

        z_mean = float(z.mean()) if not z.empty else 0.0
        z_min = float(z.min()) if not z.empty else 0.0
        z_max = float(z.max()) if not z.empty else 0.0
        speed_mean = float(v.abs().mean()) if not v.empty else 0.0

        fall_action, fall_direction, cause_guess, fall_cause, scenario, description = infer_fall_action(
            chunk,
            height_drop,
            speed_max,
            movement_after,
            fall_already_detected,
        )

        if fall_prob >= fall_threshold:
            state = "Fall Alert"
            level = "danger"
            reason = (
                f"낙상 RF/SMOTE 모델이 낙상으로 예측했습니다. "
                f"예측확률 {risk_score}%, 기준 {round(fall_threshold * 100)}%."
            )
        elif risk_score >= 40:
            state = "주의"
            level = "warning"
            reason = (
                f"낙상 모델 확률은 기준 미만이지만 움직임 변화가 있어 주의로 표시합니다. "
                f"예측확률 {risk_score}%."
            )
        else:
            state = "Normal"
            level = "normal"
            reason = f"낙상 모델 기준 미만입니다. 예측확률 {risk_score}%."

        return {
            "module": "fall",
            "title": "낙상 감지",
            "state": state,
            "level": level,
            "risk_score": risk_score,
            "reason": reason,
            "fall_action": fall_action,
            "fall_direction": fall_direction,
            "fall_cause": fall_cause,
            "cause_guess": cause_guess,
            "scenario": scenario,
            "description": description,
            "features": {
                "fall_prob": round(fall_prob, 4),
                "speed_max": round(speed_max, 4),
                "speed_mean": round(speed_mean, 4),
                "height_drop": round(height_drop, 4),
                "movement_after": round(movement_after, 4),
                "z_mean": round(z_mean, 4),
                "z_min": round(z_min, 4),
                "z_max": round(z_max, 4),
                "model_mode": "mmwave_rf_smote_model",
                "threshold": fall_threshold,
                "feature_count": len(fall_feature_columns),
            },
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "module": "fall",
            "title": "낙상 감지",
            "state": "모델 오류",
            "level": "warning",
            "risk_score": 0,
            "reason": f"낙상 모델 예측 중 오류가 발생했습니다: {e}",
            "fall_action": "-",
            "fall_direction": "-",
            "fall_cause": "-",
            "cause_guess": "-",
            "scenario": "-",
            "description": "-",
            "features": {
                "fall_prob": 0,
                "speed_max": 0,
                "speed_mean": 0,
                "height_drop": 0,
                "movement_after": 0,
                "z_mean": 0,
                "z_min": 0,
                "z_max": 0,
                "model_mode": "error",
                "error": str(e),
            },
        }


# =========================================================
# 이상행동 분석
# =========================================================

def normalize_abnormal_state(value):
    text = clean_text(value)

    if not is_valid_text(text):
        return "기타"

    if text in ABNORMAL_ALLOWED_STATES:
        return text

    compact = compact_text(text)

    if compact in ABNORMAL_NUMERIC_STATE_MAP:
        return ABNORMAL_NUMERIC_STATE_MAP[compact]

    for state, aliases in ABNORMAL_STATE_ALIASES.items():
        for alias in aliases:
            if compact_text(alias) == compact:
                return state

    for state, aliases in ABNORMAL_STATE_ALIASES.items():
        for alias in aliases:
            key = compact_text(alias)

            if key and key in compact:
                return state

    return "기타"


def risk_score_by_abnormal_state(state):
    state = normalize_abnormal_state(state)
    return ABNORMAL_RISK_SCORE_MAP.get(state, 18)


def abnormal_level_by_state(state):
    state = normalize_abnormal_state(state)

    if state == "위험":
        return "danger"

    if state == "주의":
        return "warning"

    return "normal"


def get_recent_warning_count(limit=2):
    count = 0

    for state in reversed(recent_abnormal_states[-limit:]):
        if state == "주의":
            count += 1
        else:
            break

    return count


def guardian_alert_by_abnormal_state(state):
    state = normalize_abnormal_state(state)

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


def safe_inverse_transform(encoder, value):
    if encoder is None:
        return str(value)

    try:
        return str(encoder.inverse_transform([value])[0])
    except Exception:
        return str(value)


def find_eldercare_feature_col(chunk: pd.DataFrame, feature_name: str):
    if chunk is None or chunk.empty:
        return None

    aliases = ELDERCARE_FEATURE_ALIASES.get(feature_name, [feature_name])
    col = find_col(chunk, aliases)

    if col is not None:
        return col

    feature_key = normalize_column_name(feature_name)

    for df_col in chunk.columns:
        col_key = normalize_column_name(df_col)

        if col_key == feature_key:
            return df_col

    return None


def get_eldercare_metric(chunk: pd.DataFrame, feature_name: str):
    aliases = ELDERCARE_FEATURE_ALIASES.get(feature_name, [feature_name])
    return mean_value(chunk, aliases)


def get_text_blob_from_chunk(chunk: pd.DataFrame):
    if chunk is None or chunk.empty:
        return ""

    parts = []

    for col_name in ABNORMAL_TEXT_COLUMN_CANDIDATES:
        col = find_col(chunk, [col_name])

        if col is None:
            continue

        values = chunk[col].dropna().astype(str)

        if values.empty:
            continue

        value = values.iloc[-1].replace("\ufeff", "").strip()

        if value and value.lower() not in ["nan", "none", "null", "-"]:
            parts.append(value)

    return " ".join(parts)


def text_contains_any(text: str, keywords):
    compact = compact_text(text)

    for keyword in keywords:
        key = compact_text(keyword)

        if key and key in compact:
            return True

    return False


def infer_state_from_text(text: str):
    """
    Reason/Activity/Label 같은 텍스트 컬럼에서 생활 상태를 먼저 복원한다.
    '주의'라는 라벨이 들어있어도 설명이 '일상적인 앉기 동작'이면 기타로 내려보내기 위해 사용한다.
    """
    if not is_valid_text(text):
        return None, "", 0

    danger_words = [
        "위험", "응급", "emergency", "critical", "danger", "button", "shout", "비명", "고함",
    ]
    warning_words = [
        "주의", "배회", "wandering", "무활동", "inactive", "inactivity", "장시간", "longstay",
        "이상", "abnormal", "불안", "낙상위험", "도움", "확인필요",
    ]
    outing_words = [
        "외출", "outside", "outing", "outdoor", "goout", "go out", "leaving", "귀가", "이동중",
    ]
    meal_words = [
        "식사", "meal", "eating", "eat", "breakfast", "lunch", "dinner", "음식", "조리", "주방",
    ]
    sleep_words = [
        "수면", "sleep", "sleeping", "취침", "기상전", "침대", "bed", "lying", "lie", "휴식", "rest",
    ]
    normal_words = [
        "기타", "정상", "normal", "일상", "ordinary", "daily", "특이사항없음", "문제없음",
        "앉기", "앉음", "sit", "sitting", "자세변화", "안정", "stable", "활동없음아님",
    ]

    if text_contains_any(text, danger_words):
        return "위험", "텍스트 컬럼에서 위험 상황 표현이 감지되었습니다.", 95

    # 생활 상태는 '주의'보다 먼저 본다. 그래야 '일상적인 앉기 동작'이 주의로 고정되지 않는다.
    if text_contains_any(text, sleep_words):
        return "수면", "텍스트 컬럼에서 수면 상태가 확인되었습니다.", 80

    if text_contains_any(text, meal_words):
        return "식사", "텍스트 컬럼에서 식사 상태가 확인되었습니다.", 80

    if text_contains_any(text, outing_words):
        return "외출", "텍스트 컬럼에서 외출 상태가 확인되었습니다.", 80

    if text_contains_any(text, normal_words):
        return "기타", "텍스트 컬럼에서 일상/정상 활동으로 확인되었습니다.", 75

    if text_contains_any(text, warning_words):
        return "주의", "텍스트 컬럼에서 주의가 필요한 이상행동 표현이 감지되었습니다.", 80

    return None, "", 0


def infer_abnormal_state_by_rules(chunk: pd.DataFrame):
    """
    이상행동 최종 상태 보정 규칙.
    - 텍스트 컬럼(Reason/Activity/Label)에서 수면/식사/외출/정상 활동을 먼저 복원
    - 응급버튼/고함/바이탈 위험은 위험/주의로 올림
    - 그 외에는 기타로 둠
    """
    if chunk is None or chunk.empty:
        return "기타", "데이터가 없어 기타로 처리했습니다.", 0

    text_blob = get_text_blob_from_chunk(chunk)
    text_state, text_reason, text_confidence = infer_state_from_text(text_blob)

    # 텍스트에서 위험이 잡힌 경우는 바로 인정한다.
    if text_state == "위험":
        return text_state, text_reason, text_confidence

    button = get_eldercare_metric(chunk, "Button")
    shout = get_eldercare_metric(chunk, "Shout")
    heart_rate = get_eldercare_metric(chunk, "HeartRate")
    breath_rate = get_eldercare_metric(chunk, "BreathRate")
    spo2 = get_eldercare_metric(chunk, "SPO2")
    skin_temp = get_eldercare_metric(chunk, "SkinTemperature")
    sleep_phase = get_eldercare_metric(chunk, "SleepPhase")
    sleep_score = get_eldercare_metric(chunk, "SleepScore")
    walking_steps = get_eldercare_metric(chunk, "WalkingSteps")
    stress_index = get_eldercare_metric(chunk, "StressIndex")
    activity_intensity = get_eldercare_metric(chunk, "ActivityIntensity")
    caloric = get_eldercare_metric(chunk, "CaloricExpenditure")
    activity_ir = get_eldercare_metric(chunk, "Activity_IR")
    illuminance = get_eldercare_metric(chunk, "Illuminance")

    reasons = []

    if button is not None and button >= 1:
        return "위험", "응급버튼이 감지되어 위험으로 판단했습니다.", 100

    if shout is not None and shout >= 1:
        return "위험", "응급음성이 감지되어 위험으로 판단했습니다.", 100

    danger = False

    if heart_rate is not None and (heart_rate >= 125 or heart_rate <= 42):
        danger = True
        reasons.append("심박이 위험 범위입니다.")

    if breath_rate is not None and (breath_rate >= 32 or breath_rate <= 7):
        danger = True
        reasons.append("호흡수가 위험 범위입니다.")

    if spo2 is not None and 0 < spo2 < 90:
        danger = True
        reasons.append("산소포화도가 위험 범위입니다.")

    if skin_temp is not None and (skin_temp >= 38.3 or skin_temp <= 34.0):
        danger = True
        reasons.append("피부온도가 위험 범위입니다.")

    if stress_index is not None and stress_index >= 90:
        danger = True
        reasons.append("스트레스 지수가 매우 높습니다.")

    if danger:
        return "위험", " ".join(reasons), 90

    # 텍스트에서 생활 상태가 잡힌 경우는 정상 범위 바이탈보다 우선한다.
    if text_state in ["수면", "식사", "외출", "기타"]:
        return text_state, text_reason, text_confidence

    warning = False

    if heart_rate is not None and (heart_rate >= 110 or heart_rate <= 50):
        warning = True
        reasons.append("심박 변화가 주의 범위입니다.")

    if breath_rate is not None and (breath_rate >= 25 or breath_rate <= 10):
        warning = True
        reasons.append("호흡수 변화가 주의 범위입니다.")

    if spo2 is not None and 0 < spo2 < 94:
        warning = True
        reasons.append("산소포화도가 주의 범위입니다.")

    if skin_temp is not None and (skin_temp >= 37.8 or skin_temp <= 35.0):
        warning = True
        reasons.append("피부온도가 주의 범위입니다.")

    if stress_index is not None and stress_index >= 75:
        warning = True
        reasons.append("스트레스 지수가 높습니다.")

    if warning:
        return "주의", " ".join(reasons), 70

    # 위험/주의가 아니면 생활 상태를 먼저 분리한다.
    if sleep_phase is not None and sleep_phase > 0:
        return "수면", "수면 단계 값이 감지되어 수면 상태로 판단했습니다.", 60

    if sleep_score is not None and sleep_score > 0:
        return "수면", "수면 점수 값이 감지되어 수면 상태로 판단했습니다.", 55

    if walking_steps is not None and walking_steps >= 80:
        return "외출", "걸음 수가 높아 외출 또는 이동 상태로 판단했습니다.", 55

    if activity_intensity is not None and activity_intensity >= 7:
        return "외출", "활동 강도가 높아 외출 또는 이동 상태로 판단했습니다.", 50

    if caloric is not None and caloric > 0 and (walking_steps is None or walking_steps < 80):
        return "식사", "칼로리 소모 값이 감지되어 식사/일상 활동 상태로 판단했습니다.", 45

    if activity_ir is not None and activity_ir <= 0 and illuminance is not None and illuminance <= 5:
        return "수면", "움직임과 조도가 낮아 수면 또는 휴식 상태로 판단했습니다.", 45

    if text_state == "주의":
        return "주의", text_reason, text_confidence

    return "기타", "위험/주의 기준을 넘는 센서 변화가 없어 기타 상태로 판단했습니다.", 30


def build_eldercare_input_dataframe(chunk: pd.DataFrame):
    load_eldercare_models()

    row = chunk.iloc[-1].to_dict() if chunk is not None and not chunk.empty else {}

    input_data = {}
    matched_features = []
    missing_features = []

    for feature in eldercare_features:
        col = find_eldercare_feature_col(chunk, feature)

        if col is not None:
            value = row.get(col, 0)
            matched_features.append(
                {
                    "feature": feature,
                    "csv_column": col,
                }
            )
        else:
            value = 0
            missing_features.append(feature)

        input_data[feature] = value

    input_df = pd.DataFrame([input_data], columns=eldercare_features)

    for col in eldercare_features:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_df = input_df.fillna(0)

    meta = {
        "matched_feature_count": len(matched_features),
        "missing_feature_count": len(missing_features),
        "non_zero_feature_count": int((input_df.iloc[0] != 0).sum()),
    }

    return input_df, meta


def get_csv_estimation_value(chunk: pd.DataFrame):
    """
    CSV 안에 상태 라벨 컬럼이 있으면 그 값을 최우선으로 사용한다.
    A0003 계열 파일마다 Estimation/state/label/activity/상태 이름이 달라서
    컬럼명을 넓게 스캔한다.
    """
    if chunk is None or chunk.empty:
        return "-"

    # 1) 명시 후보 컬럼 우선
    direct = text_value(chunk, ABNORMAL_ESTIMATION_COLUMN_CANDIDATES, default="-")
    if is_valid_text(direct):
        return direct

    # 2) 컬럼명에 상태/라벨/행동 의미가 들어간 것 전체 스캔
    name_keywords = [
        "estimation", "estimate", "state", "status", "label", "class", "target",
        "activity", "behavior", "action", "result", "prediction", "pred",
        "상태", "라벨", "행동", "활동", "판정", "결과", "분류",
    ]

    row = chunk.iloc[-1].to_dict()

    for col in chunk.columns:
        col_key = compact_text(col)
        if not any(compact_text(keyword) in col_key for keyword in name_keywords):
            continue

        value = row.get(col, "-")
        if not is_valid_text(value):
            continue

        normalized = normalize_abnormal_state(value)
        if normalized in ABNORMAL_ALLOWED_STATES:
            return str(value).strip()

    # 3) Reason/Description 안에 수면/식사/외출/위험/주의 같은 단어가 있으면 상태로 복원
    reason_text = get_csv_reason_value(chunk)
    reason_state, _, _ = infer_state_from_text(reason_text)
    if reason_state in ABNORMAL_ALLOWED_STATES:
        return reason_state

    return "-"


def get_csv_reason_value(chunk: pd.DataFrame):
    return text_value(chunk, ABNORMAL_REASON_COLUMN_CANDIDATES, default="-")


def get_abnormal_metric(chunk: pd.DataFrame, metric_key: str):
    aliases = ABNORMAL_VITAL_ALIASES.get(metric_key, [metric_key])
    return mean_value(chunk, aliases)


def get_model_probability(model_obj, input_df, pred_encoded):
    if not hasattr(model_obj, "predict_proba"):
        return None

    try:
        proba = model_obj.predict_proba(input_df)[0]

        if hasattr(model_obj, "classes_"):
            classes = list(model_obj.classes_)

            if pred_encoded in classes:
                idx = classes.index(pred_encoded)
            else:
                idx = int(np.argmax(proba))
        else:
            idx = int(np.argmax(proba))

        return float(proba[idx])

    except Exception:
        return None


def predict_abnormal_model(input_df):
    pred_encoded = eldercare_model.predict(input_df)[0]
    raw_model_state = safe_inverse_transform(eldercare_label_encoder, pred_encoded)
    model_state = normalize_abnormal_state(raw_model_state)
    model_probability = get_model_probability(eldercare_model, input_df, pred_encoded)

    model_reason = "-"

    try:
        reason_encoded = eldercare_reason_model.predict(input_df)[0]
        model_reason = safe_inverse_transform(eldercare_reason_encoder, reason_encoded)
    except Exception:
        model_reason = "-"

    return raw_model_state, model_state, model_probability, model_reason


def choose_abnormal_final_state(
    csv_estimation,
    csv_reason,
    rule_state,
    rule_confidence,
    model_state,
    model_probability,
    input_meta=None,
):
    """
    v4 결정 방식:
    1. CSV에 상태 라벨이 있으면 그대로 표시한다.
       - 이게 현재 통합 시나리오 CSV의 정답 흐름이다.
       - 이전처럼 주의/기타로 임의 보정하지 않는다.
    2. CSV 라벨이 없으면 Reason 텍스트를 본다.
    3. 그래도 없으면 센서 rule, 마지막으로 모델을 사용한다.
    """
    input_meta = input_meta or {}

    normalized_csv_state = normalize_abnormal_state(csv_estimation)
    normalized_rule_state = normalize_abnormal_state(rule_state)
    normalized_model_state = normalize_abnormal_state(model_state)
    model_prob = 0 if model_probability is None else float(model_probability)

    reason_state, _, reason_confidence = infer_state_from_text(csv_reason)

    # 응급 버튼/고함/위험 바이탈은 CSV보다 우선
    if normalized_rule_state == "위험" and rule_confidence >= 90:
        return "위험", "sensor_rule_danger"

    # CSV 라벨이 있으면 그대로 사용. 주의도 주의, 기타도 기타로 그대로 간다.
    # 이것이 600행짜리 시나리오 CSV의 라벨 흐름을 살리는 핵심이다.
    if is_valid_text(csv_estimation) and normalized_csv_state in ABNORMAL_ALLOWED_STATES:
        return normalized_csv_state, "csv_label_strict"

    # Reason/설명 컬럼에 생활상태 단어가 있으면 사용
    if reason_state in ABNORMAL_ALLOWED_STATES and reason_confidence >= 50:
        return reason_state, "csv_reason_text"

    # CSV 라벨이 없을 때만 센서 rule 사용
    if normalized_rule_state in ["위험", "주의"] and rule_confidence >= 65:
        return normalized_rule_state, "sensor_rule"

    if normalized_rule_state in ["수면", "식사", "외출", "기타"] and rule_confidence >= 45:
        return normalized_rule_state, "sensor_rule_lifestyle"

    # 모델 입력이 어느 정도 맞으면 모델 사용
    matched_count = int(input_meta.get("matched_feature_count", 0) or 0)
    non_zero_count = int(input_meta.get("non_zero_feature_count", 0) or 0)
    model_input_reliable = matched_count >= 10 or non_zero_count >= 5

    if model_input_reliable and normalized_model_state in ABNORMAL_ALLOWED_STATES:
        if model_prob is None or model_prob >= 0.25:
            return normalized_model_state, "model_reliable"

    return "기타", "default_etc"


def get_strict_estimation_value(chunk: pd.DataFrame):
    """
    v5 핵심:
    이상행동 상태는 CSV의 Estimation 컬럼을 그대로 읽는다.
    모델, rule, Reason 보정으로 Estimation을 덮어쓰지 않는다.
    """
    if chunk is None or chunk.empty:
        return "-"

    # Estimation 컬럼만 1순위로 정확히 찾는다.
    col = find_col(chunk, ["Estimation", "estimation", "ESTIMATION", "Estimate", "estimate"])

    # 혹시 사용자가 컬럼명을 한글로 바꾼 경우만 보조 허용
    if col is None:
        col = find_col(chunk, ["상태", "라벨", "행동상태", "이상행동", "분류", "판정", "결과"])

    if col is None:
        return "-"

    value = chunk.iloc[-1].get(col, "-")

    if not is_valid_text(value):
        return "-"

    return clean_text(value)


def predict_abnormal_chunk(chunk: pd.DataFrame):
    global recent_abnormal_states

    if chunk.empty:
        return {
            "module": "abnormal",
            "title": "이상행동",
            "state": "대기",
            "level": "idle",
            "risk_score": 0,
            "reason": "해당 구간의 이상행동 데이터가 없습니다.",
            "behavior": "-",
            "abnormal_type": "-",
            "detail": "-",
            "guardian_alert": False,
            "guardian_status": "대기",
            "guardian_message": "데이터 대기 중입니다.",
            "heart_rate": None,
            "respiratory_rate": None,
            "temperature": None,
            "stress_score": None,
            "features": {
                "code_version": CODE_VERSION,
                "model_mode": "waiting",
                "state_source": "waiting",
                "csv_estimation": "-",
                "csv_state": "-",
            },
        }

    raw_estimation = get_strict_estimation_value(chunk)
    csv_reason = get_csv_reason_value(chunk)

    if is_valid_text(raw_estimation):
        final_state = normalize_abnormal_state(raw_estimation)
        state_source = "ESTIMATION_COLUMN_ONLY"
        final_reason = (
            csv_reason
            if is_valid_text(csv_reason)
            else f"CSV Estimation 컬럼 값({raw_estimation})을 그대로 표시했습니다."
        )
    else:
        # Estimation이 정말 없을 때만 fallback 사용
        reason_state, reason_text, reason_confidence = infer_state_from_text(csv_reason)
        if reason_state in ABNORMAL_ALLOWED_STATES and reason_confidence >= 50:
            final_state = reason_state
            state_source = "reason_fallback"
            final_reason = csv_reason if is_valid_text(csv_reason) else reason_text
        else:
            final_state, rule_reason, _ = infer_abnormal_state_by_rules(chunk)
            final_state = normalize_abnormal_state(final_state)
            state_source = "rule_fallback_no_estimation"
            final_reason = rule_reason

    heart_rate = round_optional(get_abnormal_metric(chunk, "HeartRate"), 1)
    respiratory_rate = round_optional(get_abnormal_metric(chunk, "BreathRate"), 1)
    temperature = round_optional(get_abnormal_metric(chunk, "SkinTemperature"), 1)
    stress_score = round_optional(get_abnormal_metric(chunk, "StressIndex"), 1)

    risk_score = risk_score_by_abnormal_state(final_state)
    level = abnormal_level_by_state(final_state)
    guardian = guardian_alert_by_abnormal_state(final_state)

    recent_abnormal_states.append(final_state)
    recent_abnormal_states = recent_abnormal_states[-50:]

    return {
        "module": "abnormal",
        "title": "이상행동",
        "state": final_state,
        "level": level,
        "risk_score": risk_score,
        "reason": final_reason,
        "behavior": final_state,
        "abnormal_type": final_state,
        "detail": final_reason,
        **guardian,
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "stress_score": stress_score,
        "features": {
            "code_version": CODE_VERSION,
            "model_mode": "estimation_only_no_model_override",
            "state_source": state_source,
            "raw_estimation": raw_estimation,
            "csv_estimation": raw_estimation,
            "csv_state": final_state,
            "final_state": final_state,
            "final_level": level,
            "csv_reason": csv_reason,
        },
    }


# =========================================================
# 바이탈 분석
# =========================================================

def extract_preprocessed_vital_matrix(chunk: pd.DataFrame):
    if chunk is None or chunk.empty:
        return None, None

    selected_cols = []

    for feature_name in VITAL_FEATURE_NAMES:
        col = find_col(
            chunk,
            FEATURE_COLUMN_ALIASES.get(feature_name, [feature_name]),
        )

        if col is None:
            return None, None

        selected_cols.append(col)

    feature_df = chunk[selected_cols].copy()
    feature_df.columns = VITAL_FEATURE_NAMES
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if feature_df.empty:
        return None, None

    feature_df = feature_df.fillna(0.0)
    matrix = feature_df.to_numpy(dtype=float)

    feature_mean = {
        name: round(float(feature_df[name].mean()), 6)
        for name in VITAL_FEATURE_NAMES
    }

    return matrix, feature_mean


def extract_raw_vital_signal(chunk: pd.DataFrame):
    signal_col = find_col(chunk, RAW_VITAL_SIGNAL_COLS)

    if signal_col is None:
        return None

    signal = pd.to_numeric(chunk[signal_col], errors="coerce").dropna()

    if signal.empty:
        return None

    return signal.to_numpy(dtype=float)


def get_raw_vital_time_info(chunk: pd.DataFrame):
    time_col = find_col(chunk, RAW_VITAL_TIME_COLS)

    if time_col is None:
        return {
            "time_start": None,
            "time_end": None,
            "sample_count": int(len(chunk)),
        }

    times = pd.to_numeric(chunk[time_col], errors="coerce").dropna()

    if times.empty:
        return {
            "time_start": None,
            "time_end": None,
            "sample_count": int(len(chunk)),
        }

    return {
        "time_start": round(float(times.min()), 3),
        "time_end": round(float(times.max()), 3),
        "sample_count": int(len(times)),
    }


def compute_raw_vital_features(signal):
    if signal is None or len(signal) == 0:
        return {name: 0.0 for name in VITAL_FEATURE_NAMES}

    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]

    if len(signal) == 0:
        return {name: 0.0 for name in VITAL_FEATURE_NAMES}

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    peak_to_peak = float(np.max(signal) - np.min(signal))

    if len(signal) >= 2:
        signs = np.sign(signal)
        zero_crossings = int(np.sum(np.diff(signs) != 0))
        fft_values = np.abs(np.fft.rfft(signal))
        fft_mean = float(np.mean(fft_values))
        fft_max = float(np.max(fft_values))
        fft_std = float(np.std(fft_values))
    else:
        zero_crossings = 0
        fft_mean = 0.0
        fft_max = 0.0
        fft_std = 0.0

    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "peak_to_peak": round(peak_to_peak, 6),
        "zero_crossings": round(float(zero_crossings), 6),
        "fft_mean": round(fft_mean, 6),
        "fft_max": round(fft_max, 6),
        "fft_std": round(fft_std, 6),
    }


def feature_dict_to_matrix(features_dict: dict):
    return np.array(
        [[float(features_dict.get(name, 0.0)) for name in VITAL_FEATURE_NAMES]],
        dtype=float,
    )


def get_vital_condition(chunk: pd.DataFrame):
    return text_value(chunk, RAW_VITAL_CONDITION_COLS, default="-")


def fallback_vital_result_from_features(chunk: pd.DataFrame, features_dict: dict, reason_prefix: str, model_mode: str):
    condition = get_vital_condition(chunk)
    condition_text = str(condition or "").lower()

    peak_to_peak = float(features_dict.get("peak_to_peak", 0))
    std = float(features_dict.get("std", 0))
    fft_max = float(features_dict.get("fft_max", 0))

    if "apnea" in condition_text or "무호흡" in condition_text:
        state = "호흡 이상"
        level = "warning"
        risk_score = 72
        reason = "Condition 값이 Apnea로 표시되어 호흡 이상 가능성이 있습니다."

    elif "normal" in condition_text or "정상" in condition_text:
        state = "정상"
        level = "normal"
        risk_score = 18
        reason = "Condition 값이 Normal이며 바이탈 신호가 정상 참고 상태입니다."

    elif peak_to_peak >= 2.0 or std >= 0.8:
        state = "주의"
        level = "warning"
        risk_score = 60
        reason = "바이탈 신호의 진폭 변화가 크게 나타났습니다."

    elif fft_max >= 50:
        state = "주의"
        level = "warning"
        risk_score = 55
        reason = "주파수 성분의 최대값이 높게 나타났습니다."

    else:
        state = "정상"
        level = "normal"
        risk_score = 18
        reason = "바이탈 신호가 현재 기준에서 큰 이상을 보이지 않습니다."

    time_info = get_raw_vital_time_info(chunk)

    return {
        "module": "vital",
        "title": "바이탈",
        "state": state,
        "level": level,
        "risk_score": risk_score,
        "reason": f"{reason_prefix} {reason}".strip(),
        "status": level.capitalize(),
        "model_mode": model_mode,
        "condition": condition,
        "time_start": time_info["time_start"],
        "time_end": time_info["time_end"],
        "sample_count": time_info["sample_count"],
        "error_mean": 0,
        "error_max": 0,
        "error_min": 0,
        "threshold": 0,
        "error_ratio": 0,
        "anomaly_ratio": 0,
        "is_anomaly": level in ["danger", "warning"],
        "features": features_dict,
    }


def vital_status_to_level(status: str):
    text = str(status or "").lower()

    if text == "danger" or "위험" in text:
        return "danger"

    if text == "warning" or "주의" in text:
        return "warning"

    if text == "normal" or "정상" in text:
        return "normal"

    return "idle"


def predict_vital_with_autoencoder(chunk: pd.DataFrame, feature_matrix: np.ndarray, feature_mean: dict, model_mode: str, default_reason: str):
    condition = get_vital_condition(chunk)
    time_info = get_raw_vital_time_info(chunk)

    if vital_module is None:
        return fallback_vital_result_from_features(
            chunk,
            feature_mean,
            f"vital_signal.py import 실패: {VITAL_MODULE_IMPORT_ERROR}.",
            f"{model_mode}_fallback",
        )

    if not hasattr(vital_module, "check_model_ready") or not vital_module.check_model_ready():
        return fallback_vital_result_from_features(
            chunk,
            feature_mean,
            "생체신호 AutoEncoder 모델이 로드되지 않아 feature 기준으로 판정합니다.",
            f"{model_mode}_fallback",
        )

    try:
        window_errors = vital_module.predict_errors(feature_matrix)

        segment = vital_module.make_segment_result(
            segment_index=0,
            window_matrix=feature_matrix,
            window_errors=window_errors,
        )

        status = segment.get("status", "Normal")
        state = segment.get("state", "정상")
        level = vital_status_to_level(status)

        return {
            "module": "vital",
            "title": "바이탈",
            "state": state,
            "level": level,
            "risk_score": segment.get("risk_score", 0),
            "reason": segment.get("message", default_reason),
            "status": status,
            "model_mode": model_mode,
            "condition": condition,
            "time_start": time_info["time_start"],
            "time_end": time_info["time_end"],
            "sample_count": time_info["sample_count"],
            "error_mean": segment.get("error_mean", 0),
            "error_max": segment.get("error_max", 0),
            "error_min": segment.get("error_min", 0),
            "threshold": segment.get("threshold", 0),
            "error_ratio": segment.get("error_ratio", 0),
            "anomaly_ratio": segment.get("anomaly_ratio", 0),
            "is_anomaly": segment.get("is_anomaly", False),
            "features": segment.get("features", feature_mean),
        }

    except Exception as e:
        traceback.print_exc()

        return fallback_vital_result_from_features(
            chunk,
            feature_mean,
            f"AutoEncoder 분석 중 오류가 발생해 feature 기준으로 판정합니다: {e}.",
            f"{model_mode}_fallback",
        )


def predict_vital_chunk(chunk: pd.DataFrame):
    if chunk.empty:
        return {
            "module": "vital",
            "title": "호흡",
            "state": "대기",
            "level": "idle",
            "risk_score": 0,
            "reason": "해당 구간의 호흡 데이터가 없습니다.",
            "status": "Idle",
            "model_mode": "waiting",
            "condition": "-",
            "time_start": None,
            "time_end": None,
            "sample_count": 0,
            "error_mean": 0,
            "error_max": 0,
            "error_min": 0,
            "threshold": 0,
            "error_ratio": 0,
            "anomaly_ratio": 0,
            "is_anomaly": False,
            "features": {name: 0.0 for name in VITAL_FEATURE_NAMES},
        }

    feature_matrix, feature_mean = extract_preprocessed_vital_matrix(chunk)

    if feature_matrix is not None:
        return predict_vital_with_autoencoder(
            chunk,
            feature_matrix,
            feature_mean,
            "feature_csv_autoencoder",
            "전처리 완료 호흡 feature CSV를 AutoEncoder로 분석했습니다.",
        )

    signal = extract_raw_vital_signal(chunk)

    if signal is not None:
        raw_features = compute_raw_vital_features(signal)

        return predict_vital_with_autoencoder(
            chunk,
            feature_dict_to_matrix(raw_features),
            raw_features,
            "raw_signal_autoencoder",
            "VitalSignal 원시 신호를 feature로 변환한 뒤 AutoEncoder로 분석했습니다.",
        )

    return {
        "module": "vital",
        "title": "호흡",
        "state": "신호 없음",
        "level": "idle",
        "risk_score": 0,
        "reason": "바이탈 CSV에 전처리 feature 컬럼 또는 원본 VitalSignal 컬럼이 없습니다.",
        "status": "NoSignal",
        "model_mode": "no_vital_feature_or_signal",
        "condition": get_vital_condition(chunk),
        "time_start": None,
        "time_end": None,
        "sample_count": 0,
        "error_mean": 0,
        "error_max": 0,
        "error_min": 0,
        "threshold": 0,
        "error_ratio": 0,
        "anomaly_ratio": 0,
        "is_anomaly": False,
        "features": {name: 0.0 for name in VITAL_FEATURE_NAMES},
    }


# =========================================================
# 종합 / 최종 결론 / 저장
# =========================================================

def to_display_second(raw_second, fallback=1):
    """
    내부 계산 second는 0부터 시작하지만 화면/저장 메시지는 1초 구간부터 보여준다.
    예: cursor=0, second=0.0 -> display_second=1
    """
    try:
        return max(1, int(float(raw_second)) + 1)
    except Exception:
        return fallback


def make_current_overall(fall_result, abnormal_result, vital_result):
    results = [fall_result, abnormal_result]

    if any(item["level"] == "danger" for item in results):
        level = "danger"
        label = "위험"
        message = "현재 구간에서 즉시 확인이 필요한 위험 상황입니다."
    elif any(item["level"] == "warning" for item in results):
        level = "warning"
        label = "주의"
        message = "현재 구간에서 주의 상태가 감지되었습니다."
    elif any(item["level"] == "normal" for item in results):
        level = "normal"
        label = "정상"
        message = "현재 구간은 정상 범위입니다."
    else:
        level = "idle"
        label = "대기"
        message = "CSV를 업로드하고 통합 재생을 시작하세요."

    risk_score = max(item.get("risk_score", 0) for item in results)

    return {
        "level": level,
        "label": label,
        "risk_score": risk_score,
        "message": message,
    }


def make_fall_summary(ordered):
    fall_events = [
        item for item in ordered
        if item.get("fall", {}).get("level") == "danger"
    ]

    if not fall_events:
        return {
            "detected": False,
            "level": "normal",
            "label": "정상",
            "risk_score": 0,
            "second": None,
            "display_second": None,
            "action": "-",
            "direction": "-",
            "cause": "-",
            "cause_guess": "-",
            "message": "낙상 위험 구간이 감지되지 않았습니다.",
        }

    best_fall = max(
        fall_events,
        key=lambda x: x.get("fall", {}).get("risk_score", 0),
    )

    fall = best_fall.get("fall", {})
    raw_second = best_fall.get("second", 0)
    display_second = to_display_second(raw_second)

    return {
        "detected": True,
        "level": "danger",
        "label": "낙상 발생",
        "risk_score": int(fall.get("risk_score", 0)),
        "second": raw_second,
        "display_second": display_second,
        "action": fall.get("fall_action", "-"),
        "direction": fall.get("fall_direction", "-"),
        "cause": fall.get("fall_cause", "-"),
        "cause_guess": fall.get("cause_guess", "-"),
        "message": f"{display_second}초 구간에서 낙상이 감지되었습니다.",
    }


def make_abnormal_summary(ordered):
    abnormal_items = [
        item for item in ordered
        if item.get("abnormal", {}).get("level") in ["normal", "warning", "danger"]
    ]

    latest_abnormal = abnormal_items[-1].get("abnormal", {}) if abnormal_items else {}

    abnormal_events = [
        item for item in ordered
        if item.get("abnormal", {}).get("state") in ["위험", "주의"]
        or item.get("abnormal", {}).get("level") in ["danger", "warning"]
        or int(float(item.get("abnormal", {}).get("risk_score", 0) or 0)) >= 65
    ]

    if not abnormal_events:
        return {
            "detected": False,
            "level": latest_abnormal.get("level", "normal") if latest_abnormal else "normal",
            "label": latest_abnormal.get("state", "정상") if latest_abnormal else "정상",
            "risk_score": int(latest_abnormal.get("risk_score", 0)) if latest_abnormal else 0,
            "second": None,
            "display_second": None,
            "state": latest_abnormal.get("state", "-") if latest_abnormal else "-",
            "reason": latest_abnormal.get("reason", "-") if latest_abnormal else "-",
            "message": "위험 또는 주의 이상행동은 감지되지 않았습니다.",
        }

    best_abnormal = max(
        abnormal_events,
        key=lambda x: x.get("abnormal", {}).get("risk_score", 0),
    )

    abnormal = best_abnormal.get("abnormal", {})
    state = normalize_abnormal_state(abnormal.get("state", "-"))
    raw_second = best_abnormal.get("second", 0)
    display_second = to_display_second(raw_second)

    return {
        "detected": True,
        "level": abnormal.get("level", abnormal_level_by_state(state)),
        "label": state,
        "risk_score": int(abnormal.get("risk_score", risk_score_by_abnormal_state(state))),
        "second": raw_second,
        "display_second": display_second,
        "state": state,
        "reason": abnormal.get("reason", "-"),
        "message": f"{display_second}초 구간에서 {state} 상태가 감지되었습니다.",
    }


def make_final_summary(history):
    if not history:
        return {
            "level": "idle",
            "label": "대기",
            "risk_score": 0,
            "message": "아직 실행된 결과가 없습니다.",
            "fall_detected": False,
            "fall_second": None,
            "fall_display_second": None,
            "fall_action": "-",
            "fall_direction": "-",
            "fall_cause": "-",
            "cause_guess": "-",
            "abnormal_detected": False,
            "abnormal_second": None,
            "abnormal_display_second": None,
            "abnormal_type": "-",
            "vital_detected": False,
            "saved": False,
            "fall_summary": None,
            "abnormal_summary": None,
        }

    ordered = sorted(history, key=lambda x: x.get("second", 0))

    fall_summary = make_fall_summary(ordered)
    abnormal_summary = make_abnormal_summary(ordered)

    vital_events = [
        item for item in ordered
        if item.get("vital", {}).get("level") in ["danger", "warning"]
    ]

    max_risk = max(
        int(item.get("overall", {}).get("risk_score", 0))
        for item in ordered
    )

    if fall_summary["detected"] and abnormal_summary["detected"]:
        if fall_summary["risk_score"] >= abnormal_summary["risk_score"]:
            level = fall_summary["level"]
            label = fall_summary["label"]
            risk_score = fall_summary["risk_score"]
        else:
            level = abnormal_summary["level"]
            label = "이상행동 감지"
            risk_score = abnormal_summary["risk_score"]

        message = (
            f"낙상과 이상행동이 각각 독립적으로 감지되었습니다. "
            f"낙상: {fall_summary['display_second']}초 구간, "
            f"이상행동: {abnormal_summary['display_second']}초 구간 {abnormal_summary['state']}."
        )

    elif fall_summary["detected"]:
        level = fall_summary["level"]
        label = fall_summary["label"]
        risk_score = fall_summary["risk_score"]
        message = fall_summary["message"]

    elif abnormal_summary["detected"]:
        level = abnormal_summary["level"]
        label = "이상행동 감지"
        risk_score = abnormal_summary["risk_score"]
        message = abnormal_summary["message"]

    else:
        level = "normal"
        label = "정상"
        risk_score = max_risk
        message = "전체 구간에서 낙상 또는 위험/주의 이상행동이 감지되지 않았습니다."

    return {
        "level": level,
        "label": label,
        "risk_score": risk_score,
        "message": message,
        "fall_detected": fall_summary["detected"],
        "fall_second": fall_summary["second"],
        "fall_display_second": fall_summary["display_second"],
        "fall_action": fall_summary["action"],
        "fall_direction": fall_summary["direction"],
        "fall_cause": fall_summary["cause"],
        "cause_guess": fall_summary["cause_guess"],
        "abnormal_detected": abnormal_summary["detected"],
        "abnormal_second": abnormal_summary["second"],
        "abnormal_display_second": abnormal_summary["display_second"],
        "abnormal_type": abnormal_summary["state"],
        "vital_detected": len(vital_events) > 0,
        "saved": fall_summary["detected"] or abnormal_summary["detected"],
        "fall_summary": fall_summary,
        "abnormal_summary": abnormal_summary,
    }


def should_save_event(result, final_summary=None):
    """
    DB 저장 기준.
    - 낙상: danger/Fall Alert 저장
    - 이상행동: Estimation 결과가 위험/주의면 저장
    - 프론트에서만 바뀐 값이 아니라 백엔드 abnormal_result 자체 기준으로 저장한다.
    """
    result = result or {}
    fall = result.get("fall", {}) or {}
    abnormal = result.get("abnormal", {}) or {}

    fall_state = clean_text(fall.get("state", ""))
    fall_level = clean_text(fall.get("level", ""))

    abnormal_state = normalize_abnormal_state(abnormal.get("state", ""))
    abnormal_level = clean_text(abnormal.get("level", ""))
    abnormal_risk = int(float(abnormal.get("risk_score", 0) or 0))

    if fall_state == "Fall Alert" or fall_level == "danger":
        return True

    if abnormal_state in ["위험", "주의"]:
        return True

    if abnormal_level in ["danger", "warning"]:
        return True

    if abnormal_risk >= 65:
        return True

    if final_summary:
        if final_summary.get("fall_detected"):
            return True

        if final_summary.get("abnormal_detected"):
            return True

        abnormal_summary = final_summary.get("abnormal_summary") or {}
        summary_state = normalize_abnormal_state(abnormal_summary.get("state", ""))
        summary_level = clean_text(abnormal_summary.get("level", ""))
        summary_risk = int(float(abnormal_summary.get("risk_score", 0) or 0))

        if summary_state in ["위험", "주의"]:
            return True

        if summary_level in ["danger", "warning"]:
            return True

        if summary_risk >= 65:
            return True

    return False


def dataframe_row_count(df):
    if df is None or df.empty:
        return 0
    return int(len(df))


def build_data_profile(fall_df, abnormal_df, vital_df):
    fall_min, fall_max, fall_frame_count = get_frame_range(fall_df)

    abnormal_estimation_col = None
    abnormal_estimation_preview = []

    if abnormal_df is not None and not abnormal_df.empty:
        abnormal_estimation_col = find_col(
            abnormal_df,
            ["Estimation", "estimation", "ESTIMATION", "Estimate", "estimate"],
        )

        if abnormal_estimation_col is not None:
            abnormal_estimation_preview = [
                normalize_abnormal_state(value)
                for value in abnormal_df[abnormal_estimation_col].head(20).tolist()
            ]

    return {
        "fall_rows": dataframe_row_count(fall_df),
        "fall_frame_min": fall_min,
        "fall_frame_max": fall_max,
        "fall_frame_count": fall_frame_count,
        "abnormal_rows": dataframe_row_count(abnormal_df),
        "abnormal_total_seconds": dataframe_row_count(abnormal_df) * ABNORMAL_ROW_SECONDS,
        "abnormal_columns": list(abnormal_df.columns) if abnormal_df is not None else [],
        "abnormal_estimation_column": abnormal_estimation_col,
        "abnormal_estimation_preview": abnormal_estimation_preview,
        "vital_rows": dataframe_row_count(vital_df),
        "policy": "이상행동 CSV는 TimeStamp를 무시하고 1행 = 1초로 계산, 상태는 Estimation 컬럼만 직접 사용",
    }


def save_event_to_db(event_type, payload):
    saved_payload = json_safe({
        "event_type": event_type,
        "saved_at": now_iso(),
        **payload,
    })

    saved_id = None
    db_status = "memory"

    if mongo_collection is not None:
        try:
            result = mongo_collection.insert_one(saved_payload)
            saved_id = str(result.inserted_id)
            db_status = "mongodb"
        except Exception as e:
            saved_payload["db_error"] = str(e)
            db_status = "memory"

    memory_payload = {
        **saved_payload,
        "_id": saved_id or str(uuid4()),
        "db_status": db_status,
    }

    SAVED_LOGS.insert(0, memory_payload)

    if len(SAVED_LOGS) > 200:
        del SAVED_LOGS[200:]

    return {
        "saved": True,
        "db_status": db_status,
        "id": memory_payload["_id"],
    }


def session_status(session_id: str):
    session = SESSIONS.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="통합 시뮬레이션 세션을 찾을 수 없습니다.")

    current_step = int(session["cursor"])
    step_seconds = float(session.get("step_seconds", 1) or 1)
    profile = session.get("profile") or {}
    abnormal_rows = int(profile.get("abnormal_rows", 0) or 0)

    # v4 핵심 잠금:
    # 이상행동 CSV가 있으면 총 시간/현재 시간은 무조건 row 기준이다.
    # 600행이면 600초, 30번째 행이면 30초다.
    if abnormal_rows > 0:
        total_seconds = float(abnormal_rows * ABNORMAL_ROW_SECONDS)
        current_seconds = round(min(current_step, abnormal_rows) * ABNORMAL_ROW_SECONDS, 2)
        step_seconds = float(ABNORMAL_ROW_SECONDS)
    else:
        computed_total_seconds = float(session["total_steps"]) * step_seconds
        total_seconds = computed_total_seconds
        current_seconds = round(current_step * step_seconds, 2)

    abnormal_total_seconds = float(abnormal_rows * ABNORMAL_ROW_SECONDS)

    return {
        "session_id": session_id,
        "code_version": CODE_VERSION,
        "fps": session["fps"],
        "fall_window_frames": session["fall_window_frames"],
        "step_seconds": step_seconds,
        "window_seconds": step_seconds,
        "abnormal_row_seconds": ABNORMAL_ROW_SECONDS,
        "vital_sample_interval_seconds": VITAL_SAMPLE_INTERVAL_SECONDS,
        "current_step": current_step,
        "total_steps": int(session["total_steps"]),
        "current_seconds": current_seconds,
        "total_seconds": round(total_seconds, 2),
        "done": session["cursor"] >= session["total_steps"],
        "files": session["files"],
        "created_at": session["created_at"],
        "profile": profile,
        "abnormal_row_count": int(profile.get("abnormal_rows", 0) or 0),
        "abnormal_total_seconds": round(abnormal_total_seconds, 2),
        "time_policy": "이상행동 CSV는 TimeStamp를 무시하고 1행 = 1초",
    }


# =========================================================
# API
# =========================================================

@router.get("/health")
def health():
    vital_ready = False

    if vital_module is not None and hasattr(vital_module, "check_model_ready"):
        try:
            vital_ready = vital_module.check_model_ready()
        except Exception:
            vital_ready = False

    fall_model_loaded = False
    eldercare_model_loaded = False
    model_errors = {}

    try:
        load_fall_model()
        fall_model_loaded = True
    except Exception as e:
        model_errors["fall"] = str(e)

    try:
        load_eldercare_models()
        eldercare_model_loaded = True
    except Exception as e:
        model_errors["eldercare"] = str(e)

    return {
        "status": "ok",
        "service": "integrated-dashboard",
        "code_version": CODE_VERSION,
        "time": now_iso(),
        "db_connected": mongo_collection is not None,
        "db_error": mongo_error,

        "fall_model_loaded": fall_model_loaded,
        "fall_model_path": str(FALL_MODEL_PATH),
        "fall_meta_path": str(FALL_META_PATH),
        "fall_feature_count": len(fall_feature_columns),
        "fall_threshold": fall_threshold,

        "eldercare_model_loaded": eldercare_model_loaded,
        "eldercare_model_path": str(ELDERCARE_MODEL_PATH),
        "eldercare_reason_model_path": str(ELDERCARE_REASON_MODEL_PATH),
        "eldercare_feature_count": len(eldercare_features),
        "eldercare_features": eldercare_features,

        "abnormal_row_seconds": ABNORMAL_ROW_SECONDS,
        "abnormal_state_policy": "CSV Estimation 우선, 없으면 센서 rule 보정, 마지막에 eldercare 모델 사용",
        "time_policy": "이상행동 CSV가 있으면 TimeStamp를 무시하고 1행 = 1초로 계산",

        "vital_module_loaded": vital_module is not None,
        "vital_module_import_error": VITAL_MODULE_IMPORT_ERROR,
        "vital_model_ready": vital_ready,
        "vital_sample_interval_seconds": VITAL_SAMPLE_INTERVAL_SECONDS,

        "model_file_status": get_model_file_status(),
        "model_errors": model_errors,

        "save_policy": {
            "fall": "Fall Alert만 저장",
            "abnormal": "위험/주의만 저장",
            "vital": "저장하지 않음",
        },
    }


@router.post("/simulation/upload")
async def upload_integrated_csv(
    fall_csv: Optional[UploadFile] = File(None),
    abnormal_csv: Optional[UploadFile] = File(None),
    vital_csv: Optional[UploadFile] = File(None),
    fps: int = Form(DEFAULT_FPS),
    fall_window_frames: int = Form(DEFAULT_FALL_WINDOW_FRAMES),
):
    global recent_abnormal_states

    if not fall_csv and not abnormal_csv and not vital_csv:
        raise HTTPException(status_code=400, detail="최소 1개 이상의 CSV 파일을 업로드해야 합니다.")

    if fps <= 0:
        raise HTTPException(status_code=400, detail="fps는 1 이상이어야 합니다.")

    if fall_window_frames <= 0:
        raise HTTPException(status_code=400, detail="fall_window_frames는 1 이상이어야 합니다.")

    fall_df = await read_upload_file(fall_csv) if fall_csv else None
    abnormal_df = await read_upload_file(abnormal_csv) if abnormal_csv else None
    vital_df = await read_upload_file(vital_csv) if vital_csv else None

    step_seconds = resolve_step_seconds(
        abnormal_df=abnormal_df,
        fps=fps,
        fall_window_frames=fall_window_frames,
    )

    total_steps = total_steps_by_window(
        fall_df=fall_df,
        abnormal_df=abnormal_df,
        vital_df=vital_df,
        fps=fps,
        window_frames=fall_window_frames,
        step_seconds=step_seconds,
    )

    profile = build_data_profile(fall_df, abnormal_df, vital_df)

    # v4 핵심 잠금:
    # 이상행동 CSV가 있으면 통합 재생의 기준은 무조건 이상행동 row다.
    # fall/vital 길이가 60초여도 abnormal 600행이면 600 step / 600초다.
    if profile.get("abnormal_rows", 0) > 0:
        step_seconds = float(ABNORMAL_ROW_SECONDS)
        total_steps = int(profile["abnormal_rows"])

    session_id = str(uuid4())
    recent_abnormal_states = []

    # v5 핵심 잠금:
    # 이상행동 파일이 있으면 통합 재생 길이는 무조건 이상행동 CSV 행 수와 동일하다.
    # 다른 모듈(fall/vital)이 60초여도 abnormal이 600행이면 total_steps=600이다.
    if profile["abnormal_rows"] > 0:
        step_seconds = float(ABNORMAL_ROW_SECONDS)
        total_steps = int(profile["abnormal_rows"])

    SESSIONS[session_id] = {
        "fps": fps,
        "fall_window_frames": fall_window_frames,
        "step_seconds": step_seconds,
        "cursor": 0,
        "total_steps": total_steps,
        "created_at": now_iso(),
        "data": {
            "fall": fall_df,
            "abnormal": abnormal_df,
            "vital": vital_df,
        },
        "files": {
            "fall": fall_csv.filename if fall_csv else None,
            "abnormal": abnormal_csv.filename if abnormal_csv else None,
            "vital": vital_csv.filename if vital_csv else None,
        },
        "history": [],
        "profile": profile,
        "saved_event_keys": set(),
        "final_saved": False,
        "fall_confirmed": False,
    }

    return {
        "message": "통합 CSV 업로드가 완료되었습니다.",
        "code_version": CODE_VERSION,
        "status": session_status(session_id),
        "history": [],
        "final_summary": make_final_summary([]),
    }


@router.get("/simulation/{session_id}/status")
def get_status(session_id: str):
    session = SESSIONS.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="통합 시뮬레이션 세션을 찾을 수 없습니다.")

    return {
        "status": session_status(session_id),
        "history": session["history"][:2000],
        "final_summary": make_final_summary(session["history"]),
    }


@router.post("/simulation/{session_id}/reset")
def reset_session(session_id: str):
    global recent_abnormal_states

    session = SESSIONS.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="통합 시뮬레이션 세션을 찾을 수 없습니다.")

    session["cursor"] = 0
    session["history"] = []
    session["saved_event_keys"] = set()
    session["final_saved"] = False
    session["fall_confirmed"] = False
    recent_abnormal_states = []

    return {
        "message": "통합 시뮬레이션이 초기화되었습니다.",
        "status": session_status(session_id),
        "history": [],
        "final_summary": make_final_summary([]),
    }


@router.get("/simulation/{session_id}/next")
def next_step(session_id: str):
    session = SESSIONS.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="통합 시뮬레이션 세션을 찾을 수 없습니다.")

    cursor = session["cursor"]
    total_steps = session["total_steps"]
    fps = session["fps"]
    window_frames = session["fall_window_frames"]
    step_seconds = session["step_seconds"]

    if cursor >= total_steps:
        final_summary = make_final_summary(session["history"])

        if should_save_event({}, final_summary) and not session["final_saved"]:
            save_event_to_db(
                "integrated_final_summary",
                {
                    "session_id": session_id,
                    "code_version": CODE_VERSION,
                    "final_summary": final_summary,
                    "status": session_status(session_id),
                },
            )
            session["final_saved"] = True

        return {
            "done": True,
            "status": session_status(session_id),
            "history": session["history"][:2000],
            "final_summary": final_summary,
            "message": "시뮬레이션이 종료되었습니다.",
        }

    fall_df = session["data"]["fall"]
    abnormal_df = session["data"]["abnormal"]
    vital_df = session["data"]["vital"]

    fall_chunk = slice_by_time_window(
        fall_df,
        cursor,
        fps,
        window_frames,
        "fall",
        step_seconds,
    )

    abnormal_chunk = slice_by_time_window(
        abnormal_df,
        cursor,
        fps,
        window_frames,
        "abnormal",
        step_seconds,
    )

    vital_chunk = slice_by_time_window(
        vital_df,
        cursor,
        fps,
        window_frames,
        "vital",
        step_seconds,
    )

    if int((session.get("profile") or {}).get("abnormal_rows", 0) or 0) > 0:
        current_seconds = round(cursor * ABNORMAL_ROW_SECONDS, 2)
        step_seconds = float(ABNORMAL_ROW_SECONDS)
    else:
        current_seconds = round(cursor * step_seconds, 2)

    display_second = cursor + 1

    fall_result = predict_fall_chunk(
        fall_chunk,
        fall_already_detected=session.get("fall_confirmed", False),
    )

    if fall_result.get("level") == "danger":
        session["fall_confirmed"] = True

    abnormal_result = predict_abnormal_chunk(abnormal_chunk)
    vital_result = predict_vital_chunk(vital_chunk)
    current_overall = make_current_overall(fall_result, abnormal_result, vital_result)

    result = {
        "time": now_iso(),
        "step": cursor + 1,
        "second": current_seconds,
        "display_second": display_second,
        "window": {
            "fps": fps,
            "fall_window_frames": window_frames,
            "window_seconds": step_seconds,
            "step_seconds": step_seconds,
            "abnormal_row_seconds": ABNORMAL_ROW_SECONDS,
            "abnormal_policy": "1 row = 1 second",
        },
        "overall": current_overall,
        "fall": fall_result,
        "abnormal": abnormal_result,
        "vital": vital_result,
        "db_saved": False,
        "db_status": "none",
    }

    # 먼저 history에 넣고 final_summary를 만든 뒤 저장한다.
    # 그래야 저장 payload에도 현재 위험/주의 결과와 요약이 같이 들어간다.
    session["history"].insert(0, result)
    session["cursor"] += 1

    final_summary = make_final_summary(session["history"])
    done = session["cursor"] >= total_steps

    event_key = (
        f"{cursor}:"
        f"{fall_result.get('state', '-')}:"
        f"{fall_result.get('level', '-')}:"
        f"{abnormal_result.get('state', '-')}:"
        f"{abnormal_result.get('level', '-')}:"
        f"{abnormal_result.get('risk_score', 0)}"
    )

    if should_save_event(result, final_summary) and event_key not in session["saved_event_keys"]:
        save_result = save_event_to_db(
            "integrated_realtime_event",
            {
                "session_id": session_id,
                "code_version": CODE_VERSION,
                "result": result,
                "final_summary": final_summary,
                "status": session_status(session_id),
                "save_reason": "fall danger 또는 abnormal 위험/주의 감지",
            },
        )

        result["db_saved"] = save_result["saved"]
        result["db_status"] = save_result["db_status"]
        result["db_id"] = save_result["id"]

        session["saved_event_keys"].add(event_key)

    if done and should_save_event({}, final_summary) and not session["final_saved"]:
        save_event_to_db(
            "integrated_final_summary",
            {
                "session_id": session_id,
                "code_version": CODE_VERSION,
                "final_summary": final_summary,
                "status": session_status(session_id),
            },
        )
        session["final_saved"] = True

    return {
        "done": done,
        "result": result,
        "status": session_status(session_id),
        "history": session["history"][:2000],
        "final_summary": final_summary,
    }


@router.get("/saved-events")
def get_saved_events(limit: int = 20):
    limit = max(1, min(limit, 100))

    if mongo_collection is not None:
        try:
            docs = list(
                mongo_collection
                .find({})
                .sort("saved_at", -1)
                .limit(limit)
            )

            for doc in docs:
                doc["_id"] = str(doc["_id"])

            return {
                "db_connected": True,
                "db_error": None,
                "items": docs,
                "count": len(docs),
            }

        except Exception as e:
            return {
                "db_connected": False,
                "db_error": str(e),
                "items": SAVED_LOGS[:limit],
                "count": len(SAVED_LOGS[:limit]),
            }

    return {
        "db_connected": False,
        "db_error": mongo_error,
        "items": SAVED_LOGS[:limit],
        "count": len(SAVED_LOGS[:limit]),
    }


@router.delete("/saved-events")
def clear_saved_events():
    deleted_count = 0

    if mongo_collection is not None:
        try:
            result = mongo_collection.delete_many({})
            deleted_count = result.deleted_count
        except Exception as e:
            return {
                "success": False,
                "message": "MongoDB 저장 로그 삭제 중 오류가 발생했습니다.",
                "error": str(e),
            }

    memory_count = len(SAVED_LOGS)
    SAVED_LOGS.clear()

    return {
        "success": True,
        "message": "DB 저장 로그가 삭제되었습니다.",
        "deleted_count": deleted_count,
        "memory_deleted_count": memory_count,
    }