from pathlib import Path
from datetime import datetime
import json
import math
import os
import sys
import traceback
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field


# =========================================================
# TensorFlow import
# =========================================================

try:
    import tensorflow as tf
    TENSORFLOW_IMPORT_ERROR = None
except Exception as e:
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(e)


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
# 환경변수 설정
# =========================================================

def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


VITAL_MODEL_FILE = os.getenv("VITAL_MODEL_FILE", "vital_signal_model_ver1.keras")
VITAL_SCALER_FILE = os.getenv("VITAL_SCALER_FILE", "vital_signal_scaler.pkl")
VITAL_THRESHOLD_FILE = os.getenv("VITAL_THRESHOLD_FILE", "anomaly_threshold.txt")

VITAL_MODEL_PATH = MODEL_DIR / VITAL_MODEL_FILE
VITAL_SCALER_PATH = MODEL_DIR / VITAL_SCALER_FILE
VITAL_THRESHOLD_PATH = MODEL_DIR / VITAL_THRESHOLD_FILE

SAMPLE_INTERVAL_SECONDS = get_env_float("VITAL_SAMPLE_INTERVAL_SECONDS", 0.03)
WINDOW_SECONDS = get_env_float("VITAL_WINDOW_SECONDS", 1.0)
SUSTAINED_SECONDS = get_env_float("VITAL_SUSTAINED_SECONDS", 5.0)
SEGMENT_ANOMALY_RATIO = get_env_float("VITAL_SEGMENT_ANOMALY_RATIO", 0.3)
HISTORY_MAX_ITEMS = get_env_int("VITAL_HISTORY_MAX_ITEMS", 500)

# 모델 임계값을 그대로 쓰면 예시/시연 데이터가 모두 주의로 뜰 수 있어서
# warning/danger 기준은 환경변수 배율로 조정 가능하게 둔다.
VITAL_WARNING_MULTIPLIER = get_env_float("VITAL_WARNING_MULTIPLIER", 3.0)
VITAL_DANGER_MULTIPLIER = get_env_float("VITAL_DANGER_MULTIPLIER", 5.0)

SAMPLES_PER_WINDOW = max(1, int(round(WINDOW_SECONDS / SAMPLE_INTERVAL_SECONDS)))
REQUIRED_CONTINUOUS_SEGMENTS = max(1, int(math.ceil(SUSTAINED_SECONDS / WINDOW_SECONDS)))

FEATURE_NAMES = [
    "mean",
    "std",
    "peak_to_peak",
    "zero_crossings",
    "fft_mean",
    "fft_max",
    "fft_std",
]

RAW_TIME_COLUMNS = [
    "time_sec",
    "time_seconds",
    "Time_Seconds",
    "time",
    "seconds",
    "sec",
    "timestamp",
]

RAW_SIGNAL_COLUMNS = [
    "VitalSignal",
    "vital_signal",
    "vitalSignal",
    "signal",
    "value",
    "breathing_signal",
    "breath_signal",
    "resp_signal",
    "respiration_signal",
    "wave",
]

RAW_CONDITION_COLUMNS = [
    "Condition",
    "condition",
    "label",
    "state",
    "class",
    "target",
]


# =========================================================
# Router
# =========================================================

router = APIRouter(tags=["Vital Signal"])


# =========================================================
# 전역 모델 변수
# =========================================================

vital_model = None
vital_scaler = None
vital_threshold = None

model_loaded = False
model_load_error = None

current_segment_count = 0
max_segment_count = 0


# =========================================================
# 실시간 시뮬레이션 상태
# =========================================================

simulation_state = {
    "loaded": False,
    "file_name": None,
    "input_mode": None,
    "input_description": None,
    "windows": [],
    "window_metas": [],
    "current_index": 0,
    "results": [],
    "total_samples": 0,
    "raw_total_rows": 0,
    "started_at": None,
    "finished": False,
    "history_saved": False,
}


# =========================================================
# 입력 스키마
# =========================================================

class VitalInput(BaseModel):
    features: Optional[List[float]] = Field(
        default=None,
        example=[0.0521, 0.1245, 0.8456, 0.0231, 1.2541, 0.0042, -0.0125],
    )

    mean: Optional[float] = None
    std: Optional[float] = None
    peak_to_peak: Optional[float] = None
    zero_crossings: Optional[float] = None
    fft_mean: Optional[float] = None
    fft_max: Optional[float] = None
    fft_std: Optional[float] = None


# =========================================================
# 공통 유틸
# =========================================================

def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    history = history[:HISTORY_MAX_ITEMS]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def reset_counter():
    global current_segment_count
    global max_segment_count

    current_segment_count = 0
    max_segment_count = 0


def reset_simulation_state():
    global simulation_state

    simulation_state = {
        "loaded": False,
        "file_name": None,
        "input_mode": None,
        "input_description": None,
        "windows": [],
        "window_metas": [],
        "current_index": 0,
        "results": [],
        "total_samples": 0,
        "raw_total_rows": 0,
        "started_at": None,
        "finished": False,
        "history_saved": False,
    }


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def features_to_dict(features: List[float]) -> Dict[str, float]:
    return {
        name: float(features[index])
        for index, name in enumerate(FEATURE_NAMES)
    }


def build_feature_list(data: Dict[str, Any]) -> List[float]:
    features = data.get("features")

    if features is not None:
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(f"features는 반드시 {len(FEATURE_NAMES)}개의 숫자여야 합니다.")

        return [safe_float(value) for value in features]

    return [safe_float(data.get(name)) for name in FEATURE_NAMES]


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    lower_map = {str(col).strip().lower(): col for col in df.columns}

    # 정확히 일치하는 컬럼 우선
    for name in candidates:
        key = str(name).strip().lower()
        if key in lower_map:
            return lower_map[key]

    # 그 다음 부분 포함 컬럼 허용
    for name in candidates:
        key = str(name).strip().lower()
        for lower_col, original_col in lower_map.items():
            if key and key in lower_col:
                return original_col

    return None


def mode_label(input_mode: Optional[str]) -> str:
    if input_mode == "feature_csv":
        return "전처리 feature CSV"
    if input_mode == "raw_signal_csv":
        return "원본 VitalSignal CSV 자동 전처리"
    if input_mode == "numeric_feature_fallback":
        return "숫자 컬럼 feature 자동 매핑"
    return "알 수 없음"


# =========================================================
# CSV 전처리
# - feature CSV: mean, std, peak_to_peak, zero_crossings, fft_mean, fft_max, fft_std
# - 원본 CSV: Time_Seconds, VitalSignal, Condition 등
# =========================================================

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("CSV 파일에 데이터가 없습니다.")

    result = df.copy()
    result.columns = [str(col).strip() for col in result.columns]
    return result


def has_feature_columns(df: pd.DataFrame) -> bool:
    lower_columns = {str(col).strip().lower() for col in df.columns}
    return all(name.lower() in lower_columns for name in FEATURE_NAMES)


def get_feature_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    lower_column_map = {str(col).strip().lower(): col for col in df.columns}
    selected_columns = []

    for feature_name in FEATURE_NAMES:
        original_col = lower_column_map.get(feature_name.lower())
        if original_col is None:
            raise ValueError("feature 컬럼을 찾을 수 없습니다.")
        selected_columns.append(original_col)

    feature_df = df[selected_columns].copy()
    feature_df.columns = FEATURE_NAMES
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    time_col = find_column(df, RAW_TIME_COLUMNS)
    return feature_df, time_col


def split_feature_dataframe_into_windows(df: pd.DataFrame, feature_df: pd.DataFrame, time_col: Optional[str]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    windows: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    if time_col is not None:
        times = pd.to_numeric(df[time_col], errors="coerce")
        valid_mask = times.notna()

        if valid_mask.any():
            valid_times = times[valid_mask]
            min_time = float(valid_times.min())
            max_time = float(valid_times.max())
            total_windows = max(1, int(math.ceil((max_time - min_time + 1e-9) / WINDOW_SECONDS)))

            for index in range(total_windows):
                start_time = min_time + index * WINDOW_SECONDS
                end_time = start_time + WINDOW_SECONDS

                if index == total_windows - 1:
                    mask = valid_mask & (times >= start_time) & (times <= end_time)
                else:
                    mask = valid_mask & (times >= start_time) & (times < end_time)

                chunk = feature_df.loc[mask].copy()

                if chunk.empty:
                    continue

                windows.append(chunk.to_numpy(dtype=float))
                metas.append({
                    "input_mode": "feature_csv",
                    "feature_source": "preprocessed_feature_columns",
                    "time_start": round(float(times.loc[chunk.index].min()), 3),
                    "time_end": round(float(times.loc[chunk.index].max()), 3),
                    "sample_count": int(len(chunk)),
                })

            if windows:
                return windows, metas

    total_rows = len(feature_df)

    for start in range(0, total_rows, SAMPLES_PER_WINDOW):
        end = min(start + SAMPLES_PER_WINDOW, total_rows)
        chunk = feature_df.iloc[start:end].copy()

        if chunk.empty:
            continue

        segment_index = len(windows)
        windows.append(chunk.to_numpy(dtype=float))
        metas.append({
            "input_mode": "feature_csv",
            "feature_source": "preprocessed_feature_columns",
            "time_start": round(segment_index * WINDOW_SECONDS, 3),
            "time_end": round((segment_index + 1) * WINDOW_SECONDS, 3),
            "sample_count": int(len(chunk)),
        })

    return windows, metas


def compute_raw_vital_features(signal_values: np.ndarray) -> List[float]:
    signal = np.asarray(signal_values, dtype=float)
    signal = signal[~np.isnan(signal)]

    if len(signal) == 0:
        return [0.0 for _ in FEATURE_NAMES]

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    peak_to_peak = float(np.max(signal) - np.min(signal))

    centered = signal - mean

    if len(centered) >= 2:
        signs = np.sign(centered)
        # 0이 길게 이어지는 경우를 줄이기 위해 작은 값을 0으로 처리
        signs[np.abs(centered) < 1e-12] = 0
        zero_crossings = int(np.sum(np.diff(signs) != 0))
    else:
        zero_crossings = 0

    if len(centered) >= 2:
        fft_values = np.abs(np.fft.rfft(centered))
        fft_mean = float(np.mean(fft_values))
        fft_max = float(np.max(fft_values))
        fft_std = float(np.std(fft_values))
    else:
        fft_mean = 0.0
        fft_max = 0.0
        fft_std = 0.0

    return [
        round(mean, 6),
        round(std, 6),
        round(peak_to_peak, 6),
        round(float(zero_crossings), 6),
        round(fft_mean, 6),
        round(fft_max, 6),
        round(fft_std, 6),
    ]


def dominant_text_value(series: pd.Series, default: str = "-") -> str:
    try:
        values = series.dropna().astype(str)
        if values.empty:
            return default
        counts = values.value_counts()
        if counts.empty:
            return values.iloc[-1]
        return str(counts.index[0])
    except Exception:
        return default


def split_raw_signal_dataframe_into_windows(df: pd.DataFrame, signal_col: str, time_col: Optional[str], condition_col: Optional[str]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    windows: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    temp = df.copy()
    temp["__signal__"] = pd.to_numeric(temp[signal_col], errors="coerce")
    temp = temp.dropna(subset=["__signal__"])

    if temp.empty:
        raise ValueError("원본 바이탈 CSV에서 숫자형 VitalSignal 값을 찾을 수 없습니다.")

    if time_col is not None:
        temp["__time__"] = pd.to_numeric(temp[time_col], errors="coerce")
        temp = temp.dropna(subset=["__time__"])
        temp = temp.sort_values("__time__")

        if temp.empty:
            raise ValueError("원본 바이탈 CSV에서 숫자형 시간 값을 찾을 수 없습니다.")

        min_time = float(temp["__time__"].min())
        max_time = float(temp["__time__"].max())
        total_windows = max(1, int(math.ceil((max_time - min_time + 1e-9) / WINDOW_SECONDS)))

        for index in range(total_windows):
            start_time = min_time + index * WINDOW_SECONDS
            end_time = start_time + WINDOW_SECONDS

            if index == total_windows - 1:
                chunk = temp[(temp["__time__"] >= start_time) & (temp["__time__"] <= end_time)].copy()
            else:
                chunk = temp[(temp["__time__"] >= start_time) & (temp["__time__"] < end_time)].copy()

            if chunk.empty:
                continue

            features = compute_raw_vital_features(chunk["__signal__"].to_numpy(dtype=float))
            condition = dominant_text_value(chunk[condition_col]) if condition_col else "-"

            windows.append(np.asarray([features], dtype=float))
            metas.append({
                "input_mode": "raw_signal_csv",
                "feature_source": "computed_from_raw_vital_signal",
                "raw_signal_column": signal_col,
                "raw_time_column": time_col,
                "condition": condition,
                "time_start": round(float(chunk["__time__"].min()), 3),
                "time_end": round(float(chunk["__time__"].max()), 3),
                "sample_count": int(len(chunk)),
            })

        return windows, metas

    total_rows = len(temp)

    for start in range(0, total_rows, SAMPLES_PER_WINDOW):
        end = min(start + SAMPLES_PER_WINDOW, total_rows)
        chunk = temp.iloc[start:end].copy()

        if chunk.empty:
            continue

        features = compute_raw_vital_features(chunk["__signal__"].to_numpy(dtype=float))
        condition = dominant_text_value(chunk[condition_col]) if condition_col else "-"
        segment_index = len(windows)

        windows.append(np.asarray([features], dtype=float))
        metas.append({
            "input_mode": "raw_signal_csv",
            "feature_source": "computed_from_raw_vital_signal",
            "raw_signal_column": signal_col,
            "raw_time_column": None,
            "condition": condition,
            "time_start": round(segment_index * WINDOW_SECONDS, 3),
            "time_end": round((segment_index + 1) * WINDOW_SECONDS, 3),
            "sample_count": int(len(chunk)),
        })

    return windows, metas


def split_numeric_fallback_into_windows(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if len(numeric_df.columns) < len(FEATURE_NAMES):
        raise ValueError(
            "CSV에서 생체신호 입력을 만들 수 없습니다. "
            "전처리 feature 컬럼(mean, std, peak_to_peak, zero_crossings, fft_mean, fft_max, fft_std) "
            "또는 원본 VitalSignal 컬럼이 필요합니다."
        )

    feature_df = numeric_df.iloc[:, :len(FEATURE_NAMES)].copy()
    feature_df.columns = FEATURE_NAMES
    feature_df = feature_df.fillna(0.0)

    windows: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    for start in range(0, len(feature_df), SAMPLES_PER_WINDOW):
        end = min(start + SAMPLES_PER_WINDOW, len(feature_df))
        chunk = feature_df.iloc[start:end].copy()

        if chunk.empty:
            continue

        segment_index = len(windows)
        windows.append(chunk.to_numpy(dtype=float))
        metas.append({
            "input_mode": "numeric_feature_fallback",
            "feature_source": "first_7_numeric_columns",
            "time_start": round(segment_index * WINDOW_SECONDS, 3),
            "time_end": round((segment_index + 1) * WINDOW_SECONDS, 3),
            "sample_count": int(len(chunk)),
        })

    return windows, metas


def prepare_vital_windows_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    df = normalize_dataframe(df)

    # 1순위: 이미 전처리된 feature CSV
    if has_feature_columns(df):
        feature_df, time_col = get_feature_dataframe(df)
        windows, metas = split_feature_dataframe_into_windows(df, feature_df, time_col)
        return {
            "input_mode": "feature_csv",
            "input_description": mode_label("feature_csv"),
            "windows": windows,
            "window_metas": metas,
            "total_samples": int(len(feature_df)),
            "raw_total_rows": int(len(df)),
            "detected_columns": {
                "features": FEATURE_NAMES,
                "time": time_col,
                "signal": None,
            },
        }

    # 2순위: 원본 VitalSignal CSV → 1초 단위 feature 자동 계산
    signal_col = find_column(df, RAW_SIGNAL_COLUMNS)

    if signal_col is not None:
        time_col = find_column(df, RAW_TIME_COLUMNS)
        condition_col = find_column(df, RAW_CONDITION_COLUMNS)
        windows, metas = split_raw_signal_dataframe_into_windows(df, signal_col, time_col, condition_col)

        return {
            "input_mode": "raw_signal_csv",
            "input_description": mode_label("raw_signal_csv"),
            "windows": windows,
            "window_metas": metas,
            "total_samples": int(len(df)),
            "raw_total_rows": int(len(df)),
            "detected_columns": {
                "features": None,
                "time": time_col,
                "signal": signal_col,
                "condition": condition_col,
            },
        }

    # 3순위: 하위 호환. 숫자 컬럼 7개를 feature로 간주
    windows, metas = split_numeric_fallback_into_windows(df)

    return {
        "input_mode": "numeric_feature_fallback",
        "input_description": mode_label("numeric_feature_fallback"),
        "windows": windows,
        "window_metas": metas,
        "total_samples": int(len(df)),
        "raw_total_rows": int(len(df)),
        "detected_columns": {
            "features": "first_7_numeric_columns",
            "time": None,
            "signal": None,
        },
    }


# 기존 코드 호환용 함수
# feature CSV면 row별 feature matrix를 반환하고, 원본 CSV면 1초 feature matrix를 반환한다.
def extract_feature_matrix_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    prepared = prepare_vital_windows_from_dataframe(df)

    if not prepared["windows"]:
        raise ValueError("분석 가능한 생체신호 구간이 없습니다.")

    rows = []
    for window in prepared["windows"]:
        window = np.asarray(window, dtype=float)
        if window.ndim == 1:
            rows.append(window)
        else:
            rows.extend(window.tolist())

    return np.asarray(rows, dtype=float)


def split_matrix_into_windows(feature_matrix: np.ndarray) -> List[np.ndarray]:
    windows = []
    total_rows = len(feature_matrix)

    for start in range(0, total_rows, SAMPLES_PER_WINDOW):
        end = min(start + SAMPLES_PER_WINDOW, total_rows)
        window = feature_matrix[start:end]

        if len(window) > 0:
            windows.append(window)

    return windows


# =========================================================
# 모델 로딩
# =========================================================

def startup_vital_signal():
    global vital_model
    global vital_scaler
    global vital_threshold
    global model_loaded
    global model_load_error

    ensure_history_file()

    vital_model = None
    vital_scaler = None
    vital_threshold = None
    model_loaded = False
    model_load_error = None

    if tf is None:
        model_load_error = f"TensorFlow import 실패: {TENSORFLOW_IMPORT_ERROR}"
        print(f"[VITAL] {model_load_error}")
        return

    missing_files = []

    if not VITAL_MODEL_PATH.exists():
        missing_files.append(str(VITAL_MODEL_PATH))

    if not VITAL_SCALER_PATH.exists():
        missing_files.append(str(VITAL_SCALER_PATH))

    if not VITAL_THRESHOLD_PATH.exists():
        missing_files.append(str(VITAL_THRESHOLD_PATH))

    if missing_files:
        model_load_error = "필수 모델 파일이 없습니다: " + ", ".join(missing_files)
        print(f"[VITAL] {model_load_error}")
        return

    try:
        vital_model = tf.keras.models.load_model(VITAL_MODEL_PATH, compile=False)
        vital_scaler = joblib.load(VITAL_SCALER_PATH)

        with open(VITAL_THRESHOLD_PATH, "r", encoding="utf-8") as f:
            vital_threshold = float(f.read().strip())

        model_loaded = True
        model_load_error = None

        print("[VITAL] 생체신호 AutoEncoder 모델 로드 성공")
        print(f"[VITAL] model    : {VITAL_MODEL_PATH}")
        print(f"[VITAL] scaler   : {VITAL_SCALER_PATH}")
        print(f"[VITAL] threshold: {vital_threshold}")
        print(f"[VITAL] warning  : threshold x {VITAL_WARNING_MULTIPLIER}")
        print(f"[VITAL] danger   : threshold x {VITAL_DANGER_MULTIPLIER}")
        print(f"[VITAL] sample interval: {SAMPLE_INTERVAL_SECONDS}s")
        print(f"[VITAL] window seconds : {WINDOW_SECONDS}s")
        print(f"[VITAL] samples/window : {SAMPLES_PER_WINDOW}")

    except Exception as e:
        vital_model = None
        vital_scaler = None
        vital_threshold = None
        model_loaded = False
        model_load_error = str(e)

        print("[VITAL] 모델 로드 실패")
        traceback.print_exc()


def shutdown_vital_signal():
    print("[VITAL] shutdown 완료")


# =========================================================
# 상태 확인
# =========================================================

def health():
    return {
        "status": "ok",
        "api": "vital-running",
        "message": "Vital Signal API is running",

        "model_loaded": model_loaded,
        "model_path": str(VITAL_MODEL_PATH),
        "scaler_path": str(VITAL_SCALER_PATH),
        "threshold_path": str(VITAL_THRESHOLD_PATH),
        "model_file_exists": VITAL_MODEL_PATH.exists(),
        "scaler_file_exists": VITAL_SCALER_PATH.exists(),
        "threshold_file_exists": VITAL_THRESHOLD_PATH.exists(),
        "model_load_error": model_load_error,

        "tensorflow_available": tf is not None,
        "tensorflow_version": getattr(tf, "__version__", None) if tf is not None else None,
        "tensorflow_import_error": TENSORFLOW_IMPORT_ERROR,

        "feature_names": FEATURE_NAMES,
        "required_feature_count": len(FEATURE_NAMES),
        "raw_signal_columns": RAW_SIGNAL_COLUMNS,
        "raw_time_columns": RAW_TIME_COLUMNS,
        "raw_condition_columns": RAW_CONDITION_COLUMNS,
        "supported_input_modes": ["feature_csv", "raw_signal_csv", "numeric_feature_fallback"],

        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "samples_per_window": SAMPLES_PER_WINDOW,
        "sustained_seconds": SUSTAINED_SECONDS,
        "required_continuous_segments": REQUIRED_CONTINUOUS_SEGMENTS,
        "segment_anomaly_ratio": SEGMENT_ANOMALY_RATIO,
        "warning_multiplier": VITAL_WARNING_MULTIPLIER,
        "danger_multiplier": VITAL_DANGER_MULTIPLIER,

        "history_file": str(HISTORY_FILE),
        "history_file_exists": HISTORY_FILE.exists(),

        "current_segment_count": current_segment_count,
        "max_segment_count": max_segment_count,

        "simulation_loaded": simulation_state["loaded"],
        "simulation_file_name": simulation_state["file_name"],
        "simulation_input_mode": simulation_state["input_mode"],
        "simulation_input_description": simulation_state["input_description"],
        "simulation_total_windows": len(simulation_state["windows"]),
        "simulation_current_index": simulation_state["current_index"],
        "simulation_finished": simulation_state["finished"],

        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "model_dir": str(MODEL_DIR),
    }


# =========================================================
# 예측 로직
# =========================================================

def check_model_ready():
    return model_loaded and vital_model is not None and vital_scaler is not None and vital_threshold is not None


def predict_errors(feature_matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(feature_matrix, dtype=float)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"입력 feature 개수는 {len(FEATURE_NAMES)}개여야 합니다.")

    x_scaled = vital_scaler.transform(x)
    x_pred = vital_model.predict(x_scaled, verbose=0)

    errors = np.mean(np.square(x_scaled - x_pred), axis=1)

    return errors


def make_segment_result(
    segment_index: int,
    window_matrix: np.ndarray,
    window_errors: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    global current_segment_count
    global max_segment_count

    meta = meta or {}
    threshold = float(vital_threshold)
    warning_threshold = threshold * VITAL_WARNING_MULTIPLIER
    danger_threshold = threshold * VITAL_DANGER_MULTIPLIER

    start_second = meta.get("time_start")
    end_second = meta.get("time_end")

    if start_second is None:
        start_second = round(segment_index * WINDOW_SECONDS, 2)
    if end_second is None:
        end_second = round(float(start_second) + WINDOW_SECONDS, 2)

    window_errors = np.asarray(window_errors, dtype=float)

    warning_flags = window_errors > warning_threshold
    danger_flags = window_errors > danger_threshold

    warning_count = int(np.sum(warning_flags))
    danger_count = int(np.sum(danger_flags))
    row_count = int(len(window_errors))

    warning_ratio = float(warning_count / row_count) if row_count > 0 else 0.0
    danger_ratio = float(danger_count / row_count) if row_count > 0 else 0.0

    error_mean = float(np.mean(window_errors)) if row_count > 0 else 0.0
    error_max = float(np.max(window_errors)) if row_count > 0 else 0.0
    error_min = float(np.min(window_errors)) if row_count > 0 else 0.0

    representative_features = np.mean(np.asarray(window_matrix, dtype=float), axis=0).tolist()

    is_segment_warning = (
        error_max > warning_threshold
        or warning_ratio >= SEGMENT_ANOMALY_RATIO
    )

    is_segment_danger = (
        error_max > danger_threshold
        or danger_ratio >= SEGMENT_ANOMALY_RATIO
    )

    if is_segment_warning:
        current_segment_count += 1

        if current_segment_count > max_segment_count:
            max_segment_count = current_segment_count
    else:
        current_segment_count = 0

    sustained_anomaly = current_segment_count >= REQUIRED_CONTINUOUS_SEGMENTS

    ratio = error_max / threshold if threshold > 0 else 0.0

    if is_segment_danger or sustained_anomaly:
        status = "Danger"
        state = "위험"
        alert = True
        risk_score = 95
        message = "생체신호 이상이 강하게 감지되었거나 여러 초 동안 지속되었습니다."
    elif is_segment_warning:
        status = "Warning"
        state = "주의"
        alert = False
        risk_score = min(89, max(60, int((error_max / warning_threshold) * 70))) if warning_threshold > 0 else 65
        message = "해당 1초 구간에서 생체신호 이상 가능성이 감지되었습니다."
    else:
        status = "Normal"
        state = "정상"
        alert = False
        risk_score = min(35, max(0, int((error_max / warning_threshold) * 35))) if warning_threshold > 0 else 0
        message = "해당 1초 구간은 정상 패턴으로 판단됩니다."

    return {
        "segment_index": segment_index + 1,
        "start_second": round(float(start_second), 3),
        "end_second": round(float(end_second), 3),
        "window_seconds": WINDOW_SECONDS,
        "sample_count": int(meta.get("sample_count", row_count)),

        "input_mode": meta.get("input_mode"),
        "input_description": mode_label(meta.get("input_mode")),
        "feature_source": meta.get("feature_source"),
        "condition": meta.get("condition", "-"),
        "raw_signal_column": meta.get("raw_signal_column"),
        "raw_time_column": meta.get("raw_time_column"),

        "status": status,
        "state": state,
        "alert": alert,
        "risk_score": int(risk_score),
        "message": message,

        "error_mean": round(error_mean, 6),
        "error_max": round(error_max, 6),
        "error_min": round(error_min, 6),
        "threshold": round(threshold, 6),
        "warning_threshold": round(warning_threshold, 6),
        "danger_threshold": round(danger_threshold, 6),
        "warning_multiplier": VITAL_WARNING_MULTIPLIER,
        "danger_multiplier": VITAL_DANGER_MULTIPLIER,
        "error_ratio": round(ratio, 4),

        "is_anomaly": bool(is_segment_warning or is_segment_danger),
        "anomaly_count": warning_count,
        "anomaly_ratio": round(warning_ratio, 4),
        "danger_count": danger_count,
        "danger_ratio": round(danger_ratio, 4),

        "current_segment_count": current_segment_count,
        "max_segment_count": max_segment_count,
        "required_continuous_segments": REQUIRED_CONTINUOUS_SEGMENTS,

        "features": features_to_dict(representative_features),
        "feature_list": [float(v) for v in representative_features],
    }


def summarize_segments(
    segments: List[Dict[str, Any]],
    source: str,
    total_samples: int,
    save_history: bool = True,
    input_mode: Optional[str] = None,
    input_description: Optional[str] = None,
    raw_total_rows: Optional[int] = None,
) -> Dict[str, Any]:
    if not segments:
        return {
            "success": False,
            "time": now_text(),
            "source": source,
            "status": "Error",
            "state": "오류",
            "message": "분석 가능한 1초 구간이 없습니다.",
            "segments": [],
        }

    input_mode = input_mode or segments[0].get("input_mode")
    input_description = input_description or mode_label(input_mode)

    danger_segments = [item for item in segments if item["status"] == "Danger"]
    warning_segments = [item for item in segments if item["status"] == "Warning"]
    normal_segments = [item for item in segments if item["status"] == "Normal"]

    max_risk_segment = max(segments, key=lambda item: item.get("risk_score", 0))
    max_error_segment = max(segments, key=lambda item: item.get("error_max", 0))

    if danger_segments:
        status = "Danger"
        state = "위험"
        alert = True
        message = "생체신호 위험 구간이 감지되었습니다."
    elif warning_segments:
        status = "Warning"
        state = "주의"
        alert = False
        message = "생체신호 주의 구간이 감지되었습니다."
    else:
        status = "Normal"
        state = "정상"
        alert = False
        message = "전체 구간이 정상 패턴으로 판단됩니다."

    result = {
        "success": True,
        "time": now_text(),
        "source": source,
        "mode": "autoencoder_realtime_window",
        "input_mode": input_mode,
        "input_description": input_description,

        "status": status,
        "state": state,
        "alert": alert,
        "guardian_alert": alert,
        "guardian_message": (
            "생체신호 위험 상태로 판단되어 보호자 확인이 필요합니다."
            if alert
            else "보호자 즉시 알림 기준에는 도달하지 않았습니다."
        ),

        "risk_score": max_risk_segment.get("risk_score", 0),
        "message": message,
        "reasons": [
            f"입력 방식: {input_description}",
            f"{WINDOW_SECONDS}초 단위로 생체신호를 묶어 분석했습니다.",
            f"원본 CSV는 VitalSignal을 1초 단위 feature로 자동 전처리합니다.",
            f"최대 복원 오차 구간: {max_error_segment.get('segment_index')}번 구간",
        ],

        "total_samples": total_samples,
        "raw_total_rows": raw_total_rows if raw_total_rows is not None else total_samples,
        "total_segments": len(segments),
        "normal_count": len(normal_segments),
        "warning_count": len(warning_segments),
        "danger_count": len(danger_segments),

        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "samples_per_window": SAMPLES_PER_WINDOW,

        "max_error": max_error_segment.get("error_max"),
        "threshold": max_error_segment.get("threshold"),
        "warning_threshold": max_error_segment.get("warning_threshold"),
        "danger_threshold": max_error_segment.get("danger_threshold"),
        "is_anomaly": len(warning_segments) > 0 or len(danger_segments) > 0,

        "segments": segments,

        "model_loaded": model_loaded,
        "model_path": str(VITAL_MODEL_PATH),
    }

    if save_history:
        save_history_item(result)

    return result


def predict_single(features: List[float], source: str = "manual") -> Dict[str, Any]:
    if not check_model_ready():
        return {
            "success": False,
            "time": now_text(),
            "source": source,
            "status": "Error",
            "state": "모델 미로드",
            "message": "생체신호 모델이 로드되지 않았습니다.",
            "model_loaded": False,
            "model_load_error": model_load_error,
            "features": features_to_dict(features),
        }

    reset_counter()

    feature_matrix = np.asarray([features], dtype=float)
    errors = predict_errors(feature_matrix)

    meta = {
        "input_mode": "feature_csv",
        "feature_source": "manual_feature_input",
        "sample_count": 1,
        "time_start": 0,
        "time_end": WINDOW_SECONDS,
    }

    segment = make_segment_result(
        segment_index=0,
        window_matrix=feature_matrix,
        window_errors=errors,
        meta=meta,
    )

    return summarize_segments(
        segments=[segment],
        source=source,
        total_samples=1,
        input_mode="feature_csv",
        input_description=mode_label("feature_csv"),
    )


def predict_prepared_windows(prepared: Dict[str, Any], source: str, save_history: bool = True) -> Dict[str, Any]:
    if not check_model_ready():
        return {
            "success": False,
            "time": now_text(),
            "source": source,
            "status": "Error",
            "state": "모델 미로드",
            "message": "생체신호 모델이 로드되지 않았습니다.",
            "model_loaded": False,
            "model_load_error": model_load_error,
            "input_mode": prepared.get("input_mode"),
            "input_description": prepared.get("input_description"),
        }

    reset_counter()

    segments = []
    windows = prepared.get("windows", [])
    metas = prepared.get("window_metas", [])

    for index, window_matrix in enumerate(windows):
        window_errors = predict_errors(window_matrix)
        meta = metas[index] if index < len(metas) else {}

        segment = make_segment_result(
            segment_index=index,
            window_matrix=window_matrix,
            window_errors=window_errors,
            meta=meta,
        )

        segments.append(segment)

    return summarize_segments(
        segments=segments,
        source=source,
        total_samples=prepared.get("total_samples", 0),
        save_history=save_history,
        input_mode=prepared.get("input_mode"),
        input_description=prepared.get("input_description"),
        raw_total_rows=prepared.get("raw_total_rows"),
    )


# 기존 코드 호환용
# feature_matrix를 직접 받은 경우에는 feature CSV로 간주한다.
def predict_window_file(feature_matrix: np.ndarray, source: str) -> Dict[str, Any]:
    prepared = {
        "input_mode": "feature_csv",
        "input_description": mode_label("feature_csv"),
        "windows": split_matrix_into_windows(feature_matrix),
        "window_metas": [],
        "total_samples": len(feature_matrix),
        "raw_total_rows": len(feature_matrix),
    }
    return predict_prepared_windows(prepared, source=source)


def get_simulation_summary(save_history_when_finished: bool = False) -> Dict[str, Any]:
    should_save = save_history_when_finished and not simulation_state.get("history_saved", False)

    summary = summarize_segments(
        segments=simulation_state["results"],
        source=simulation_state["file_name"] or "simulation",
        total_samples=simulation_state["total_samples"],
        save_history=should_save,
        input_mode=simulation_state.get("input_mode"),
        input_description=simulation_state.get("input_description"),
        raw_total_rows=simulation_state.get("raw_total_rows"),
    )

    if should_save and summary.get("success"):
        simulation_state["history_saved"] = True

    return summary


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
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "description": {
            "mean": "신호 평균값",
            "std": "신호 표준편차",
            "peak_to_peak": "최댓값과 최솟값 차이",
            "zero_crossings": "평균 기준 교차 횟수",
            "fft_mean": "FFT 평균",
            "fft_max": "FFT 최댓값",
            "fft_std": "FFT 표준편차",
        },
        "supported_csv_formats": {
            "feature_csv": {
                "required_columns": FEATURE_NAMES,
                "description": "이미 전처리된 7개 feature CSV",
            },
            "raw_signal_csv": {
                "required_signal_columns_any_of": RAW_SIGNAL_COLUMNS,
                "optional_time_columns_any_of": RAW_TIME_COLUMNS,
                "optional_condition_columns_any_of": RAW_CONDITION_COLUMNS,
                "description": "원본 VitalSignal을 1초 단위 feature로 자동 전처리",
            },
        },
        "window_config": {
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "samples_per_window": SAMPLES_PER_WINDOW,
            "sustained_seconds": SUSTAINED_SECONDS,
            "required_continuous_segments": REQUIRED_CONTINUOUS_SEGMENTS,
            "warning_multiplier": VITAL_WARNING_MULTIPLIER,
            "danger_multiplier": VITAL_DANGER_MULTIPLIER,
        },
    }


@router.post("/vital/predict")
async def vital_predict(data: VitalInput):
    try:
        features = build_feature_list(data.dict())
        return predict_single(features, source="manual")

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "status": "Error",
            "state": "오류",
            "message": f"생체신호 예측 중 오류가 발생했습니다: {e}",
            "time": now_text(),
        }


@router.post("/vital/predict-file")
async def vital_predict_file(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".csv"):
            return {
                "success": False,
                "status": "Error",
                "state": "오류",
                "message": "CSV 파일만 업로드할 수 있습니다.",
                "time": now_text(),
            }

        df = pd.read_csv(file.file)
        prepared = prepare_vital_windows_from_dataframe(df)

        return predict_prepared_windows(
            prepared=prepared,
            source=file.filename,
            save_history=True,
        )

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "status": "Error",
            "state": "오류",
            "message": f"CSV 생체신호 예측 중 오류가 발생했습니다: {e}",
            "time": now_text(),
        }


# =========================================================
# 실시간 시뮬레이션 API
# =========================================================

@router.post("/vital/simulation/upload")
async def vital_simulation_upload(file: UploadFile = File(...)):
    global simulation_state

    try:
        if not file.filename.lower().endswith(".csv"):
            return {
                "success": False,
                "status": "Error",
                "message": "CSV 파일만 업로드할 수 있습니다.",
                "time": now_text(),
            }

        if not check_model_ready():
            return {
                "success": False,
                "status": "Error",
                "message": "생체신호 모델이 로드되지 않았습니다.",
                "model_load_error": model_load_error,
                "time": now_text(),
            }

        df = pd.read_csv(file.file)
        prepared = prepare_vital_windows_from_dataframe(df)

        reset_counter()

        simulation_state = {
            "loaded": True,
            "file_name": file.filename,
            "input_mode": prepared.get("input_mode"),
            "input_description": prepared.get("input_description"),
            "detected_columns": prepared.get("detected_columns"),
            "windows": prepared.get("windows", []),
            "window_metas": prepared.get("window_metas", []),
            "current_index": 0,
            "results": [],
            "total_samples": prepared.get("total_samples", 0),
            "raw_total_rows": prepared.get("raw_total_rows", 0),
            "started_at": now_text(),
            "finished": False,
            "history_saved": False,
        }

        return {
            "success": True,
            "status": "ok",
            "message": "생체신호 CSV가 실시간 시뮬레이션용으로 업로드되었습니다.",
            "file_name": file.filename,
            "input_mode": simulation_state["input_mode"],
            "input_description": simulation_state["input_description"],
            "detected_columns": simulation_state["detected_columns"],
            "total_samples": simulation_state["total_samples"],
            "raw_total_rows": simulation_state["raw_total_rows"],
            "total_segments": len(simulation_state["windows"]),
            "current_index": 0,
            "remaining_segments": len(simulation_state["windows"]),
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "samples_per_window": SAMPLES_PER_WINDOW,
            "time": now_text(),
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "status": "Error",
            "message": f"실시간 시뮬레이션 업로드 중 오류가 발생했습니다: {e}",
            "time": now_text(),
        }


@router.get("/vital/simulation/status")
def vital_simulation_status():
    total_segments = len(simulation_state["windows"])
    current_index = simulation_state["current_index"]

    return {
        "success": True,
        "loaded": simulation_state["loaded"],
        "file_name": simulation_state["file_name"],
        "input_mode": simulation_state.get("input_mode"),
        "input_description": simulation_state.get("input_description"),
        "detected_columns": simulation_state.get("detected_columns"),
        "total_samples": simulation_state["total_samples"],
        "raw_total_rows": simulation_state.get("raw_total_rows", 0),
        "total_segments": total_segments,
        "current_index": current_index,
        "current_segment": min(current_index + 1, total_segments) if total_segments else 0,
        "processed_segments": len(simulation_state["results"]),
        "remaining_segments": max(0, total_segments - current_index),
        "finished": simulation_state["finished"],
        "started_at": simulation_state["started_at"],
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "samples_per_window": SAMPLES_PER_WINDOW,
        "results": simulation_state["results"],
    }


@router.post("/vital/simulation/reset")
def vital_simulation_reset():
    if not simulation_state["loaded"]:
        reset_counter()
        return {
            "success": True,
            "status": "ok",
            "message": "업로드된 생체신호 시뮬레이션 파일이 없습니다.",
            "time": now_text(),
        }

    reset_counter()

    simulation_state["current_index"] = 0
    simulation_state["results"] = []
    simulation_state["finished"] = False
    simulation_state["started_at"] = now_text()
    simulation_state["history_saved"] = False

    return {
        "success": True,
        "status": "ok",
        "message": "생체신호 실시간 시뮬레이션이 처음부터 다시 시작됩니다.",
        "file_name": simulation_state["file_name"],
        "input_mode": simulation_state.get("input_mode"),
        "input_description": simulation_state.get("input_description"),
        "total_segments": len(simulation_state["windows"]),
        "current_index": 0,
        "time": now_text(),
    }


@router.post("/vital/simulation/clear")
def vital_simulation_clear():
    reset_counter()
    reset_simulation_state()

    return {
        "success": True,
        "status": "ok",
        "message": "생체신호 실시간 시뮬레이션 데이터가 초기화되었습니다.",
        "time": now_text(),
    }


@router.post("/vital/simulation/next")
def vital_simulation_next():
    try:
        if not simulation_state["loaded"]:
            return {
                "success": False,
                "status": "Error",
                "message": "먼저 CSV 파일을 업로드해주세요.",
                "time": now_text(),
            }

        if not check_model_ready():
            return {
                "success": False,
                "status": "Error",
                "message": "생체신호 모델이 로드되지 않았습니다.",
                "model_load_error": model_load_error,
                "time": now_text(),
            }

        total_segments = len(simulation_state["windows"])
        current_index = simulation_state["current_index"]

        if current_index >= total_segments:
            simulation_state["finished"] = True

            summary = get_simulation_summary(save_history_when_finished=True)

            return {
                "success": True,
                "status": "finished",
                "message": "생체신호 실시간 시뮬레이션이 완료되었습니다.",
                "finished": True,
                "summary": summary,
                "segment": None,
                "results": simulation_state["results"],
                "time": now_text(),
            }

        window_matrix = simulation_state["windows"][current_index]
        window_errors = predict_errors(window_matrix)
        meta = simulation_state["window_metas"][current_index] if current_index < len(simulation_state["window_metas"]) else {}

        segment = make_segment_result(
            segment_index=current_index,
            window_matrix=window_matrix,
            window_errors=window_errors,
            meta=meta,
        )

        simulation_state["results"].append(segment)
        simulation_state["current_index"] += 1

        finished = simulation_state["current_index"] >= total_segments
        simulation_state["finished"] = finished

        summary = get_simulation_summary(save_history_when_finished=finished)

        return {
            "success": True,
            "status": "ok",
            "message": "다음 1초 구간 분석 완료",
            "finished": finished,
            "file_name": simulation_state["file_name"],
            "input_mode": simulation_state.get("input_mode"),
            "input_description": simulation_state.get("input_description"),
            "segment": segment,
            "summary": summary,
            "results": simulation_state["results"],
            "total_segments": total_segments,
            "current_index": simulation_state["current_index"],
            "remaining_segments": max(0, total_segments - simulation_state["current_index"]),
            "time": now_text(),
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "status": "Error",
            "message": f"다음 구간 분석 중 오류가 발생했습니다: {e}",
            "time": now_text(),
        }


@router.post("/vital_predict")
async def old_vital_predict(data: VitalInput):
    try:
        features = build_feature_list(data.dict())
        return predict_single(features, source="old_vital_predict")

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "status": "Error",
            "state": "오류",
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

    segment_total = 0

    for item in history:
        segments = item.get("segments") or []
        segment_total += len(segments)

    return {
        "status": "ok",
        "total": total,
        "normal_count": normal_count,
        "warning_count": warning_count,
        "danger_count": danger_count,
        "alert_count": alert_count,
        "segment_total": segment_total,
        "model_loaded": model_loaded,
        "model_path": str(VITAL_MODEL_PATH),
        "model_load_error": model_load_error,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "samples_per_window": SAMPLES_PER_WINDOW,
        "warning_multiplier": VITAL_WARNING_MULTIPLIER,
        "danger_multiplier": VITAL_DANGER_MULTIPLIER,
    }


@router.post("/vital/reset")
def vital_reset():
    reset_counter()

    return {
        "status": "ok",
        "message": "생체신호 이상 연속 카운트가 초기화되었습니다.",
        "current_segment_count": current_segment_count,
        "max_segment_count": max_segment_count,
        "time": now_text(),
    }
