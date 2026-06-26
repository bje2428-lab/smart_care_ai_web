from pathlib import Path
from datetime import datetime
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional
from uuid import uuid4
import inspect
import json
import math
import os
import sys
import tempfile
import traceback

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile


# =========================================================
# Router
# =========================================================

router = APIRouter(tags=["Fall Dashboard"])


# =========================================================
# 경로 설정
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
RF_DIR = ROOT_DIR / "rf"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

for path in [ROOT_DIR, BACKEND_DIR, RF_DIR, MODEL_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =========================================================
# 기본 설정
# =========================================================

DEFAULT_FPS = 10
DEFAULT_WINDOW_FRAMES = 10

MODEL_CANDIDATES = [
    MODEL_DIR / "mmwave_rf_aug_model.pkl",
    MODEL_DIR / "mmwave_rf_smote_model.pkl",
    MODEL_DIR / "mmwave_rf_model.pkl",
    RF_DIR / "models" / "mmwave_rf_aug_model.pkl",
    RF_DIR / "models" / "mmwave_rf_smote_model.pkl",
    RF_DIR / "models" / "mmwave_rf_model.pkl",
]

META_CANDIDATES = [
    MODEL_DIR / "mmwave_rf_aug_model_meta.json",
    MODEL_DIR / "mmwave_rf_smote_model_meta.json",
    MODEL_DIR / "mmwave_rf_model_meta.json",
    RF_DIR / "models" / "mmwave_rf_aug_model_meta.json",
    RF_DIR / "models" / "mmwave_rf_smote_model_meta.json",
    RF_DIR / "models" / "mmwave_rf_model_meta.json",
]

DATA_DIR = ROOT_DIR / "data" / "mmwave_fall" / "GatheredData"
FALL_DIR = DATA_DIR / "Fall"
NOT_DIR = DATA_DIR / "Not"


# =========================================================
# 컬럼 후보
# =========================================================

FRAME_COLS = ["frame", "Frame", "frame_id", "frameId", "frame_index"]
X_COLS = ["x", "X", "pos_x", "point_x", "x_pos"]
Y_COLS = ["y", "Y", "pos_y", "point_y", "y_pos"]
Z_COLS = ["z", "Z", "height", "pos_z", "point_z", "z_pos"]
V_COLS = [
    "v",
    "V",
    "velocity",
    "Velocity",
    "speed",
    "Speed",
    "vel",
    "doppler",
    "radial_velocity",
]

TEXT_LABEL_COLS = [
    "behavior_label",
    "behavior",
    "action",
    "activity",
    "state",
    "label",
    "actual_state",
    "class",
    "target",
]

SCENARIO_COLS = ["scenario", "scene", "phase"]
DESCRIPTION_COLS = ["description", "desc", "actual_reason", "reason"]


# =========================================================
# 전역 상태
# =========================================================

model = None
model_meta: Dict[str, Any] = {}
model_path: Optional[Path] = None
model_load_error: Optional[str] = None

feature_extractor_import_error: Optional[str] = None
predict_csv_import_error: Optional[str] = None

extract_features_from_csv = None
external_postprocess_fall_result = None
external_get_fall_probability = None

# DB 저장 제거. 화면 표시용 메모리만 사용.
SAVED_EVENTS: List[Dict[str, Any]] = []
RECENT_RESULTS: List[Dict[str, Any]] = []

SIMULATION_STATE: Dict[str, Any] = {
    "loaded": False,
    "file_name": None,
    "df": None,
    "cursor": 0,
    "fps": DEFAULT_FPS,
    "window_frames": DEFAULT_WINDOW_FRAMES,
    "total_steps": 0,
    "created_at": None,
    "history": [],
    "fall_confirmed": False,
}


# =========================================================
# 외부 RF 모듈 import
# =========================================================

try:
    from rf.feature_extractor import extract_features_from_csv as _extract_features_from_csv

    extract_features_from_csv = _extract_features_from_csv
    print("[FALL] rf.feature_extractor import 성공")

except Exception as e:
    feature_extractor_import_error = str(e)
    extract_features_from_csv = None
    print("[FALL] rf.feature_extractor import 실패")
    traceback.print_exc()


try:
    from rf.predict_csv import (
        postprocess_fall_result as _postprocess_fall_result,
        get_fall_probability as _get_fall_probability,
    )

    external_postprocess_fall_result = _postprocess_fall_result
    external_get_fall_probability = _get_fall_probability
    print("[FALL] rf.predict_csv import 성공")

except Exception as e:
    predict_csv_import_error = str(e)
    external_postprocess_fall_result = None
    external_get_fall_probability = None
    print("[FALL] rf.predict_csv import 실패 - 내부 fallback 사용")
    traceback.print_exc()


# =========================================================
# 공통 유틸
# =========================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(data):
    if isinstance(data, dict):
        return {str(key): json_safe(value) for key, value in data.items()}

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

    if isinstance(data, float):
        if math.isnan(data):
            return None
        return data

    if isinstance(data, Path):
        return str(data)

    return data


def clamp(value, min_value=0, max_value=100):
    try:
        value = float(value)
    except Exception:
        value = 0.0

    return max(min_value, min(max_value, value))


def round_float(value, digits=4):
    try:
        value = float(value)
        if math.isnan(value):
            return 0.0
        return round(value, digits)
    except Exception:
        return 0.0


def find_col(df: pd.DataFrame, candidates: List[str]):
    if df is None or len(df.columns) == 0:
        return None

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def numeric_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    col = find_col(df, candidates)

    if col is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(df[col], errors="coerce").dropna()


def text_value(df: pd.DataFrame, candidates: List[str], default="-"):
    col = find_col(df, candidates)

    if col is None or df is None or df.empty:
        return default

    values = df[col].dropna().astype(str)

    if values.empty:
        return default

    return values.iloc[-1]


def first_existing_path(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path

    return None


def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_dataframe_from_bytes(content: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    filename_lower = (filename or "").lower()

    if content is None or len(content) == 0:
        return pd.DataFrame()

    try:
        if filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls"):
            df = pd.read_excel(BytesIO(content))
        else:
            df = pd.read_csv(BytesIO(content))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except UnicodeDecodeError:
        try:
            text = content.decode("cp949", errors="ignore")
            df = pd.read_csv(StringIO(text))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} 파일을 읽을 수 없습니다: {exc}",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{filename} 파일을 읽을 수 없습니다: {exc}",
        )

    df.columns = [str(col).strip() for col in df.columns]
    return df


async def read_upload_file(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    return read_dataframe_from_bytes(content, file.filename or "upload.csv")


def dataframe_from_csv_text(text: str) -> pd.DataFrame:
    if text is None or str(text).strip() == "":
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(str(text)))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV 텍스트를 읽을 수 없습니다: {exc}")

    df.columns = [str(col).strip() for col in df.columns]
    return df


def dataframe_from_payload(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, list):
        df = pd.DataFrame(payload)

    elif isinstance(payload, dict):
        list_keys = [
            "data",
            "records",
            "rows",
            "chunk",
            "items",
            "sensors",
            "points",
            "frames",
        ]

        text_keys = [
            "csv_text",
            "csvText",
            "csv",
            "content",
            "file_content",
        ]

        for key in text_keys:
            if key in payload and isinstance(payload[key], str):
                return dataframe_from_csv_text(payload[key])

        for key in list_keys:
            if key in payload and isinstance(payload[key], list):
                df = pd.DataFrame(payload[key])
                break
        else:
            if isinstance(payload.get("sensor"), dict):
                df = pd.DataFrame([payload["sensor"]])
            elif isinstance(payload.get("payload"), dict):
                return dataframe_from_payload(payload["payload"])
            else:
                df = pd.DataFrame([payload])

    else:
        return pd.DataFrame()

    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_fall_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    mapping = {
        "frame": FRAME_COLS,
        "x": X_COLS,
        "y": Y_COLS,
        "z": Z_COLS,
        "v": V_COLS,
    }

    for target, candidates in mapping.items():
        if target not in df.columns:
            found = find_col(df, candidates)
            if found is not None:
                df[target] = df[found]

    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df))

    if "v" not in df.columns:
        df["v"] = 0.0

    return df


def can_analyze_fall_df(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False

    df = normalize_fall_columns(df)
    return all(col in df.columns for col in ["x", "y", "z"])


def validate_fall_dataframe(df: pd.DataFrame):
    if df is None:
        raise HTTPException(status_code=400, detail="CSV 데이터가 없습니다.")

    if df.empty:
        return

    df = normalize_fall_columns(df)

    missing = []

    for col in ["x", "y", "z"]:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "낙상 CSV에 필요한 컬럼이 없습니다.",
                "required": ["x", "y", "z"],
                "missing": missing,
                "received_columns": list(df.columns),
            },
        )


def dataframe_to_temp_csv(df: pd.DataFrame) -> Path:
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8-sig",
        newline="",
    )
    temp_path = Path(temp.name)

    try:
        df.to_csv(temp, index=False)
    finally:
        temp.close()

    return temp_path


# =========================================================
# 모델 로딩
# =========================================================

def load_model_and_meta():
    global model, model_meta, model_path, model_load_error

    model_load_error = None
    model_meta = {}
    model_path = None
    model = None

    selected_model_path = first_existing_path(MODEL_CANDIDATES)

    if selected_model_path is None:
        checked = "\n".join(str(path) for path in MODEL_CANDIDATES)
        model_load_error = f"모델 파일을 찾을 수 없습니다. 확인 경로:\n{checked}"
        print("[FALL] 모델 로드 실패")
        print(model_load_error)
        return

    try:
        loaded_model = joblib.load(selected_model_path)
        model = loaded_model
        model_path = selected_model_path

        selected_meta_path = first_existing_path(META_CANDIDATES)

        if selected_meta_path is not None:
            model_meta = read_json_file(selected_meta_path)
            print(f"[FALL] 메타 파일 사용: {selected_meta_path}")

        print(f"[FALL] 모델 파일 사용: {selected_model_path}")

    except Exception as e:
        model = None
        model_path = selected_model_path
        model_load_error = str(e)
        print("[FALL] 모델 로드 중 오류")
        traceback.print_exc()


def startup_fall_dashboard():
    load_model_and_meta()
    print("[FALL] FallDashboard startup 완료 - DB 저장 비활성화")


def shutdown_fall_dashboard():
    print("[FALL] FallDashboard shutdown 완료")


def check_model_ready() -> bool:
    return model is not None and model_load_error is None


def get_feature_names(feat: Dict[str, Any]) -> List[str]:
    if model_meta and "feature_names" in model_meta:
        return list(model_meta["feature_names"])

    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return sorted(feat.keys())


# =========================================================
# 특징 추출 / 행동 분석
# =========================================================

def center_points(df: pd.DataFrame):
    df = normalize_fall_columns(df)

    if df is None or df.empty:
        return pd.DataFrame(columns=["x", "y", "z"])

    if not all(col in df.columns for col in ["x", "y", "z"]):
        return pd.DataFrame(columns=["x", "y", "z"])

    temp = df[["frame", "x", "y", "z"]].copy()
    temp["frame"] = pd.to_numeric(temp["frame"], errors="coerce")
    temp["x"] = pd.to_numeric(temp["x"], errors="coerce")
    temp["y"] = pd.to_numeric(temp["y"], errors="coerce")
    temp["z"] = pd.to_numeric(temp["z"], errors="coerce")
    temp = temp.dropna()

    if temp.empty:
        return pd.DataFrame(columns=["x", "y", "z"])

    return temp.groupby("frame")[["x", "y", "z"]].mean().reset_index(drop=True)


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


def fallback_features_from_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    df = normalize_fall_columns(df)

    if df is None or df.empty:
        return {
            "speed_max": 0.0,
            "speed_mean": 0.0,
            "height_drop": 0.0,
            "movement_after": 0.0,
            "center_move": 0.0,
            "z_mean": 0.0,
            "z_min": 0.0,
            "z_max": 0.0,
            "row_count": 0,
        }

    x = numeric_series(df, ["x"])
    y = numeric_series(df, ["y"])
    z = numeric_series(df, ["z", "height"])
    v = numeric_series(df, ["v", "velocity", "speed"])

    speed_abs = v.abs() if not v.empty else pd.Series(dtype=float)

    speed_max = float(speed_abs.max()) if not speed_abs.empty else 0.0
    speed_mean = float(speed_abs.mean()) if not speed_abs.empty else 0.0

    z_min = float(z.min()) if not z.empty else 0.0
    z_max = float(z.max()) if not z.empty else 0.0
    z_mean = float(z.mean()) if not z.empty else 0.0
    height_drop = z_max - z_min if not z.empty else 0.0

    centers = center_points(df)
    center_move = 0.0

    if len(centers) >= 2:
        diffs = centers[["x", "y", "z"]].diff().dropna()
        movement = np.sqrt((diffs ** 2).sum(axis=1))
        center_move = float(movement.sum()) if len(movement) > 0 else 0.0

    movement_after = movement_after_value(df)

    return {
        "speed_max": round_float(speed_max),
        "speed_mean": round_float(speed_mean),
        "height_drop": round_float(height_drop),
        "movement_after": round_float(movement_after),
        "center_move": round_float(center_move),
        "z_mean": round_float(z_mean),
        "z_min": round_float(z_min),
        "z_max": round_float(z_max),
        "row_count": int(len(df)),
    }


def extract_features_safe(df: pd.DataFrame, file_name: str = "input.csv") -> Dict[str, Any]:
    df = normalize_fall_columns(df)

    if df is None or df.empty:
        return fallback_features_from_dataframe(df)

    if extract_features_from_csv is None:
        return fallback_features_from_dataframe(df)

    temp_path = dataframe_to_temp_csv(df)

    try:
        feat = extract_features_from_csv(temp_path)

        if not isinstance(feat, dict):
            return fallback_features_from_dataframe(df)

        fallback = fallback_features_from_dataframe(df)

        for key, value in fallback.items():
            if key not in feat:
                feat[key] = value

        return feat

    except Exception:
        print("[FALL] feature_extractor 실행 실패 - fallback features 사용")
        traceback.print_exc()
        return fallback_features_from_dataframe(df)

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def infer_fall_action(
    chunk: pd.DataFrame,
    height_drop: float,
    speed_max: float,
    movement_after: float,
    fall_already_detected: bool = False,
):
    behavior_label = text_value(chunk, TEXT_LABEL_COLS, default="-")
    scenario = text_value(chunk, SCENARIO_COLS, default="-")
    description = text_value(chunk, DESCRIPTION_COLS, default="-")

    text = f"{behavior_label} {scenario} {description}".lower()

    has_fall_word = (
        "fall_forward" in text
        or "fall_backward" in text
        or "fall_left" in text
        or "fall_right" in text
        or "fall_alert" in text
        or "fall alert" in text
        or "fall" in text
        or "낙상" in text
    )

    has_post_fall_word = (
        "post_fall" in text
        or "after_fall" in text
        or "fall_no_movement" in text
        or "낙상 후" in text
        or "낙상후" in text
    )

    has_real_fall_motion = (
        height_drop >= 0.45 and speed_max >= 0.60
    ) or (
        height_drop >= 0.65
    ) or (
        has_fall_word and height_drop >= 0.35 and speed_max >= 0.35
    )

    if has_post_fall_word and fall_already_detected:
        action = "낙상 후 무움직임"
        direction = "바닥에 머문 상태"
        cause_guess = "이전 구간에서 낙상이 감지된 뒤 움직임이 거의 없습니다."

    elif has_post_fall_word and not fall_already_detected:
        action = "무활동 상태"
        direction = "방향 정보 없음"
        cause_guess = "움직임은 적지만 앞선 낙상 감지가 없어 낙상 후 상태로 보지 않습니다."

    elif "fall_forward" in text and has_real_fall_motion:
        action = "걷다가 전방 낙상"
        direction = "전방 방향 추정"
        cause_guess = "이동 중 몸의 높이가 급격히 낮아지고 속도 변화가 크게 나타났습니다."

    elif "fall_backward" in text and has_real_fall_motion:
        action = "후방 낙상"
        direction = "후방 방향 추정"
        cause_guess = "몸이 뒤쪽으로 무너지며 높이 변화와 속도 변화가 함께 나타났습니다."

    elif "fall_left" in text and has_real_fall_motion:
        action = "좌측 낙상"
        direction = "좌측 방향 추정"
        cause_guess = "좌측으로 기울어진 낙상 패턴과 센서 변화가 함께 나타났습니다."

    elif "fall_right" in text and has_real_fall_motion:
        action = "우측 낙상"
        direction = "우측 방향 추정"
        cause_guess = "우측으로 기울어진 낙상 패턴과 센서 변화가 함께 나타났습니다."

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

    elif height_drop >= 0.70 and speed_max >= 0.80:
        action = "이동 중 낙상"
        direction = "전방 또는 측면 방향 추정"
        cause_guess = "높이 하강과 속도 변화가 동시에 나타났습니다."

    elif height_drop >= 0.45 and speed_max >= 0.60:
        action = "자세 변화 중 낙상 의심"
        direction = "하방 이동"
        cause_guess = "상체 높이 변화와 속도 변화가 함께 나타났습니다."

    elif movement_after <= 0.20 and speed_max <= 0.25 and height_drop < 0.30:
        action = "무활동 상태"
        direction = "방향 정보 없음"
        cause_guess = "움직임이 거의 없지만 낙상 전 높이 급감이나 속도 증가가 없어 낙상으로 판단하지 않습니다."

    else:
        action = "낙상 가능성 낮음"
        direction = "방향 정보 부족"
        cause_guess = "낙상 기준을 넘는 변화가 충분하지 않습니다."

    causes = []

    if height_drop >= 0.70:
        causes.append("높이가 급격히 낮아졌습니다.")
    elif height_drop >= 0.45:
        causes.append("상체 높이 변화가 크게 나타났습니다.")
    else:
        causes.append("높이 변화가 낙상 기준보다 낮습니다.")

    if speed_max >= 1.20:
        causes.append("순간 속도가 매우 크게 증가했습니다.")
    elif speed_max >= 0.80:
        causes.append("중간 수준 이상의 속도 변화가 있습니다.")
    elif speed_max >= 0.60:
        causes.append("낙상 후보로 볼 수 있는 속도 변화가 있습니다.")
    else:
        causes.append("속도 변화가 낙상 기준보다 낮습니다.")

    if movement_after <= 0.20 and has_real_fall_motion:
        causes.append("동작 이후 움직임이 거의 없어 낙상 후 움직임 감소로 볼 수 있습니다.")
    elif movement_after <= 0.20 and not has_real_fall_motion:
        causes.append("움직임은 적지만 낙상 전 높이 급감이나 속도 증가가 부족합니다.")
    elif movement_after <= 0.50 and has_real_fall_motion:
        causes.append("동작 이후 이동이 줄었습니다.")
    else:
        causes.append("동작 이후 움직임이 유지되어 낙상 후 정지 패턴은 약합니다.")

    if description != "-":
        causes.append(description)

    fall_cause = " ".join(causes)

    return action, direction, cause_guess, fall_cause, scenario, description


def make_behavior_analysis(
    chunk: pd.DataFrame,
    status: str,
    level: str,
    risk_score: int,
    speed_max: float,
    height_drop: float,
    movement_after: float,
    center_move: float,
    fall_already_detected: bool = False,
):
    action, direction, cause_guess, fall_cause, scenario, description = infer_fall_action(
        chunk=chunk,
        height_drop=height_drop,
        speed_max=speed_max,
        movement_after=movement_after,
        fall_already_detected=fall_already_detected,
    )

    is_danger = status == "Fall Alert" or level == "danger" or risk_score >= 70
    is_warning = level == "warning" or risk_score >= 40

    # 중요:
    # 모델 최종 판단이 Fall Alert인데 행동 케이스가 "낙상 가능성 낮음"으로 나오면
    # 화면에서 논리 충돌이 발생한다. 최종 위험도가 70% 이상이면 행동 분석도
    # 반드시 낙상 위험 패턴 기준으로 보정한다.
    low_action_texts = [
        "낙상 가능성 낮음",
        "행동 케이스 분석 없음",
        "데이터 없음",
        "보행",
        "무활동 상태",
    ]

    if is_danger and any(keyword in str(action) for keyword in low_action_texts):
        if speed_max >= 0.80 and movement_after <= 0.20:
            action = "속도 급증 후 움직임 감소 낙상 의심"
            cause_guess = (
                f"최종 낙상 위험도 {int(risk_score)}%로 기준선 70%를 넘었습니다. "
                "높이 변화는 크지 않지만 순간 속도 증가와 이후 움직임 감소가 함께 나타났습니다."
            )
        elif center_move >= 0.70 and speed_max >= 0.60:
            action = "이동 중 낙상 위험 패턴"
            cause_guess = (
                f"최종 낙상 위험도 {int(risk_score)}%로 기준선 70%를 넘었습니다. "
                "중심 이동량과 속도 변화가 크게 나타나 낙상 위험 패턴으로 판단했습니다."
            )
        elif height_drop >= 0.35:
            action = "자세 변화 중 낙상 위험 패턴"
            cause_guess = (
                f"최종 낙상 위험도 {int(risk_score)}%로 기준선 70%를 넘었습니다. "
                "높이 변화와 모델 확률이 함께 낙상 위험 구간으로 판단되었습니다."
            )
        else:
            action = "낙상 위험 패턴"
            cause_guess = (
                f"최종 낙상 위험도 {int(risk_score)}%로 기준선 70%를 넘었습니다. "
                "일부 물리 지표는 낮지만 모델이 전체 feature 조합을 낙상 위험으로 판단했습니다."
            )

        if not direction or direction == "-" or "부족" in str(direction):
            direction = "방향 정보 부족"

    causes = []

    if is_danger:
        causes.append(f"최종 낙상 위험도 {int(risk_score)}%로 Fall Alert 기준선 70% 이상입니다.")

        if height_drop >= 0.45:
            causes.append("높이 변화가 낙상 후보 기준에 도달했습니다.")
        else:
            causes.append("높이 변화는 크지 않지만 다른 지표와 모델 판단을 함께 반영했습니다.")

        if speed_max >= 1.20:
            causes.append("순간 속도가 매우 크게 증가했습니다.")
        elif speed_max >= 0.80:
            causes.append("중간 수준 이상의 속도 변화가 있습니다.")
        elif speed_max >= 0.60:
            causes.append("낙상 후보로 볼 수 있는 속도 변화가 있습니다.")
        else:
            causes.append("속도 변화는 크지 않아 다른 feature와 함께 판단했습니다.")

        if movement_after <= 0.20:
            causes.append("동작 이후 움직임이 적어 낙상 후 정지 패턴 가능성이 있습니다.")
        elif movement_after <= 0.50:
            causes.append("동작 이후 움직임이 감소했습니다.")
        else:
            causes.append("동작 이후 움직임은 유지되지만 모델 위험도가 높게 나타났습니다.")

        if center_move >= 0.70:
            causes.append("중심 이동량이 크게 나타났습니다.")

    else:
        if height_drop >= 0.70:
            causes.append("높이가 급격히 낮아졌습니다.")
        elif height_drop >= 0.45:
            causes.append("상체 높이 변화가 크게 나타났습니다.")
        else:
            causes.append("높이 변화가 낙상 기준보다 낮습니다.")

        if speed_max >= 1.20:
            causes.append("순간 속도가 매우 크게 증가했습니다.")
        elif speed_max >= 0.80:
            causes.append("중간 수준 이상의 속도 변화가 있습니다.")
        elif speed_max >= 0.60:
            causes.append("낙상 후보로 볼 수 있는 속도 변화가 있습니다.")
        else:
            causes.append("속도 변화가 낙상 기준보다 낮습니다.")

        if movement_after <= 0.20:
            causes.append("움직임은 적지만 낙상 전 높이 급감이나 속도 증가가 부족합니다.")
        elif movement_after <= 0.50:
            causes.append("동작 이후 이동이 줄었습니다.")
        else:
            causes.append("동작 이후 움직임이 유지되어 낙상 후 정지 패턴은 약합니다.")

    if description != "-":
        causes.append(description)

    fall_cause = " ".join(causes)

    if is_danger:
        summary = f"{action} 패턴이 감지되었습니다."
        recommendation = "관제 담당자는 즉시 대상자의 상태를 확인하고, 필요 시 보호자 연락 또는 응급 대응을 진행해야 합니다."
    elif is_warning:
        summary = f"{action} 패턴이 관찰되었지만 Fall Alert 기준은 넘지 않았습니다."
        recommendation = "추가 움직임 변화를 관찰하고, 반복적으로 발생하면 보호자 확인 또는 현장 확인이 필요합니다."
    else:
        summary = f"{action} 상태로 판단되며 낙상 위험은 낮습니다."
        recommendation = "즉시 대응보다는 모니터링을 유지하면 됩니다."

    judgement_basis_list = [
        f"최대 속도: {round_float(speed_max)}",
        f"높이 변화: {round_float(height_drop)}",
        f"이후 움직임: {round_float(movement_after)}",
        f"중심 이동량: {round_float(center_move)}",
        f"위험도: {int(risk_score)}%",
    ]

    judgement_basis = " / ".join(judgement_basis_list)

    analysis = {
        "summary": summary,
        "behavior_case": action,
        "action": action,
        "fall_action": action,
        "direction": direction,
        "fall_direction": direction,
        "cause": cause_guess,
        "cause_guess": cause_guess,
        "fall_cause": fall_cause,
        "estimated_cause": cause_guess,
        "judgement_basis": judgement_basis,
        "evidence": judgement_basis,
        "evidence_list": judgement_basis_list,
        "recommendation": recommendation,
        "response_recommendation": recommendation,
        "scenario": scenario,
        "description": description,
    }

    return analysis

# =========================================================
# 모델 예측
# =========================================================

def get_model_fall_probability(X: pd.DataFrame):
    if model is None:
        return 0.0, "Normal"

    if external_get_fall_probability is not None:
        try:
            output = external_get_fall_probability(model, X)

            if isinstance(output, tuple) or isinstance(output, list):
                if len(output) >= 2:
                    return float(output[0]), output[1]
                if len(output) == 1:
                    return float(output[0]), "Fall" if float(output[0]) >= 0.5 else "Normal"

            return float(output), "Fall" if float(output) >= 0.5 else "Normal"

        except Exception:
            pass

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", []))

            if 1 in classes:
                fall_index = classes.index(1)
            elif "Fall" in classes:
                fall_index = classes.index("Fall")
            elif "fall" in [str(c).lower() for c in classes]:
                lower_classes = [str(c).lower() for c in classes]
                fall_index = lower_classes.index("fall")
            else:
                fall_index = min(1, len(proba) - 1)

            fall_prob = float(proba[fall_index])
            pred = model.predict(X)[0] if hasattr(model, "predict") else int(fall_prob >= 0.5)

            return fall_prob, pred

        if hasattr(model, "predict"):
            pred = model.predict(X)[0]
            fall_prob = 1.0 if str(pred).lower() in ["1", "fall", "fall alert"] else 0.0
            return fall_prob, pred

    except Exception:
        traceback.print_exc()

    return 0.0, "Normal"


def build_model_dataframe(feat: Dict[str, Any]) -> pd.DataFrame:
    feature_names = get_feature_names(feat)
    X = pd.DataFrame([feat])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_names]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X


def local_postprocess_fall_result(
    fall_prob: float,
    features: Dict[str, Any],
    file_name: str = "input.csv",
):
    speed_max = float(features.get("speed_max", 0.0) or 0.0)

    height_drop = float(
        features.get("height_drop")
        or features.get("z_center_first_to_min_drop")
        or features.get("z_center_peak_to_last_drop")
        or 0.0
    )

    movement_after = float(features.get("movement_after", 0.0) or 0.0)
    center_move = float(features.get("center_move", 0.0) or 0.0)

    reasons = []

    is_alert = False

    if fall_prob >= 0.50 and height_drop >= 0.45:
        is_alert = True
        reasons.append("모델 확률과 높이 하강 기준을 함께 만족했습니다.")

    if height_drop >= 0.65 and movement_after <= 0.35:
        is_alert = True
        reasons.append("높이가 크게 낮아진 뒤 후반 움직임이 줄었습니다.")

    if height_drop >= 0.55 and speed_max >= 0.80:
        is_alert = True
        reasons.append("높이 하강과 순간 속도 증가가 동시에 나타났습니다.")

    if fall_prob >= 0.70:
        is_alert = True
        reasons.append("모델 낙상 확률이 높게 나타났습니다.")

    if is_alert:
        status = "Fall Alert"
        level = "danger"
        risk_score = int(clamp(max(fall_prob * 100, 75)))
        message = "낙상 위험 패턴이 감지되었습니다."
        alert = True

    elif fall_prob >= 0.35 or height_drop >= 0.35 or speed_max >= 0.50:
        status = "주의"
        level = "warning"
        risk_score = int(clamp(max(fall_prob * 100, 45)))
        message = "낙상과 유사한 움직임이 감지되었지만 Fall Alert 기준은 넘지 않았습니다."
        alert = False

    else:
        status = "Normal"
        level = "normal"
        risk_score = int(clamp(fall_prob * 100))
        message = "낙상 기준을 넘지 않았습니다."
        alert = False

    if not reasons:
        if status == "Normal":
            reasons.append("속도, 높이 변화, 후반 움직임 기준이 낙상 조건을 넘지 않았습니다.")
        else:
            reasons.append("일부 지표가 기준에 근접했습니다.")

    return {
        "status": status,
        "state": status,
        "level": level,
        "alert": alert,
        "fall_prob": round_float(fall_prob),
        "raw_fall_prob": round_float(fall_prob),
        "risk_score": risk_score,
        "message": message,
        "reason": " ".join(reasons),
        "file_name": file_name,
        "speed_max": round_float(speed_max),
        "height_drop": round_float(height_drop),
        "movement_after": round_float(movement_after),
        "center_move": round_float(center_move),
        "features": {
            **{
                key: round_float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
                for key, value in features.items()
            },
            "speed_max": round_float(speed_max),
            "height_drop": round_float(height_drop),
            "movement_after": round_float(movement_after),
            "center_move": round_float(center_move),
        },
    }


def postprocess_safe(fall_prob, features, file_name):
    if external_postprocess_fall_result is not None:
        try:
            sig = inspect.signature(external_postprocess_fall_result)
            params = sig.parameters

            if "features" in params:
                result = external_postprocess_fall_result(
                    fall_prob=fall_prob,
                    features=features,
                    file_name=file_name,
                )
            elif "feat" in params:
                result = external_postprocess_fall_result(
                    fall_prob=fall_prob,
                    feat=features,
                    file_name=file_name,
                )
            else:
                result = external_postprocess_fall_result(
                    fall_prob=fall_prob,
                    speed_max=features.get("speed_max", 0.0),
                    height_drop=features.get("height_drop", 0.0),
                    movement_after=features.get("movement_after", 0.0),
                )

            if isinstance(result, dict):
                local = local_postprocess_fall_result(fall_prob, features, file_name)

                merged = {
                    **local,
                    **result,
                }

                merged["features"] = {
                    **local.get("features", {}),
                    **result.get("features", {}),
                }

                if "alert" not in merged:
                    merged["alert"] = merged.get("status") == "Fall Alert"

                if "level" not in merged:
                    merged["level"] = "danger" if merged.get("alert") else "normal"

                if "risk_score" not in merged:
                    merged["risk_score"] = int(clamp(float(merged.get("fall_prob", fall_prob)) * 100))

                return merged

        except Exception:
            print("[FALL] 외부 postprocess 실패 - 내부 fallback 사용")
            traceback.print_exc()

    return local_postprocess_fall_result(fall_prob, features, file_name)


def make_empty_result(file_name: str, message: str = "분석할 낙상 데이터가 없습니다.") -> Dict[str, Any]:
    behavior_analysis = {
        "summary": "분석할 낙상 센서 데이터가 없습니다.",
        "behavior_case": "데이터 없음",
        "action": "데이터 없음",
        "fall_action": "데이터 없음",
        "direction": "-",
        "fall_direction": "-",
        "cause": message,
        "cause_guess": message,
        "fall_cause": message,
        "estimated_cause": message,
        "judgement_basis": "x, y, z 센서 데이터가 없습니다.",
        "evidence": "x, y, z 센서 데이터가 없습니다.",
        "evidence_list": ["x, y, z 센서 데이터가 없습니다."],
        "recommendation": "CSV 파일과 컬럼 구성을 확인하세요.",
        "response_recommendation": "CSV 파일과 컬럼 구성을 확인하세요.",
        "scenario": "-",
        "description": "-",
    }

    return {
        "success": True,
        "module": "fall",
        "title": "낙상 감지",
        "time": now_iso(),
        "file_name": file_name,
        "status": "Normal",
        "state": "Normal",
        "level": "normal",
        "alert": False,
        "fall_prob": 0,
        "raw_fall_prob": 0,
        "raw_model_fall_prob": 0,
        "risk_score": 0,
        "message": message,
        "reason": message,
        "speed_max": 0,
        "height_drop": 0,
        "movement_after": 0,
        "center_move": 0,
        "model_pred_label": "Normal",
        "model_ready": check_model_ready(),
        "model_path": str(model_path) if model_path else None,
        "model_error": model_load_error,
        "fall_action": behavior_analysis["fall_action"],
        "fall_direction": behavior_analysis["fall_direction"],
        "fall_cause": behavior_analysis["fall_cause"],
        "cause_guess": behavior_analysis["cause_guess"],
        "scenario": "-",
        "description": "-",
        "analysis_summary": behavior_analysis["summary"],
        "judgement_basis": behavior_analysis["judgement_basis"],
        "recommendation": behavior_analysis["recommendation"],
        "behavior_analysis": behavior_analysis,
        "fall_behavior_analysis": behavior_analysis,
        "pattern_analysis": behavior_analysis,
        "db_saved": False,
        "db_status": "disabled",
        "db_id": None,
        "features": {
            "fall_prob": 0,
            "speed_max": 0,
            "speed_mean": 0,
            "height_drop": 0,
            "movement_after": 0,
            "center_move": 0,
            "z_mean": 0,
            "z_min": 0,
            "z_max": 0,
            "row_count": 0,
        },
    }


def normalize_status_by_score(result: Dict[str, Any]) -> Dict[str, Any]:
    risk_score = int(clamp(result.get("risk_score", 0)))

    if risk_score >= 70:
        result["status"] = "Fall Alert"
        result["state"] = "Fall Alert"
        result["level"] = "danger"
        result["alert"] = True
    elif risk_score >= 40:
        result["status"] = "주의"
        result["state"] = "주의"
        result["level"] = "warning"
        result["alert"] = False
    else:
        result["status"] = "Normal"
        result["state"] = "Normal"
        result["level"] = "normal"
        result["alert"] = False

    result["risk_score"] = risk_score
    return result


def predict_dataframe(
    df: pd.DataFrame,
    file_name: str = "input.csv",
    fall_already_detected: bool = False,
) -> Dict[str, Any]:
    df = normalize_fall_columns(df)

    if not can_analyze_fall_df(df):
        return json_safe(
            make_empty_result(
                file_name,
                "해당 10프레임 구간에 낙상 분석용 x, y, z 데이터가 없습니다.",
            )
        )

    validate_fall_dataframe(df)

    features = extract_features_safe(df, file_name=file_name)

    raw_model_fall_prob = 0.0
    model_pred_label = "Normal"

    if check_model_ready():
        X = build_model_dataframe(features)
        raw_model_fall_prob, model_pred_label = get_model_fall_probability(X)
    else:
        speed_max = float(features.get("speed_max", 0.0) or 0.0)
        height_drop = float(features.get("height_drop", 0.0) or 0.0)
        movement_after = float(features.get("movement_after", 0.0) or 0.0)

        rule_prob = 0.0

        if height_drop >= 0.65 and speed_max >= 0.60:
            rule_prob = 0.80
        elif height_drop >= 0.45 and speed_max >= 0.60:
            rule_prob = 0.65
        elif height_drop >= 0.45 and movement_after <= 0.35:
            rule_prob = 0.55
        elif height_drop >= 0.35 or speed_max >= 0.60:
            rule_prob = 0.40

        raw_model_fall_prob = rule_prob
        model_pred_label = "Fall" if rule_prob >= 0.5 else "Normal"

    result = postprocess_safe(
        fall_prob=float(raw_model_fall_prob),
        features=features,
        file_name=file_name,
    )

    speed_max = round_float(result.get("speed_max", features.get("speed_max", 0.0)))
    height_drop = round_float(result.get("height_drop", features.get("height_drop", 0.0)))
    movement_after = round_float(result.get("movement_after", features.get("movement_after", 0.0)))
    center_move = round_float(result.get("center_move", features.get("center_move", 0.0)))

    fall_prob_percent = int(clamp(float(raw_model_fall_prob) * 100))
    current_risk_score = int(clamp(result.get("risk_score", fall_prob_percent)))

    # Fall Alert와 위험도 불일치 방지
    if result.get("status") == "Fall Alert" or result.get("alert") is True:
        current_risk_score = max(current_risk_score, 70)

    result["risk_score"] = current_risk_score
    result = normalize_status_by_score(result)

    status = result.get("status", "Normal")
    level = result.get("level", "normal")
    alert = bool(result.get("alert") is True)

    behavior_analysis = make_behavior_analysis(
        chunk=df,
        status=status,
        level=level,
        risk_score=current_risk_score,
        speed_max=speed_max,
        height_drop=height_drop,
        movement_after=movement_after,
        center_move=center_move,
        fall_already_detected=fall_already_detected,
    )

    result.update({
        "success": True,
        "module": "fall",
        "title": "낙상 감지",
        "time": now_iso(),
        "file_name": file_name,
        "status": status,
        "state": status,
        "level": level,
        "alert": alert,
        "fall_prob": round_float(result.get("fall_prob", raw_model_fall_prob)),
        "raw_fall_prob": round_float(result.get("raw_fall_prob", raw_model_fall_prob)),
        "raw_model_fall_prob": round_float(raw_model_fall_prob),
        "model_pred_label": str(model_pred_label),
        "speed_max": speed_max,
        "height_drop": height_drop,
        "movement_after": movement_after,
        "center_move": center_move,
        "model_ready": check_model_ready(),
        "model_path": str(model_path) if model_path else None,
        "model_error": model_load_error,

        # 프론트 행동 패턴 분석용 필드
        "fall_action": behavior_analysis["fall_action"],
        "fall_direction": behavior_analysis["fall_direction"],
        "fall_cause": behavior_analysis["fall_cause"],
        "cause_guess": behavior_analysis["cause_guess"],
        "scenario": behavior_analysis["scenario"],
        "description": behavior_analysis["description"],
        "analysis_summary": behavior_analysis["summary"],
        "judgement_basis": behavior_analysis["judgement_basis"],
        "evidence": behavior_analysis["evidence"],
        "recommendation": behavior_analysis["recommendation"],
        "response_recommendation": behavior_analysis["response_recommendation"],
        "behavior_case": behavior_analysis["behavior_case"],
        "behavior_analysis": behavior_analysis,
        "fall_behavior_analysis": behavior_analysis,
        "pattern_analysis": behavior_analysis,

        # DB 저장 제거
        "db_saved": False,
        "db_status": "disabled",
        "db_id": None,
    })

    if "message" not in result or not result["message"]:
        result["message"] = "낙상 분석이 완료되었습니다."

    if "reason" not in result or not result["reason"]:
        result["reason"] = result["message"]

    return json_safe(result)


# =========================================================
# 메모리 로그
# =========================================================

def remember_result_for_screen(result: Dict[str, Any]):
    """
    실시간 10프레임 결과까지 화면에서 참고할 수 있도록 최근 결과만 메모리에 보관한다.
    여기서는 Fall Alert 로그(SAVED_EVENTS)에 넣지 않는다.
    """
    item = {
        "_id": str(uuid4()),
        "saved_at": now_iso(),
        "db_status": "memory_only",
        "result": result,
    }

    RECENT_RESULTS.insert(0, item)

    if len(RECENT_RESULTS) > 150:
        del RECENT_RESULTS[150:]

    return item


def save_event_if_needed(result: Dict[str, Any], save_event: bool = True):
    """
    DB 저장은 하지 않는다.
    save_event=True이고 최종 결과가 Fall Alert일 때만 최근 낙상 알림 로그(SAVED_EVENTS)에 저장한다.
    save_event=False인 10프레임 중간 분석은 로그에 쌓지 않는다.
    """
    remember_result_for_screen(result)

    if not save_event or not result.get("alert"):
        return {
            "saved": False,
            "event_saved": False,
            "db_status": "disabled",
            "id": None,
        }

    item = {
        "_id": str(uuid4()),
        "saved_at": now_iso(),
        "db_status": "memory_only",
        "result": result,
    }

    SAVED_EVENTS.insert(0, item)

    if len(SAVED_EVENTS) > 150:
        del SAVED_EVENTS[150:]

    return {
        "saved": False,
        "event_saved": True,
        "db_status": "memory_only",
        "id": item["_id"],
    }


# =========================================================
# 요청 파싱
# =========================================================

async def parse_predict_request(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()

        upload_file = None
        upload_key = None

        for key, value in form.multi_items():
            if isinstance(value, (UploadFile, StarletteUploadFile)) or hasattr(value, "filename"):
                upload_file = value
                upload_key = key
                break

        if upload_file is not None:
            df = await read_upload_file(upload_file)

            return {
                "df": df,
                "file_name": upload_file.filename or f"{upload_key}.csv",
                "source": f"multipart:{upload_key}",
            }

        for key, value in form.multi_items():
            if isinstance(value, str):
                text = value.strip()

                if not text:
                    continue

                if text.startswith("{") or text.startswith("["):
                    try:
                        payload = json.loads(text)
                        df = dataframe_from_payload(payload)

                        return {
                            "df": df,
                            "file_name": f"{key}.json",
                            "source": f"multipart-json:{key}",
                        }
                    except Exception:
                        pass

                if "," in text and ("\n" in text or "x" in text.lower()):
                    df = dataframe_from_csv_text(text)

                    return {
                        "df": df,
                        "file_name": f"{key}.csv",
                        "source": f"multipart-csv-text:{key}",
                    }

        form_dict = {
            key: value
            for key, value in form.multi_items()
            if isinstance(value, str)
        }

        if form_dict:
            df = dataframe_from_payload(form_dict)

            return {
                "df": df,
                "file_name": "form_input.csv",
                "source": "form-fields",
            }

        return {
            "df": pd.DataFrame(),
            "file_name": "empty_form.csv",
            "source": "empty-form",
        }

    try:
        payload = await request.json()
    except Exception:
        return {
            "df": pd.DataFrame(),
            "file_name": "empty_body.csv",
            "source": "empty-body",
        }

    df = dataframe_from_payload(payload)

    return {
        "df": df,
        "file_name": payload.get("file_name", "json_input.csv") if isinstance(payload, dict) else "json_input.csv",
        "source": "json",
    }


# =========================================================
# 시뮬레이션 유틸
# =========================================================

def get_frame_range(df: pd.DataFrame):
    if df is None or df.empty:
        return 0, 0, 0

    frame_col = find_col(df, FRAME_COLS)

    if frame_col:
        frames = pd.to_numeric(df[frame_col], errors="coerce").dropna()

        if frames.empty:
            return 0, len(df) - 1, len(df)

        min_frame = int(frames.min())
        max_frame = int(frames.max())
        count = max_frame - min_frame + 1

        return min_frame, max_frame, count

    return 0, len(df) - 1, len(df)


def calculate_total_steps(df: pd.DataFrame, window_frames: int) -> int:
    if df is None or df.empty:
        return 0

    _, _, frame_count = get_frame_range(df)

    if frame_count <= 0:
        frame_count = len(df)

    return max(1, math.ceil(frame_count / window_frames))


def slice_window(df: pd.DataFrame, cursor: int, window_frames: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    frame_col = find_col(df, FRAME_COLS)
    start_offset = cursor * window_frames

    if frame_col:
        frames = pd.to_numeric(df[frame_col], errors="coerce")
        valid_frames = frames.dropna()

        if valid_frames.empty:
            start_row = start_offset
            end_row = start_row + window_frames
            return df.iloc[start_row:end_row].copy()

        min_frame = int(valid_frames.min())
        start_frame = min_frame + start_offset
        end_frame = start_frame + window_frames

        return df[(frames >= start_frame) & (frames < end_frame)].copy()

    start_row = start_offset
    end_row = start_row + window_frames
    return df.iloc[start_row:end_row].copy()


def simulation_status():
    df = SIMULATION_STATE.get("df")
    total_steps = int(SIMULATION_STATE.get("total_steps", 0))
    cursor = int(SIMULATION_STATE.get("cursor", 0))
    fps = int(SIMULATION_STATE.get("fps", DEFAULT_FPS))
    window_frames = int(SIMULATION_STATE.get("window_frames", DEFAULT_WINDOW_FRAMES))

    current_seconds = round((cursor * window_frames) / fps, 2) if fps else 0
    total_seconds = round((total_steps * window_frames) / fps, 2) if fps else 0

    return {
        "loaded": bool(SIMULATION_STATE.get("loaded")),
        "file_name": SIMULATION_STATE.get("file_name"),
        "fps": fps,
        "window_frames": window_frames,
        "current_step": cursor,
        "total_steps": total_steps,
        "current_index": cursor,
        "total_rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "remaining_steps": max(0, total_steps - cursor),
        "current_seconds": current_seconds,
        "total_seconds": total_seconds,
        "done": cursor >= total_steps if total_steps else True,
        "created_at": SIMULATION_STATE.get("created_at"),
    }


# =========================================================
# API - Health
# =========================================================

@router.get("/fall/health")
def fall_health():
    return {
        "status": "ok",
        "service": "fall-dashboard",
        "time": now_iso(),
        "mode": "fall_with_behavior_analysis_no_db",
        "db_enabled": False,
        "model_ready": check_model_ready(),
        "model_path": str(model_path) if model_path else None,
        "model_error": model_load_error,
        "feature_extractor_loaded": extract_features_from_csv is not None,
        "feature_extractor_error": feature_extractor_import_error,
        "predict_csv_loaded": external_get_fall_probability is not None,
        "predict_csv_error": predict_csv_import_error,
        "supported_paths": {
            "health": ["/fall/health"],
            "stats": ["/stats", "/fall/stats"],
            "events": ["/events", "/fall/events"],
            "predict": ["/predict", "/fall/predict"],
            "predict_file": ["/predict-file", "/fall/predict-file"],
            "simulation_upload": ["/simulation/upload", "/fall/simulation/upload"],
            "simulation_status": ["/simulation/status", "/fall/simulation/status"],
            "simulation_reset": ["/simulation/reset", "/fall/simulation/reset"],
            "simulation_next": ["/simulation/next", "/fall/simulation/next"],
            "evaluate": ["/evaluate", "/fall/evaluate"],
        },
    }


# =========================================================
# API - Stats / Events
# =========================================================

def make_stats():
    # 보호자 알림/최근 낙상 로그는 최종 집계 저장 이벤트 기준으로 계산한다.
    # 10프레임 중간 분석은 RECENT_RESULTS에만 남고 SAVED_EVENTS에는 들어가지 않는다.
    fall_count = len(SAVED_EVENTS)

    warning_count = sum(
        1
        for item in RECENT_RESULTS
        if item.get("result", {}).get("level") == "warning"
    )

    normal_count = sum(
        1
        for item in RECENT_RESULTS
        if item.get("result", {}).get("level") == "normal"
    )

    return {
        "success": True,
        "time": now_iso(),
        "mode": "fall_with_behavior_analysis_no_db",
        "db_connected": False,
        "db_enabled": False,
        "total_events": len(SAVED_EVENTS),
        "total_results": len(RECENT_RESULTS),
        "fall_alert_count": fall_count,
        "normal_count": normal_count,
        "warning_count": warning_count,
        "simulation": simulation_status(),
        "model_ready": check_model_ready(),
        "model_path": str(model_path) if model_path else None,
        "model_error": model_load_error,
    }


@router.get("/stats")
@router.get("/fall/stats")
def get_stats():
    return make_stats()


@router.get("/events")
@router.get("/fall/events")
def get_events(limit: int = 50):
    limit = max(1, min(int(limit), 200))

    return {
        "success": True,
        "db_connected": False,
        "db_enabled": False,
        "db_status": "disabled",
        "items": json_safe(SAVED_EVENTS[:limit]),
        "events": json_safe(SAVED_EVENTS[:limit]),
        "count": len(SAVED_EVENTS[:limit]),
    }


@router.delete("/events")
@router.delete("/fall/events")
def clear_events():
    event_count = len(SAVED_EVENTS)
    result_count = len(RECENT_RESULTS)

    SAVED_EVENTS.clear()
    RECENT_RESULTS.clear()

    return {
        "success": True,
        "message": "메모리 낙상 로그가 삭제되었습니다. DB는 사용하지 않습니다.",
        "db_enabled": False,
        "deleted_count": 0,
        "memory_deleted_count": event_count,
        "memory_result_deleted_count": result_count,
    }


# =========================================================
# API - Predict
# =========================================================

@router.post("/predict")
@router.post("/fall/predict")
async def predict(
    request: Request,
    save_event: bool = Query(False),
):
    parsed = await parse_predict_request(request)

    result = predict_dataframe(
        parsed["df"],
        file_name=parsed["file_name"],
        fall_already_detected=False,
    )

    save_result = save_event_if_needed(result, save_event=save_event)

    result["db_saved"] = save_result["saved"]
    result["db_status"] = save_result["db_status"]
    result["db_id"] = save_result["id"]
    result["event_saved"] = save_result.get("event_saved", False)
    result["source"] = parsed["source"]
    result["save_event_requested"] = save_event

    return result


@router.post("/predict-file")
@router.post("/fall/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    save_event: bool = Query(False),
):
    df = await read_upload_file(file)

    result = predict_dataframe(
        df,
        file_name=file.filename or "upload.csv",
        fall_already_detected=False,
    )

    save_result = save_event_if_needed(result, save_event=save_event)

    result["db_saved"] = save_result["saved"]
    result["db_status"] = save_result["db_status"]
    result["db_id"] = save_result["id"]
    result["event_saved"] = save_result.get("event_saved", False)
    result["source"] = "file"
    result["save_event_requested"] = save_event

    return result


# =========================================================
# API - Simulation
# =========================================================

@router.post("/simulation/upload")
@router.post("/fall/simulation/upload")
async def upload_simulation_csv(
    file: Optional[UploadFile] = File(None),
    csv: Optional[UploadFile] = File(None),
    fall_csv: Optional[UploadFile] = File(None),
    fps: int = Form(DEFAULT_FPS),
    window_frames: int = Form(DEFAULT_WINDOW_FRAMES),
):
    upload = file or csv or fall_csv

    if upload is None:
        raise HTTPException(status_code=400, detail="업로드할 CSV 파일이 없습니다.")

    if fps <= 0:
        raise HTTPException(status_code=400, detail="fps는 1 이상이어야 합니다.")

    if window_frames <= 0:
        raise HTTPException(status_code=400, detail="window_frames는 1 이상이어야 합니다.")

    df = await read_upload_file(upload)
    df = normalize_fall_columns(df)

    if not can_analyze_fall_df(df):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "낙상 분석용 CSV가 아닙니다. x, y, z 컬럼이 필요합니다.",
                "received_columns": list(df.columns),
            },
        )

    total_steps = calculate_total_steps(df, window_frames)

    SIMULATION_STATE["loaded"] = True
    SIMULATION_STATE["file_name"] = upload.filename
    SIMULATION_STATE["df"] = df
    SIMULATION_STATE["cursor"] = 0
    SIMULATION_STATE["fps"] = fps
    SIMULATION_STATE["window_frames"] = window_frames
    SIMULATION_STATE["total_steps"] = total_steps
    SIMULATION_STATE["created_at"] = now_iso()
    SIMULATION_STATE["history"] = []
    SIMULATION_STATE["fall_confirmed"] = False

    return {
        "success": True,
        "message": "낙상 CSV 업로드가 완료되었습니다.",
        "db_enabled": False,
        "status": simulation_status(),
        "history": [],
    }


@router.get("/simulation/status")
@router.get("/fall/simulation/status")
def get_simulation_status():
    return {
        "success": True,
        "db_enabled": False,
        "status": simulation_status(),
        "history": SIMULATION_STATE.get("history", [])[:120],
    }


@router.post("/simulation/reset")
@router.post("/fall/simulation/reset")
def reset_simulation():
    SIMULATION_STATE["cursor"] = 0
    SIMULATION_STATE["history"] = []
    SIMULATION_STATE["fall_confirmed"] = False

    return {
        "success": True,
        "message": "낙상 시뮬레이션이 초기화되었습니다.",
        "db_enabled": False,
        "status": simulation_status(),
        "history": [],
    }


def run_simulation_next():
    if not SIMULATION_STATE.get("loaded") or SIMULATION_STATE.get("df") is None:
        raise HTTPException(status_code=400, detail="먼저 CSV 파일을 업로드해야 합니다.")

    df = SIMULATION_STATE["df"]
    cursor = int(SIMULATION_STATE["cursor"])
    total_steps = int(SIMULATION_STATE["total_steps"])
    fps = int(SIMULATION_STATE["fps"])
    window_frames = int(SIMULATION_STATE["window_frames"])

    if cursor >= total_steps:
        return {
            "success": True,
            "done": True,
            "message": "시뮬레이션이 종료되었습니다.",
            "db_enabled": False,
            "status": simulation_status(),
            "history": SIMULATION_STATE.get("history", [])[:120],
            "result": None,
        }

    chunk = slice_window(df, cursor, window_frames)
    current_seconds = round((cursor * window_frames) / fps, 2)

    result = predict_dataframe(
        chunk,
        file_name=f"{SIMULATION_STATE.get('file_name') or 'simulation.csv'}#{cursor + 1}",
        fall_already_detected=SIMULATION_STATE.get("fall_confirmed", False),
    )

    result["step"] = cursor + 1
    result["second"] = current_seconds
    result["window"] = {
        "fps": fps,
        "window_frames": window_frames,
        "window_seconds": round(window_frames / fps, 2),
    }

    result["db_saved"] = False
    result["db_status"] = "disabled"
    result["db_id"] = None

    remember_result_for_screen(result)

    if result.get("alert"):
        SIMULATION_STATE["fall_confirmed"] = True

    SIMULATION_STATE["history"].insert(0, result)
    SIMULATION_STATE["cursor"] = cursor + 1

    done = SIMULATION_STATE["cursor"] >= total_steps

    return {
        "success": True,
        "done": done,
        "db_enabled": False,
        "result": result,
        "status": simulation_status(),
        "history": SIMULATION_STATE.get("history", [])[:120],
    }


@router.get("/simulation/next")
@router.get("/fall/simulation/next")
def simulation_next_get():
    return run_simulation_next()


@router.post("/simulation/next")
@router.post("/fall/simulation/next")
def simulation_next_post():
    return run_simulation_next()


# =========================================================
# API - Evaluate
# =========================================================

@router.get("/evaluate")
@router.get("/fall/evaluate")
def evaluate_dataset(limit: int = 0):
    if not FALL_DIR.exists() or not NOT_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"평가 데이터 폴더가 없습니다: {DATA_DIR}",
        )

    fall_files = sorted(FALL_DIR.glob("*.csv"))
    not_files = sorted(NOT_DIR.glob("*.csv"))

    dataset = []

    for path in fall_files:
        dataset.append((path, 1, "Fall"))

    for path in not_files:
        dataset.append((path, 0, "Normal"))

    if limit and limit > 0:
        dataset = dataset[:limit]

    rows = []
    y_true = []
    y_pred = []

    for path, true_label, true_name in dataset:
        try:
            df = pd.read_csv(path)
            df.columns = [str(col).strip() for col in df.columns]
            df = normalize_fall_columns(df)

            result = predict_dataframe(df, file_name=path.name)

            pred_label = 1 if result.get("alert") is True else 0
            pred_name = "Fall" if pred_label == 1 else "Normal"

            y_true.append(true_label)
            y_pred.append(pred_label)

            rows.append({
                "file_name": path.name,
                "true_folder": true_name,
                "true_label": true_label,
                "pred_status": result.get("status"),
                "pred_alert": result.get("alert"),
                "pred_label": pred_label,
                "pred_name": pred_name,
                "correct": true_label == pred_label,
                "fall_prob": result.get("fall_prob"),
                "raw_model_fall_prob": result.get("raw_model_fall_prob"),
                "risk_score": result.get("risk_score"),
                "speed_max": result.get("speed_max"),
                "height_drop": result.get("height_drop"),
                "movement_after": result.get("movement_after"),
                "fall_action": result.get("fall_action"),
                "fall_direction": result.get("fall_direction"),
                "fall_cause": result.get("fall_cause"),
                "cause_guess": result.get("cause_guess"),
                "message": result.get("message"),
                "file_path": str(path),
            })

        except Exception as e:
            y_true.append(true_label)
            y_pred.append(0)

            rows.append({
                "file_name": path.name,
                "true_folder": true_name,
                "true_label": true_label,
                "pred_status": "Error",
                "pred_alert": False,
                "pred_label": 0,
                "pred_name": "Error",
                "correct": False,
                "fall_prob": None,
                "raw_model_fall_prob": None,
                "risk_score": None,
                "speed_max": None,
                "height_drop": None,
                "movement_after": None,
                "fall_action": "-",
                "fall_direction": "-",
                "fall_cause": "-",
                "cause_guess": "-",
                "message": str(e),
                "file_path": str(path),
            })

    result_df = pd.DataFrame(rows)

    total = len(result_df)
    correct_count = int(result_df["correct"].sum()) if total else 0
    wrong_count = total - correct_count
    accuracy = round(correct_count / total, 4) if total else 0

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "fall_eval_results_no_db_behavior.csv"
    result_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    wrong_items = result_df[result_df["correct"] == False].to_dict("records") if total else []

    return {
        "success": True,
        "mode": "fall_with_behavior_analysis_no_db",
        "db_enabled": False,
        "total": total,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "accuracy": accuracy,
        "fall_count": len(fall_files),
        "normal_count": len(not_files),
        "wrong_items": json_safe(wrong_items),
        "report_path": str(report_path),
    }