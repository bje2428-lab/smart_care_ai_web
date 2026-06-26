from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional
from uuid import uuid4
import math
import os
import traceback

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile


router = APIRouter(prefix="/integrated", tags=["Integrated Dashboard"])

SESSIONS: Dict[str, Dict[str, Any]] = {}

DEFAULT_FPS = 10
DEFAULT_FALL_WINDOW_FRAMES = 10
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


# =========================================================
# vital_signal.py 실제 모델 연동
# =========================================================

try:
    import vital_signal as vital_module
    VITAL_MODULE_IMPORT_ERROR = None
except Exception as e:
    vital_module = None
    VITAL_MODULE_IMPORT_ERROR = str(e)


# =========================================================
# MongoDB optional 연결
# =========================================================

SAVED_LOGS = []

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


def find_col(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None

    lower_map = {str(col).strip().lower(): col for col in df.columns}

    for name in candidates:
        key = str(name).strip().lower()
        if key in lower_map:
            return lower_map[key]

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

    return values.iloc[-1]


def mean_value(df: pd.DataFrame, candidates):
    series = numeric_series(df, candidates)

    if series.empty:
        return None

    return float(series.mean())


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
            df = pd.read_csv(BytesIO(content))
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

    df.columns = [str(col).strip() for col in df.columns]
    return df


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
        count = max_frame - min_frame + 1

        return min_frame, max_frame, count

    return 0, len(df) - 1, len(df)


def total_steps_by_window(fall_df, abnormal_df, vital_df, fps, window_frames):
    total_frame_count = 0

    for df in [fall_df, abnormal_df]:
        if df is not None and not df.empty:
            _, _, frame_count = get_frame_range(df)
            total_frame_count = max(total_frame_count, frame_count)

    if vital_df is not None and not vital_df.empty:
        # 바이탈은 0.003초 단위 샘플로 들어온다고 보고,
        # 통합 화면의 1초 구간 안에서 약 333개 샘플을 묶어 작게 표시한다.
        window_seconds = window_frames / fps
        time_col = find_col(vital_df, RAW_VITAL_TIME_COLS)

        if time_col:
            times = pd.to_numeric(vital_df[time_col], errors="coerce").dropna()

            if not times.empty:
                min_time = float(times.min())
                max_time = float(times.max())
                duration = max(
                    VITAL_SAMPLE_INTERVAL_SECONDS,
                    (max_time - min_time) + VITAL_SAMPLE_INTERVAL_SECONDS,
                )
                vital_steps = max(1, math.ceil(duration / window_seconds))
                total_frame_count = max(total_frame_count, vital_steps * window_frames)

        else:
            second_col = find_col(
                vital_df,
                ["second", "seconds", "sec", "time_sec", "time_seconds"],
            )

            if second_col:
                seconds = pd.to_numeric(vital_df[second_col], errors="coerce").dropna()

                if not seconds.empty:
                    duration = float(seconds.max()) + window_seconds
                    vital_steps = max(1, math.ceil(duration / window_seconds))
                    total_frame_count = max(total_frame_count, vital_steps * window_frames)
            else:
                duration = len(vital_df) * VITAL_SAMPLE_INTERVAL_SECONDS
                vital_steps = max(1, math.ceil(duration / window_seconds))
                total_frame_count = max(total_frame_count, vital_steps * window_frames)

    if total_frame_count <= 0:
        total_frame_count = fps

    return max(1, math.ceil(total_frame_count / window_frames))


def slice_by_frame_window(df, step, fps, window_frames, mode):
    if df is None or df.empty:
        return pd.DataFrame()

    current_frame_offset = step * window_frames
    current_second = current_frame_offset / fps
    window_seconds = window_frames / fps

    if mode == "vital":
        time_col = find_col(df, RAW_VITAL_TIME_COLS)

        if time_col:
            times = pd.to_numeric(df[time_col], errors="coerce")
            valid_times = times.dropna()

            if valid_times.empty:
                return pd.DataFrame()

            min_time = float(valid_times.min())
            start_time = min_time + current_second
            end_time = start_time + window_seconds

            return df[(times >= start_time) & (times < end_time)].copy()

    frame_col = find_col(df, ["frame", "Frame", "frame_id", "frameId"])

    if frame_col:
        frames = pd.to_numeric(df[frame_col], errors="coerce")
        min_frame = frames.min()

        start = min_frame + current_frame_offset
        end = start + window_frames

        return df[(frames >= start) & (frames < end)].copy()

    second_col = find_col(df, ["second", "seconds", "sec", "time_sec", "time_seconds"])

    if second_col:
        seconds = pd.to_numeric(df[second_col], errors="coerce")
        target_second = math.floor(current_second)
        return df[seconds == target_second].copy()

    if mode == "vital":
        # 시간 컬럼이 없는 바이탈 CSV는 행 하나를 0.003초 샘플로 간주한다.
        # 현재 통합 구간(기본 1초)에 해당하는 여러 샘플을 묶어서 분석한다.
        start_row = int(math.floor(current_second / VITAL_SAMPLE_INTERVAL_SECONDS))
        end_row = int(math.ceil((current_second + window_seconds) / VITAL_SAMPLE_INTERVAL_SECONDS))
        chunk = df.iloc[start_row:end_row].copy()

        if not chunk.empty:
            generated_times = np.arange(start_row, start_row + len(chunk), dtype=float)
            chunk[VITAL_GENERATED_TIME_COL] = np.round(
                generated_times * VITAL_SAMPLE_INTERVAL_SECONDS,
                6,
            )

        return chunk

    start = current_frame_offset
    end = start + window_frames

    return df.iloc[start:end].copy()


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


# =========================================================
# 낙상 분석
# =========================================================

def infer_fall_action(
    chunk,
    height_drop,
    speed_max,
    movement_after,
    fall_already_detected=False,
):
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

    has_fall_word = (
        "fall_forward" in text
        or "fall_backward" in text
        or "fall_left" in text
        or "fall_right" in text
        or "fall_alert" in text
        or "fall alert" in text
        or "낙상" in text
    )

    has_post_fall_word = (
        "post_fall" in text
        or "after_fall" in text
        or "fall_no_movement" in text
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

    elif movement_after <= 0.2 and speed_max <= 0.25 and height_drop < 0.30:
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

    if speed_max >= 1.2:
        causes.append("순간 속도가 매우 크게 증가했습니다.")
    elif speed_max >= 0.8:
        causes.append("중간 수준 이상의 속도 변화가 있습니다.")

    if movement_after <= 0.2 and has_real_fall_motion:
        causes.append("동작 이후 움직임이 거의 없어 낙상 후 움직임 감소로 볼 수 있습니다.")
    elif movement_after <= 0.2 and not has_real_fall_motion:
        causes.append("움직임은 적지만 낙상 전 높이 급감이나 속도 증가가 부족합니다.")
    elif movement_after <= 0.5 and has_real_fall_motion:
        causes.append("동작 이후 이동이 줄었습니다.")

    if description != "-":
        causes.append(description)

    if not causes:
        causes.append("낙상 기준을 넘는 변화는 크지 않습니다.")

    return action, direction, cause_guess, " ".join(causes), scenario, description

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
            },
        }

    v = numeric_series(chunk, ["v", "velocity", "speed"])
    z = numeric_series(chunk, ["z", "height"])

    speed_max = float(v.abs().max()) if not v.empty else 0.0
    speed_mean = float(v.abs().mean()) if not v.empty else 0.0

    z_mean = float(z.mean()) if not z.empty else 0.0
    z_min = float(z.min()) if not z.empty else 0.0
    z_max = float(z.max()) if not z.empty else 0.0
    height_drop = float(z_max - z_min) if not z.empty else 0.0

    movement_after = movement_after_value(chunk)

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

    has_fall_word = (
        "fall_forward" in text
        or "fall_backward" in text
        or "fall_left" in text
        or "fall_right" in text
        or "fall_alert" in text
        or "fall alert" in text
        or "낙상" in text
    )

    has_post_fall_word = (
        "post_fall" in text
        or "after_fall" in text
        or "fall_no_movement" in text
        or "낙상 후" in text
        or "낙상후" in text
    )

    height_trigger = height_drop >= 0.45
    speed_trigger = speed_max >= 0.60
    strong_height_trigger = height_drop >= 0.65

    real_fall_motion = (
        height_trigger and speed_trigger
    ) or (
        strong_height_trigger
    ) or (
        has_fall_word and height_drop >= 0.35 and speed_max >= 0.35
    )

    # 움직임이 거의 없는 것만으로는 낙상이 아님.
    only_no_movement = (
        movement_after <= 0.2
        and speed_max <= 0.25
        and height_drop < 0.30
        and not has_fall_word
    )

    score = 0

    if not only_no_movement:
        if height_drop >= 0.70:
            score += 45
        elif height_drop >= 0.55:
            score += 36
        elif height_drop >= 0.40:
            score += 24
        elif height_drop >= 0.25:
            score += 10

        if speed_max >= 1.20:
            score += 35
        elif speed_max >= 0.90:
            score += 28
        elif speed_max >= 0.60:
            score += 16
        elif speed_max >= 0.40:
            score += 6

        # 후반 움직임 감소는 낙상 핵심 조건이 있을 때만 가산.
        if real_fall_motion:
            if movement_after <= 0.20:
                score += 20
            elif movement_after <= 0.40:
                score += 13
            elif movement_after <= 0.60:
                score += 6

    # 낙상 이후 무움직임은 이전 구간에서 실제 Fall Alert가 난 뒤에만 위험 유지.
    if has_post_fall_word and fall_already_detected:
        score = max(score, 80)

    # post_fall 라벨이 있어도 이전 낙상이 없고 센서 변화도 약하면 정상/낮은 점수 처리.
    if has_post_fall_word and not fall_already_detected and not real_fall_motion:
        score = min(score, 25)

    # 낙상 라벨 단어가 있어도 센서 변화가 너무 약하면 Fall Alert로 올리지 않음.
    if has_fall_word and not real_fall_motion:
        score = min(score, 35)

    risk_score = int(clamp(score))

    fall_action, fall_direction, cause_guess, fall_cause, scenario, description = infer_fall_action(
        chunk,
        height_drop,
        speed_max,
        movement_after,
        fall_already_detected=fall_already_detected,
    )

    if risk_score >= 70 and (real_fall_motion or fall_already_detected):
        state = "Fall Alert"
        level = "danger"
        reason = "10프레임, 즉 1초 구간에서 낙상 패턴이 감지되었습니다."

    elif risk_score >= 40:
        state = "주의"
        level = "warning"
        reason = "낙상과 유사한 움직임이 감지되었지만 Fall Alert 기준은 넘지 않았습니다."

    else:
        state = "Normal"
        level = "normal"

        if only_no_movement or (has_post_fall_word and not fall_already_detected):
            reason = "움직임은 적지만 낙상 전 높이 급감이나 속도 증가가 없어 낙상으로 판단하지 않습니다."
        else:
            reason = "낙상 기준을 넘지 않았습니다."

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
            "fall_prob": round(risk_score / 100, 4),
            "speed_max": round(speed_max, 4),
            "speed_mean": round(speed_mean, 4),
            "height_drop": round(height_drop, 4),
            "movement_after": round(movement_after, 4),
            "z_mean": round(z_mean, 4),
            "z_min": round(z_min, 4),
            "z_max": round(z_max, 4),
        },
    }


# =========================================================
# 이상행동 분석
# =========================================================

def risk_score_by_state(state: str):
    text = str(state or "").lower()

    if (
        "fall" in text
        or "post_fall" in text
        or "낙상" in text
        or "no_movement" in text
    ):
        return 35

    if "위험" in text or "danger" in text or "emergency" in text:
        return 80

    if "배회" in text or "wandering" in text:
        return 72

    if "무활동" in text or "inactive" in text:
        return 65

    if "주의" in text or "warning" in text or "abnormal" in text or "이상" in text:
        return 60

    if "빠르게" in text or "fast" in text:
        return 55

    if "외출" in text or "outing" in text:
        return 38

    if "식사" in text or "meal" in text:
        return 30

    if "수면" in text or "sleep" in text or "rest" in text:
        return 22

    return 18


def classify_abnormal_type(state, behavior, speed_max, movement_after):
    text = f"{state} {behavior}".lower()

    if "fall_forward" in text or "post_fall" in text or "낙상" in text:
        return "낙상 관련 상태", "낙상 감지 결과에 포함되는 상태이므로 이상행동 위험도로 중복 계산하지 않습니다."

    if "wandering" in text or "배회" in text:
        return "배회 행동", "같은 공간에서 반복적인 이동 패턴이 나타났습니다."

    if "inactive" in text or "무활동" in text or "no_movement" in text:
        return "무활동 주의", "움직임이 거의 없는 상태입니다."

    if "sit_fast" in text or "빠르게" in text or "fast" in text:
        return "급격한 자세 변화", "빠르게 앉거나 자세가 급변한 행동으로 판단됩니다."

    if "sleep" in text or "rest" in text or "수면" in text:
        return "휴식 또는 수면", "휴식 상태로 판단되며 위험도는 낮습니다."

    if speed_max >= 0.8 and movement_after >= 0.35:
        return "활동량 증가", "평소보다 빠른 움직임이 감지되었습니다."

    if speed_max <= 0.05 and movement_after <= 0.03:
        return "무활동 주의", "움직임이 거의 없는 상태입니다."

    return "정상 활동", "이상행동 기준을 넘지 않았습니다."


def predict_abnormal_chunk(chunk: pd.DataFrame):
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
            "guardian_message": "데이터 대기 중입니다.",
            "heart_rate": None,
            "respiratory_rate": None,
            "temperature": None,
            "features": {},
        }

    state = text_value(
        chunk,
        ["state", "label", "activity", "actual_state", "class", "target"],
        default="Normal",
    )

    behavior = text_value(
        chunk,
        ["behavior_label", "behavior", "action", "activity"],
        default=state,
    )

    description = text_value(
        chunk,
        ["description", "desc", "actual_reason", "reason"],
        default="-",
    )

    heart_rate = mean_value(
        chunk,
        ["heart_rate", "heartRate", "hr", "bpm"],
    )

    respiratory_rate = mean_value(
        chunk,
        ["respiratory_rate", "respiratoryRate", "rr", "breath_rate", "breathRate"],
    )

    temperature = mean_value(
        chunk,
        ["temperature", "temp", "body_temp"],
    )

    v = numeric_series(chunk, ["v", "velocity", "speed"])
    speed_max = float(v.abs().max()) if not v.empty else 0.0
    movement_after = movement_after_value(chunk)

    abnormal_type, detail = classify_abnormal_type(
        state,
        behavior,
        speed_max,
        movement_after,
    )

    risk_score = risk_score_by_state(f"{state} {behavior} {abnormal_type}")

    if risk_score >= 80:
        level = "danger"
    elif risk_score >= 50:
        level = "warning"
    else:
        level = "normal"

    if description != "-":
        detail = f"{detail} {description}"

    guardian_alert = risk_score >= 70

    if guardian_alert:
        guardian_message = "주의 이상의 이상행동이 감지되어 보호자 확인이 필요합니다."
    else:
        guardian_message = "보호자 알림 없이 관제 화면에만 기록됩니다."

    return {
        "module": "abnormal",
        "title": "이상행동",
        "state": state,
        "level": level,
        "risk_score": risk_score,
        "reason": detail,
        "behavior": behavior,
        "abnormal_type": abnormal_type,
        "detail": detail,
        "guardian_alert": guardian_alert,
        "guardian_message": guardian_message,
        "heart_rate": round(heart_rate, 1) if heart_rate is not None else None,
        "respiratory_rate": round(respiratory_rate, 1) if respiratory_rate is not None else None,
        "temperature": round(temperature, 1) if temperature is not None else None,
        "features": {
            "speed_max": round(speed_max, 4),
            "movement_after": round(movement_after, 4),
        },
    }


# =========================================================
# 바이탈 분석
# 핵심 수정
# - 전처리 완료 CSV(mean, std, peak_to_peak, zero_crossings, fft_mean, fft_max, fft_std) 우선 지원
# - 원본 CSV(Time_Seconds, VitalSignal, Condition)도 들어오면 1초 chunk에서 자동 feature 추출
# - 즉, 통합서비스에서는 feature CSV / 원본 CSV 둘 다 받을 수 있음
# =========================================================

FEATURE_COLUMN_ALIASES = {
    "mean": ["mean", "signal_mean", "vital_mean"],
    "std": ["std", "signal_std", "vital_std"],
    "peak_to_peak": ["peak_to_peak", "peak2peak", "p2p", "ptp", "range"],
    "zero_crossings": ["zero_crossings", "zero_crossing", "zc"],
    "fft_mean": ["fft_mean", "fft_avg"],
    "fft_max": ["fft_max", "fft_peak"],
    "fft_std": ["fft_std"],
}


def get_column_by_alias(df: pd.DataFrame, aliases):
    if df is None or df.empty:
        return None

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for alias in aliases:
        key = str(alias).strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def extract_preprocessed_vital_matrix(chunk: pd.DataFrame):
    """
    전처리 완료 바이탈 CSV를 모델 입력 matrix로 변환.

    허용 컬럼:
    mean, std, peak_to_peak, zero_crossings, fft_mean, fft_max, fft_std

    한 구간 안에 여러 행이 있으면 행 단위로 모두 모델에 넣고,
    화면 표시용 features는 평균값을 사용한다.
    """
    if chunk is None or chunk.empty:
        return None, None

    selected_cols = []

    for feature_name in VITAL_FEATURE_NAMES:
        col = get_column_by_alias(
            chunk,
            FEATURE_COLUMN_ALIASES.get(feature_name, [feature_name]),
        )

        if col is None:
            return None, None

        selected_cols.append(col)

    feature_df = chunk[selected_cols].copy()
    feature_df.columns = VITAL_FEATURE_NAMES
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.dropna(how="all")

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
        second_col = find_col(chunk, ["second", "seconds", "sec", "time_sec", "time_seconds"])

        if second_col is not None:
            seconds = pd.to_numeric(chunk[second_col], errors="coerce").dropna()

            if not seconds.empty:
                return {
                    "time_start": round(float(seconds.min()), 3),
                    "time_end": round(float(seconds.max()), 3),
                    "sample_count": int(len(seconds)),
                }

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
        return {
            name: 0.0
            for name in VITAL_FEATURE_NAMES
        }

    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]

    if len(signal) == 0:
        return {
            name: 0.0
            for name in VITAL_FEATURE_NAMES
        }

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    peak_to_peak = float(np.max(signal) - np.min(signal))

    if len(signal) >= 2:
        signs = np.sign(signal)
        zero_crossings = int(np.sum(np.diff(signs) != 0))
    else:
        zero_crossings = 0

    if len(signal) >= 2:
        fft_values = np.abs(np.fft.rfft(signal))
        fft_mean = float(np.mean(fft_values))
        fft_max = float(np.max(fft_values))
        fft_std = float(np.std(fft_values))
    else:
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


def feature_dict_to_matrix(features: dict):
    return np.array(
        [[float(features.get(name, 0.0)) for name in VITAL_FEATURE_NAMES]],
        dtype=float,
    )


def matrix_to_feature_mean(matrix):
    if matrix is None or len(matrix) == 0:
        return {
            name: 0.0
            for name in VITAL_FEATURE_NAMES
        }

    arr = np.asarray(matrix, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    means = np.mean(arr, axis=0)

    return {
        name: round(float(means[index]), 6)
        for index, name in enumerate(VITAL_FEATURE_NAMES)
    }


def get_vital_condition(chunk: pd.DataFrame):
    return text_value(
        chunk,
        RAW_VITAL_CONDITION_COLS,
        default="-",
    )


def fallback_vital_result_from_features(
    chunk: pd.DataFrame,
    features: dict,
    reason_prefix: str,
    model_mode: str,
):
    """
    AutoEncoder 모델이 없거나 실패했을 때 사용하는 rule-based fallback.
    원본/전처리 CSV 모두 같은 features 기준으로 판단한다.
    """
    condition = get_vital_condition(chunk)
    condition_text = str(condition or "").lower()

    peak_to_peak = float(features.get("peak_to_peak", 0))
    std = float(features.get("std", 0))
    fft_max = float(features.get("fft_max", 0))

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
        "features": features,
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


def predict_vital_with_autoencoder(
    chunk: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_mean: dict,
    model_mode: str,
    default_reason: str,
):
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
            "current_segment_count": segment.get("current_segment_count", 0),
            "max_segment_count": segment.get("max_segment_count", 0),
            "required_continuous_segments": segment.get("required_continuous_segments", 0),
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
            "title": "바이탈",
            "state": "대기",
            "level": "idle",
            "risk_score": 0,
            "reason": "해당 구간의 바이탈 데이터가 없습니다.",
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
            "features": {
                name: 0.0
                for name in VITAL_FEATURE_NAMES
            },
        }

    # 1순위: 전처리 완료 feature CSV
    # mean, std, peak_to_peak, zero_crossings, fft_mean, fft_max, fft_std 컬럼이 있으면
    # 원본 신호로 다시 전처리하지 않고 그대로 모델 입력으로 사용한다.
    feature_matrix, feature_mean = extract_preprocessed_vital_matrix(chunk)

    if feature_matrix is not None:
        return predict_vital_with_autoencoder(
            chunk=chunk,
            feature_matrix=feature_matrix,
            feature_mean=feature_mean,
            model_mode="feature_csv_autoencoder",
            default_reason="전처리 완료 바이탈 feature CSV를 AutoEncoder로 분석했습니다.",
        )

    # 2순위: 원본 VitalSignal CSV
    # Time_Seconds, VitalSignal, Condition 형식이면 현재 1초 chunk에서 feature를 계산한다.
    signal = extract_raw_vital_signal(chunk)

    if signal is not None:
        raw_features = compute_raw_vital_features(signal)
        raw_matrix = feature_dict_to_matrix(raw_features)

        return predict_vital_with_autoencoder(
            chunk=chunk,
            feature_matrix=raw_matrix,
            feature_mean=raw_features,
            model_mode="raw_signal_autoencoder",
            default_reason="VitalSignal 원시 신호를 1초 단위 feature로 변환한 뒤 AutoEncoder로 분석했습니다.",
        )

    # 둘 다 없으면 바이탈 분석 불가
    empty_features = {
        name: 0.0
        for name in VITAL_FEATURE_NAMES
    }

    return {
        "module": "vital",
        "title": "바이탈",
        "state": "신호 없음",
        "level": "idle",
        "risk_score": 0,
        "reason": (
            "바이탈 CSV에 전처리 feature 컬럼"
            "(mean, std, peak_to_peak, zero_crossings, fft_mean, fft_max, fft_std)"
            " 또는 원본 VitalSignal 컬럼이 없습니다."
        ),
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
        "features": empty_features,
    }


# =========================================================
# 종합 / 최종 결론 / 저장
# =========================================================

def make_current_overall(fall_result, abnormal_result, vital_result):
    # 바이탈은 통합 화면에서 작은 참고 정보로만 표시하고,
    # 종합 위험도/실시간 알림 판단에는 반영하지 않는다.
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


def make_final_summary(history):
    if not history:
        return {
            "level": "idle",
            "label": "대기",
            "risk_score": 0,
            "message": "아직 실행된 결과가 없습니다.",
            "fall_detected": False,
            "fall_second": None,
            "fall_action": "-",
            "fall_direction": "-",
            "fall_cause": "-",
            "cause_guess": "-",
            "abnormal_detected": False,
            "abnormal_type": "-",
            "vital_detected": False,
            "saved": False,
        }

    ordered = sorted(history, key=lambda x: x.get("second", 0))

    fall_events = [
        item for item in ordered
        if item.get("fall", {}).get("level") == "danger"
    ]

    abnormal_events = [
        item for item in ordered
        if item.get("abnormal", {}).get("level") in ["danger", "warning"]
    ]

    vital_events = [
        item for item in ordered
        if item.get("vital", {}).get("level") in ["danger", "warning"]
    ]

    max_risk = max(
        int(item.get("overall", {}).get("risk_score", 0))
        for item in ordered
    )

    if fall_events:
        best_fall = max(
            fall_events,
            key=lambda x: x.get("fall", {}).get("risk_score", 0),
        )

        fall = best_fall["fall"]
        fall_score = int(fall.get("risk_score", 0))

        return {
            "level": "danger",
            "label": "낙상 발생",
            "risk_score": fall_score,
            "message": f"{best_fall.get('second')}초 구간에서 낙상이 감지되었습니다.",
            "fall_detected": True,
            "fall_second": best_fall.get("second"),
            "fall_action": fall.get("fall_action", "-"),
            "fall_direction": fall.get("fall_direction", "-"),
            "fall_cause": fall.get("fall_cause", "-"),
            "cause_guess": fall.get("cause_guess", "-"),
            "abnormal_detected": len(abnormal_events) > 0,
            "abnormal_type": abnormal_events[0]["abnormal"].get("abnormal_type", "-") if abnormal_events else "-",
            "vital_detected": len(vital_events) > 0,
            "saved": True,
        }

    if abnormal_events:
        first_abnormal = abnormal_events[0]
        abnormal = first_abnormal["abnormal"]

        return {
            "level": "warning",
            "label": "이상행동 감지",
            "risk_score": abnormal.get("risk_score", max_risk),
            "message": f"{first_abnormal.get('second')}초 구간에서 이상행동이 감지되었습니다.",
            "fall_detected": False,
            "fall_second": None,
            "fall_action": "-",
            "fall_direction": "-",
            "fall_cause": "-",
            "cause_guess": "-",
            "abnormal_detected": True,
            "abnormal_type": abnormal.get("abnormal_type", "-"),
            "vital_detected": len(vital_events) > 0,
            "saved": True,
        }

    # 바이탈 이벤트는 작은 참고 정보로만 사용하므로
    # 최종 종합 결론을 "바이탈 주의"로 바꾸거나 DB 저장 대상으로 만들지 않는다.
    return {
        "level": "normal",
        "label": "정상",
        "risk_score": max_risk,
        "message": "전체 구간에서 낙상 또는 위험 이벤트가 감지되지 않았습니다.",
        "fall_detected": False,
        "fall_second": None,
        "fall_action": "-",
        "fall_direction": "-",
        "fall_cause": "-",
        "cause_guess": "-",
        "abnormal_detected": False,
        "abnormal_type": "-",
        "vital_detected": False,
        "saved": False,
    }


def should_save_event(result, final_summary=None):
    if result.get("fall", {}).get("level") == "danger":
        return True

    if result.get("abnormal", {}).get("risk_score", 0) >= 70:
        return True

    # 바이탈은 DB 저장 대상에서 제외한다. 낙상/이상행동만 저장한다.
    if final_summary and final_summary.get("level") in ["danger", "warning"]:
        return True

    return False


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

    current_step = session["cursor"]
    fps = session["fps"]
    window_frames = session["fall_window_frames"]

    current_seconds = round((current_step * window_frames) / fps, 2)
    total_seconds = round((session["total_steps"] * window_frames) / fps, 2)

    return {
        "session_id": session_id,
        "fps": fps,
        "fall_window_frames": window_frames,
        "vital_sample_interval_seconds": VITAL_SAMPLE_INTERVAL_SECONDS,
        "current_step": current_step,
        "total_steps": session["total_steps"],
        "current_seconds": current_seconds,
        "total_seconds": total_seconds,
        "done": session["cursor"] >= session["total_steps"],
        "files": session["files"],
        "created_at": session["created_at"],
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

    return {
        "status": "ok",
        "service": "integrated-dashboard",
        "time": now_iso(),
        "db_connected": mongo_collection is not None,
        "db_error": mongo_error,
        "vital_module_loaded": vital_module is not None,
        "vital_module_import_error": VITAL_MODULE_IMPORT_ERROR,
        "vital_model_ready": vital_ready,
        "vital_sample_interval_seconds": VITAL_SAMPLE_INTERVAL_SECONDS,
        "raw_vital_columns": {
            "time": RAW_VITAL_TIME_COLS,
            "signal": RAW_VITAL_SIGNAL_COLS,
            "condition": RAW_VITAL_CONDITION_COLS,
        },
        "vital_feature_names": VITAL_FEATURE_NAMES,
    }


@router.post("/simulation/upload")
async def upload_integrated_csv(
    fall_csv: Optional[UploadFile] = File(None),
    abnormal_csv: Optional[UploadFile] = File(None),
    vital_csv: Optional[UploadFile] = File(None),
    fps: int = Form(DEFAULT_FPS),
    fall_window_frames: int = Form(DEFAULT_FALL_WINDOW_FRAMES),
):
    if not fall_csv and not abnormal_csv and not vital_csv:
        raise HTTPException(status_code=400, detail="최소 1개 이상의 CSV 파일을 업로드해야 합니다.")

    if fps <= 0:
        raise HTTPException(status_code=400, detail="fps는 1 이상이어야 합니다.")

    if fall_window_frames <= 0:
        raise HTTPException(status_code=400, detail="fall_window_frames는 1 이상이어야 합니다.")

    fall_df = await read_upload_file(fall_csv) if fall_csv else None
    abnormal_df = await read_upload_file(abnormal_csv) if abnormal_csv else None
    vital_df = await read_upload_file(vital_csv) if vital_csv else None

    total_steps = total_steps_by_window(
        fall_df=fall_df,
        abnormal_df=abnormal_df,
        vital_df=vital_df,
        fps=fps,
        window_frames=fall_window_frames,
    )

    session_id = str(uuid4())

    SESSIONS[session_id] = {
        "fps": fps,
        "fall_window_frames": fall_window_frames,
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
        "saved_event_keys": set(),
        "final_saved": False,
        "fall_confirmed": False,
    }

    return {
        "message": "통합 CSV 업로드가 완료되었습니다.",
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
        "history": session["history"][:120],
        "final_summary": make_final_summary(session["history"]),
    }


@router.post("/simulation/{session_id}/reset")
def reset_session(session_id: str):
    session = SESSIONS.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="통합 시뮬레이션 세션을 찾을 수 없습니다.")

    session["cursor"] = 0
    session["history"] = []
    session["saved_event_keys"] = set()
    session["final_saved"] = False
    session["fall_confirmed"] = False

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

    if cursor >= total_steps:
        final_summary = make_final_summary(session["history"])

        if should_save_event({}, final_summary) and not session["final_saved"]:
            save_event_to_db(
                "integrated_final_summary",
                {
                    "session_id": session_id,
                    "final_summary": final_summary,
                    "status": session_status(session_id),
                },
            )
            session["final_saved"] = True

        return {
            "done": True,
            "status": session_status(session_id),
            "history": session["history"][:120],
            "final_summary": final_summary,
            "message": "시뮬레이션이 종료되었습니다.",
        }

    fall_df = session["data"]["fall"]
    abnormal_df = session["data"]["abnormal"]
    vital_df = session["data"]["vital"]

    fall_chunk = slice_by_frame_window(fall_df, cursor, fps, window_frames, "fall")
    abnormal_chunk = slice_by_frame_window(abnormal_df, cursor, fps, window_frames, "abnormal")
    vital_chunk = slice_by_frame_window(vital_df, cursor, fps, window_frames, "vital")

    current_seconds = round((cursor * window_frames) / fps, 2)

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
        "window": {
            "fps": fps,
            "fall_window_frames": window_frames,
            "window_seconds": round(window_frames / fps, 2),
        },
        "overall": current_overall,
        "fall": fall_result,
        "abnormal": abnormal_result,
        "vital": vital_result,
        "db_saved": False,
        "db_status": "none",
    }

    event_key = f"{cursor}:{fall_result['level']}:{abnormal_result['level']}"

    if should_save_event(result) and event_key not in session["saved_event_keys"]:
        save_result = save_event_to_db(
            "integrated_realtime_event",
            {
                "session_id": session_id,
                "result": result,
            },
        )

        result["db_saved"] = save_result["saved"]
        result["db_status"] = save_result["db_status"]
        result["db_id"] = save_result["id"]

        session["saved_event_keys"].add(event_key)

    session["history"].insert(0, result)
    session["cursor"] += 1

    final_summary = make_final_summary(session["history"])
    done = session["cursor"] >= total_steps

    if done and should_save_event({}, final_summary) and not session["final_saved"]:
        save_event_to_db(
            "integrated_final_summary",
            {
                "session_id": session_id,
                "final_summary": final_summary,
                "status": session_status(session_id),
            },
        )
        session["final_saved"] = True

    return {
        "done": done,
        "result": result,
        "status": session_status(session_id),
        "history": session["history"][:120],
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

    # =========================================================
# main.py 호환용 함수
# main.py에서 startup_fall_dashboard, shutdown_fall_dashboard를 import하므로
# FallDashboard.py에 없으면 import 에러가 발생한다.
# =========================================================

async def startup_fall_dashboard():
    """
    FastAPI 서버 시작 시 낙상 대시보드 초기화용 함수.
    현재 FallDashboard에서 별도 시작 작업이 없으면 그대로 통과한다.
    """
    try:
        print("[STARTUP] FallDashboard 시작 완료")
        return True
    except Exception as e:
        print(f"[STARTUP] FallDashboard 시작 중 오류: {e}")
        return False


async def shutdown_fall_dashboard():
    """
    FastAPI 서버 종료 시 낙상 대시보드 정리용 함수.
    현재 별도 종료 작업이 없으면 그대로 통과한다.
    """
    try:
        print("[SHUTDOWN] FallDashboard 종료 완료")
        return True
    except Exception as e:
        print(f"[SHUTDOWN] FallDashboard 종료 중 오류: {e}")
        return False