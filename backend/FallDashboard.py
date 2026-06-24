from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Query

try:
    from pymongo import MongoClient, DESCENDING
except Exception:
    MongoClient = None
    DESCENDING = -1


# =========================================================
# 경로 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
RF_DIR = ROOT_DIR / "rf"
MODEL_DIR = ROOT_DIR / "models"

for path in [ROOT_DIR, RF_DIR, BACKEND_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:
    from rf.feature_extractor import extract_features_from_csv
    from rf.predict_csv import postprocess_fall_result, get_fall_probability
except ModuleNotFoundError:
    from feature_extractor import extract_features_from_csv
    from predict_csv import postprocess_fall_result, get_fall_probability


router = APIRouter(tags=["Fall Detection"])


# =========================================================
# 기본 설정
# =========================================================

ASSUMED_FPS = 20
REALTIME_FRAME_WINDOW = 10
REALTIME_WINDOW_SECONDS = REALTIME_FRAME_WINDOW / ASSUMED_FPS

# 화면과 판정 기준 통일: 70% 이상이면 Fall Alert
ALERT_THRESHOLD = 0.70

MODEL_CANDIDATES = [
    MODEL_DIR / "mmwave_rf_smote_model.pkl",
    MODEL_DIR / "mmwave_rf_model.pkl",
    MODEL_DIR / "mmwave_rf_aug_model.pkl",
]

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "smart_care_ai")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "detection_events")

mongo_client = None
events_collection = None

_cached_model = None
_cached_meta = None
_cached_model_path = None


# =========================================================
# 유틸
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


def safe_round(value, ndigits=4):
    return round(to_float(value), ndigits)


def pick_number(*sources, keys=None, default: float = 0.0) -> float:
    keys = keys or []

    for source in sources:
        if not isinstance(source, dict):
            continue

        for key in keys:
            if key in source and source.get(key) is not None:
                return to_float(source.get(key), default)

    return default


def get_csv_frame_meta(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = ["frame", "x", "y", "z", "v"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"CSV 필수 컬럼이 없습니다. 필요한 컬럼: {required_cols}, 누락: {missing}"
        )

    if df.empty:
        raise ValueError("CSV 파일에 데이터가 없습니다.")

    frames = pd.to_numeric(df["frame"], errors="coerce").dropna()

    if frames.empty:
        frame_start = 0
        frame_end = 0
        frame_count = 0
    else:
        unique_frames = sorted(frames.astype(int).unique().tolist())
        frame_start = int(unique_frames[0])
        frame_end = int(unique_frames[-1])
        frame_count = len(unique_frames)

    point_count = int(len(df))

    det_obj_values = []
    if "DetObj#" in df.columns:
        det_obj_values = (
            pd.to_numeric(df["DetObj#"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_count,
        "point_count": point_count,
        "det_obj_values": det_obj_values,
        "assumed_fps": ASSUMED_FPS,
        "realtime_frame_window": REALTIME_FRAME_WINDOW,
        "realtime_window_seconds": REALTIME_WINDOW_SECONDS,
    }


def split_csv_by_frame_window(csv_path: Path, window_size: int = REALTIME_FRAME_WINDOW):
    """
    전체 CSV를 frame 번호 기준으로 10프레임씩 나눈다.
    같은 frame에 여러 row가 있을 수 있으므로 row 기준이 아니라 frame 기준으로 자른다.
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    if "frame" not in df.columns:
        raise ValueError("CSV에 frame 컬럼이 없습니다.")

    frames = pd.to_numeric(df["frame"], errors="coerce")
    df = df.loc[frames.notna()].copy()
    df["frame"] = frames.loc[frames.notna()].astype(int)

    unique_frames = sorted(df["frame"].unique().tolist())

    chunks = []

    for idx, start in enumerate(range(0, len(unique_frames), window_size)):
        frame_group = unique_frames[start:start + window_size]
        if not frame_group:
            continue

        chunk_df = df[df["frame"].isin(frame_group)].copy()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = Path(tmp.name)
        tmp.close()

        chunk_df.to_csv(tmp_path, index=False)

        chunks.append(
            {
                "chunk_index": idx,
                "tmp_path": tmp_path,
                "frame_start": int(frame_group[0]),
                "frame_end": int(frame_group[-1]),
                "frame_count": len(frame_group),
                "point_count": int(len(chunk_df)),
            }
        )

    return chunks


def normalize_status(result: dict, fall_prob: float) -> str:
    """
    프론트에는 Fall Alert / Normal만 보냄.
    기준 통일:
    - fall_prob >= 0.70이면 Fall Alert
    - postprocess가 Fall Alert라고 해도 Fall Alert
    - Exception은 Normal
    """
    if not isinstance(result, dict):
        result = {}

    raw_status = str(result.get("status", "")).strip()
    message = str(result.get("message", "")).strip()

    if raw_status == "Exception":
        return "Normal"

    if raw_status == "Fall Alert":
        return "Fall Alert"

    if result.get("alert") is True:
        return "Fall Alert"

    if "Fall Alert" in message or "낙상 알림" in message or "알림 발생" in message:
        return "Fall Alert"

    if fall_prob >= ALERT_THRESHOLD:
        return "Fall Alert"

    return "Normal"


def build_sensor_feature_table(feat: dict) -> list:
    descriptions = {
        "x": ("좌우 위치", "레이더 기준 대상의 좌우 위치 변화"),
        "y": ("전후 거리", "레이더와 대상 사이의 전후 거리 변화"),
        "z": ("높이", "대상의 높이 변화. 낙상 시 크게 낮아질 수 있음"),
        "v": ("속도", "대상의 움직임 속도"),
    }

    rows = []

    for key, (label, meaning) in descriptions.items():
        rows.append(
            {
                "name": key,
                "label": label,
                "meaning": meaning,
                "mean": safe_round(feat.get(f"{key}_mean")),
                "min": safe_round(feat.get(f"{key}_min")),
                "max": safe_round(feat.get(f"{key}_max")),
                "range": safe_round(feat.get(f"{key}_range")),
            }
        )

    return rows


def infer_behavior_case(file_name: str, result: dict, all_windows=None) -> dict:
    """
    행동 케이스 / 행동 패턴 / 원인 분석

    핵심 원칙:
    - Normal이면 절대 '낙상 가능성', '균형 상실', '미끄러짐' 같은 표현을 쓰지 않음
    - Fall Alert일 때만 낙상 케이스를 추정함
    - 위험도 70% 미만은 정상 또는 주의 행동으로 설명함
    """
    name = str(file_name or "").lower()
    all_windows = all_windows or []

    status = str(result.get("status", "Normal"))
    fall_prob = to_float(result.get("fall_prob"))
    speed_max = to_float(result.get("speed_max"))
    height_drop = to_float(result.get("height_drop"))
    movement_after = to_float(result.get("movement_after"))

    fall_percent = round(fall_prob * 100)

    is_fall_alert = status == "Fall Alert" and fall_prob >= ALERT_THRESHOLD

    # =====================================================
    # 1. 정상 데이터 처리
    # =====================================================
    if not is_fall_alert:
        if fall_prob < 0.4:
            action_case = "normal_movement"
            action_label = "정상 행동"
            behavior_pattern = (
                "낙상 기준선에 도달하지 않은 정상 움직임 패턴입니다. "
                "속도 변화와 높이 변화가 낙상으로 판단될 만큼 크지 않습니다."
            )
            likely_cause = (
                "일상적인 움직임, 자세 유지, 작은 센서 변화 또는 일반적인 활동으로 판단됩니다."
            )
            analysis_summary = (
                f"정상 행동으로 판단됩니다. 낙상 위험도는 {fall_percent}%로 "
                "Fall Alert 기준인 70%보다 낮습니다."
            )

        elif fall_prob < ALERT_THRESHOLD:
            action_case = "normal_or_attention_movement"
            action_label = "정상 또는 주의 행동"
            behavior_pattern = (
                "일부 움직임 변화는 감지되었지만 낙상 기준선에는 도달하지 않았습니다. "
                "앉기, 일어서기, 방향 전환, 물건 줍기 같은 일반 행동일 수 있습니다."
            )
            likely_cause = (
                "일상 동작 중 순간적인 자세 변화가 있었지만 낙상으로 볼 만큼의 "
                "위험 패턴은 부족합니다."
            )
            analysis_summary = (
                f"주의가 필요한 움직임은 있으나 최종 판단은 정상입니다. "
                f"낙상 위험도는 {fall_percent}%이며 Fall Alert 기준 70% 미만입니다."
            )

        else:
            action_case = "normal_by_rule"
            action_label = "정상 판단"
            behavior_pattern = (
                "모델 위험도는 일부 높게 나왔지만 후처리 기준상 낙상 알림으로 확정되지 않았습니다."
            )
            likely_cause = (
                "낙상과 유사한 자세 변화가 있었지만 이후 움직임이나 속도 조건이 "
                "Fall Alert 기준을 충분히 만족하지 않았을 수 있습니다."
            )
            analysis_summary = (
                "낙상과 유사한 신호가 일부 있었지만 최종 판단은 정상입니다."
            )

        risk_reason = [
            f"낙상 위험도 {fall_percent}%로 Fall Alert 기준 70% 미만입니다.",
            f"최대 속도 {round(speed_max, 4)} m/s로 낙상 충격 패턴으로 보기 어렵습니다.",
            f"높이 변화 {round(height_drop, 4)} m로 강한 낙상 기준에 도달하지 않았습니다.",
            f"이후 이동거리 {round(movement_after * 100, 2)} cm가 확인되었습니다.",
            "최종 상태가 Normal이므로 보호자 알림 대상이 아닙니다.",
        ]

        recommendations = [
            "보호자 즉시 알림은 필요하지 않습니다.",
            "정상 데이터로 처리하고 DB 저장 대상에서 제외합니다.",
            "동일 사용자의 비슷한 움직임이 반복되면 이상행동 페이지에서 별도 추세로 확인할 수 있습니다.",
        ]

        return {
            "action_case": action_case,
            "action_label": action_label,
            "behavior_pattern": behavior_pattern,
            "likely_cause": likely_cause,
            "direction": "해당 없음",
            "risk_reason": risk_reason,
            "recommendations": recommendations,
            "analysis_summary": analysis_summary,
        }

    # =====================================================
    # 2. Fall Alert일 때만 낙상 케이스 분석
    # =====================================================

    direction = "방향 정보 부족"

    if "forward" in name or "front" in name:
        direction = "앞쪽 방향으로 넘어짐"
    elif "back" in name or "backward" in name:
        direction = "뒤쪽 방향으로 넘어짐"
    elif "left" in name:
        direction = "왼쪽 방향으로 넘어짐"
    elif "right" in name:
        direction = "오른쪽 방향으로 넘어짐"

    action_case = "general_fall"
    action_label = "일반 낙상"
    behavior_pattern = (
        "짧은 시간 안에 높이 변화와 속도 변화가 함께 발생한 낙상 패턴입니다."
    )
    likely_cause = "균형 상실, 미끄러짐, 갑작스러운 자세 변화 가능성이 있습니다."

    if "standing" in name or "stand" in name:
        action_case = "fall_after_standing"
        action_label = "서 있다가 낙상"
        behavior_pattern = (
            "서 있는 상태에서 중심 높이가 급격히 낮아지고, "
            "이후 움직임이 줄어드는 패턴입니다."
        )
        likely_cause = "기립 상태에서 균형 상실, 어지럼증, 발 헛디딤 가능성이 있습니다."

    elif "sitting" in name or "sit" in name:
        action_case = "fall_from_sitting"
        action_label = "앉아있다가 낙상"
        behavior_pattern = (
            "앉은 자세 또는 낮은 자세에서 몸의 중심이 바닥 방향으로 더 낮아지는 패턴입니다."
        )
        likely_cause = "의자나 바닥에서 자세를 바꾸는 과정에서 균형을 잃었을 가능성이 있습니다."

    elif "chair" in name:
        action_case = "fall_from_chair"
        action_label = "의자에서 낙상"
        behavior_pattern = (
            "의자 높이 부근에서 시작해 짧은 시간 안에 높이가 크게 낮아지는 패턴입니다."
        )
        likely_cause = "의자에서 일어나거나 앉는 과정에서 미끄러짐 또는 중심 상실 가능성이 있습니다."

    elif "bed" in name:
        action_case = "fall_from_bed"
        action_label = "침대에서 낙상"
        behavior_pattern = "침대 높이에서 바닥 방향으로 높이가 낮아지는 패턴입니다."
        likely_cause = "침대에서 내려오거나 몸을 돌리는 과정에서 떨어졌을 가능성이 있습니다."

    elif "walk" in name or "walking" in name:
        action_case = "fall_after_walking"
        action_label = "걷다가 낙상"
        behavior_pattern = (
            "이동 중 속도 변화가 나타난 뒤 높이가 급격히 낮아지는 패턴입니다."
        )
        likely_cause = "보행 중 장애물, 미끄러짐, 다리 힘 풀림 가능성이 있습니다."

    elif "run" in name or "running" in name:
        action_case = "fall_after_running"
        action_label = "빠르게 이동하다가 낙상"
        behavior_pattern = (
            "속도 변화가 크게 나타난 뒤 중심 높이가 급격히 떨어지는 패턴입니다."
        )
        likely_cause = "빠른 이동 중 발 헛디딤 또는 급정지로 인한 균형 상실 가능성이 있습니다."

    elif "lie" in name or "lying" in name:
        action_case = "lying_or_fall"
        action_label = "눕기 동작 또는 낙상 의심"
        behavior_pattern = (
            "높이가 낮아지는 패턴이 있으나 눕기 동작과 낙상이 유사할 수 있습니다."
        )
        likely_cause = "침대나 바닥에 눕는 동작일 수 있어 추가 확인이 필요합니다."

    if speed_max >= 1.2:
        impact_text = "순간 속도가 커서 충격성 움직임이 강합니다."
    elif speed_max >= 0.6:
        impact_text = "중간 수준 이상의 속도 변화가 있습니다."
    else:
        impact_text = "속도 변화는 크지 않지만 다른 낙상 조건과 함께 판단되었습니다."

    if height_drop >= 0.8:
        height_text = "높이 변화가 매우 커서 서 있던 자세에서 바닥 방향으로 내려간 패턴에 가깝습니다."
    elif height_drop >= 0.4:
        height_text = "높이 변화가 있어 낙상 가능성을 판단할 수 있습니다."
    else:
        height_text = "높이 변화는 크지 않지만 모델 위험도와 후속 움직임을 함께 고려했습니다."

    if movement_after <= 0.2:
        after_text = "동작 이후 이동이 적어 낙상 후 움직임이 줄어든 상태로 볼 수 있습니다."
    else:
        after_text = "동작 이후 움직임이 남아 있어 추가 확인이 필요합니다."

    risk_reason = [
        f"낙상 위험도 {fall_percent}%로 Fall Alert 기준 70% 이상입니다.",
        impact_text,
        height_text,
        after_text,
        f"방향 추정: {direction}",
    ]

    recommendations = [
        "보호자 또는 관리자 확인이 필요합니다.",
        "낙상 구간 전후의 센서 로그를 확인하세요.",
        "후속 움직임이 거의 없다면 즉시 연락 또는 방문 확인이 필요합니다.",
    ]

    return {
        "action_case": action_case,
        "action_label": action_label,
        "behavior_pattern": behavior_pattern,
        "likely_cause": likely_cause,
        "direction": direction,
        "risk_reason": risk_reason,
        "recommendations": recommendations,
        "analysis_summary": f"{action_label} 가능성이 가장 높습니다. {behavior_pattern}",
    }


def build_normalized_frontend_result(
    result: dict,
    feat: dict,
    fall_prob: float,
    pred_label,
    model_threshold: float,
    threshold_pred_label: str,
    file_name: str,
    meta: dict,
    frame_meta: dict,
) -> dict:
    result = result if isinstance(result, dict) else {}
    feat = feat if isinstance(feat, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    frame_meta = frame_meta if isinstance(frame_meta, dict) else {}

    fall_prob = to_float(fall_prob, 0.0)
    fall_percent = round(fall_prob * 100, 2)

    speed_max = pick_number(
        result,
        feat,
        keys=[
            "speed_max",
            "max_speed",
            "v_abs_max",
            "abs_v_max",
            "v_max_abs",
            "v_max",
            "velocity_max",
        ],
    )

    height_drop = pick_number(
        result,
        feat,
        keys=[
            "height_drop",
            "z_drop",
            "z_range",
            "z_center_drop",
            "z_center_first_to_min_drop",
            "z_center_peak_to_last_drop",
            "max_height_drop",
        ],
    )

    movement_after = pick_number(
        result,
        feat,
        keys=[
            "movement_after",
            "movement_after_fall",
            "after_movement",
            "tail_movement",
            "movement_after_mean",
            "center_move_tail_mean",
            "center_move_after",
        ],
    )

    status = normalize_status(result, fall_prob)
    alert = status == "Fall Alert"

    if status == "Fall Alert":
        message = result.get(
            "message",
            "낙상으로 판단되었습니다. Fall Alert 이벤트를 MongoDB에 저장합니다.",
        )
        alert_message = "알림 발생"
    else:
        message = "정상 행동으로 판단되었습니다. 낙상 알림 기준에 도달하지 않아 DB에는 저장하지 않습니다."
        alert_message = "알림 없음"

    normalized = dict(result)

    normalized.update(
        {
            "status": status,
            "alert": alert,
            "alert_message": alert_message,
            "message": message,

            "file_name": file_name,
            "filename": file_name,

            "fall_prob": round(fall_prob, 4),
            "fall_probability": round(fall_prob, 4),
            "raw_model_fall_prob": round(fall_prob, 4),
            "fall_risk": round(fall_prob, 4),
            "fall_risk_percent": fall_percent,
            "risk_score": fall_percent,

            "speed_max": round(speed_max, 4),
            "max_speed": round(speed_max, 4),
            "height_drop": round(height_drop, 4),
            "movement_after": round(movement_after, 4),
            "movement_after_cm": round(movement_after * 100, 2),

            "model_pred_label": str(pred_label),
            "model_threshold": round(float(model_threshold), 4),
            "display_alert_threshold": ALERT_THRESHOLD,
            "threshold_pred_label": threshold_pred_label,

            "loaded_model_path": meta.get("_loaded_model_path"),
            "loaded_meta_path": meta.get("_loaded_meta_path"),

            "frame_start": frame_meta.get("frame_start", 0),
            "frame_end": frame_meta.get("frame_end", 0),
            "frame_count": frame_meta.get("frame_count", 0),
            "point_count": frame_meta.get("point_count", 0),

            "assumed_fps": frame_meta.get("assumed_fps", ASSUMED_FPS),
            "realtime_frame_window": frame_meta.get(
                "realtime_frame_window",
                REALTIME_FRAME_WINDOW,
            ),
            "realtime_window_seconds": frame_meta.get(
                "realtime_window_seconds",
                REALTIME_WINDOW_SECONDS,
            ),

            "features": {
                **feat,
                "fall_prob": round(fall_prob, 4),
                "fall_probability": round(fall_prob, 4),
                "fall_risk_percent": fall_percent,
                "speed_max": round(speed_max, 4),
                "max_speed": round(speed_max, 4),
                "height_drop": round(height_drop, 4),
                "movement_after": round(movement_after, 4),
                "movement_after_cm": round(movement_after * 100, 2),
            },

            "sensor_features": normalized.get(
                "sensor_features",
                build_sensor_feature_table(feat),
            ),

            "created_at": now_text(),
        }
    )

    behavior = infer_behavior_case(file_name, normalized)
    normalized.update(behavior)

    return normalized


# =========================================================
# 모델 로딩
# =========================================================

def find_model_and_meta_paths(raise_if_missing: bool = True):
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            meta_path = model_path.with_name(f"{model_path.stem}_meta.json")
            if meta_path.exists():
                return model_path, meta_path
            return model_path, None

    if raise_if_missing:
        checked_paths = "\n".join(str(p) for p in MODEL_CANDIDATES)
        raise FileNotFoundError(
            "낙상 모델 파일을 찾을 수 없습니다.\n"
            f"아래 경로 중 하나에 모델 파일이 있어야 합니다.\n\n{checked_paths}"
        )

    return None, None


def load_model_and_meta():
    global _cached_model
    global _cached_meta
    global _cached_model_path

    model_path, meta_path = find_model_and_meta_paths(raise_if_missing=True)

    if _cached_model is not None and _cached_model_path == str(model_path):
        return _cached_model, _cached_meta

    model = joblib.load(model_path)

    meta = {}
    if meta_path is not None and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta["_loaded_model_path"] = str(model_path)
    meta["_loaded_meta_path"] = str(meta_path) if meta_path is not None else None

    _cached_model = model
    _cached_meta = meta
    _cached_model_path = str(model_path)

    print(f"[FALL MODEL] 모델 캐시 로드 완료: {model_path}")

    return model, meta


def get_feature_names(model, meta: dict, feat: dict) -> list:
    if meta and "feature_columns" in meta:
        return list(meta["feature_columns"])

    if meta and "feature_names" in meta:
        return list(meta["feature_names"])

    if hasattr(model, "feature_columns_"):
        return list(model.feature_columns_)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return sorted(feat.keys())


def get_model_threshold(model, meta: dict) -> float:
    if meta and "best_threshold" in meta:
        return float(meta["best_threshold"])

    if meta and "threshold" in meta:
        return float(meta["threshold"])

    if hasattr(model, "threshold_"):
        return float(model.threshold_)

    return ALERT_THRESHOLD


def predict_single_csv(
    csv_path: Path,
    file_name: str,
    model,
    meta: dict,
    frame_meta_override: dict | None = None,
) -> dict:
    frame_meta = frame_meta_override or get_csv_frame_meta(csv_path)

    feat = extract_features_from_csv(csv_path)

    if not isinstance(feat, dict):
        raise ValueError("feature_extractor 결과가 dict 형태가 아닙니다.")

    feature_names = get_feature_names(model, meta, feat)

    X = pd.DataFrame([feat])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_names]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    fall_prob, pred_label = get_fall_probability(model, X)

    model_threshold = get_model_threshold(model, meta)
    threshold_pred_label = "Fall" if float(fall_prob) >= ALERT_THRESHOLD else "Normal"

    post_result = postprocess_fall_result(
        fall_prob=fall_prob,
        features=feat,
        file_name=file_name,
    )

    result = build_normalized_frontend_result(
        result=post_result,
        feat=feat,
        fall_prob=fall_prob,
        pred_label=pred_label,
        model_threshold=model_threshold,
        threshold_pred_label=threshold_pred_label,
        file_name=file_name,
        meta=meta,
        frame_meta=frame_meta,
    )

    return result


def build_aggregate_result(file_name: str, window_results: list[dict]) -> dict:
    if not window_results:
        raise ValueError("집계할 window 결과가 없습니다.")

    best = max(
        window_results,
        key=lambda item: (
            to_float(item.get("fall_prob")),
            1 if item.get("status") == "Fall Alert" else 0,
        ),
    )

    final_status = "Fall Alert" if to_float(best.get("fall_prob")) >= ALERT_THRESHOLD else "Normal"
    final_alert = final_status == "Fall Alert"

    final_result = dict(best)

    final_result.update(
        {
            "status": final_status,
            "alert": final_alert,
            "file_name": file_name,
            "filename": file_name,
            "source_mode": "10frame_aggregate",
            "window_count": len(window_results),
            "detected_window": {
                "chunk_index": best.get("chunk_index"),
                "frame_start": best.get("frame_start"),
                "frame_end": best.get("frame_end"),
                "fall_risk_percent": best.get("fall_risk_percent"),
                "status": final_status,
            },
            "window_results": [
                {
                    "chunk_index": item.get("chunk_index"),
                    "frame_start": item.get("frame_start"),
                    "frame_end": item.get("frame_end"),
                    "frame_count": item.get("frame_count"),
                    "point_count": item.get("point_count"),
                    "status": item.get("status"),
                    "alert": item.get("alert"),
                    "fall_prob": item.get("fall_prob"),
                    "fall_probability": item.get("fall_probability"),
                    "raw_model_fall_prob": item.get("raw_model_fall_prob"),
                    "fall_risk_percent": item.get("fall_risk_percent"),
                    "risk_score": item.get("risk_score"),
                    "speed_max": item.get("speed_max"),
                    "height_drop": item.get("height_drop"),
                    "movement_after": item.get("movement_after"),
                }
                for item in window_results
            ],
        }
    )

    if final_status == "Fall Alert":
        final_result["message"] = (
            "전체 CSV를 10프레임 단위로 분석한 결과, "
            f"가장 위험한 구간은 frame {best.get('frame_start')} ~ {best.get('frame_end')}이며 "
            f"위험도는 {round(to_float(best.get('fall_prob')) * 100)}%입니다."
        )
    else:
        final_result["message"] = (
            "전체 CSV를 10프레임 단위로 분석했지만 Fall Alert 기준에 도달한 구간이 없습니다."
        )

    behavior = infer_behavior_case(file_name, final_result, window_results)
    final_result.update(behavior)

    return final_result


# =========================================================
# MongoDB
# =========================================================

def save_event_to_mongodb(event: dict) -> dict:
    if event.get("status") != "Fall Alert":
        return {
            "db_saved": False,
            "db_message": "Fall Alert가 아니므로 MongoDB에 저장하지 않습니다.",
        }

    if events_collection is None:
        return {
            "db_saved": False,
            "db_message": "MongoDB가 연결되지 않아 저장하지 못했습니다.",
        }

    try:
        insert_data = dict(event)
        insert_data["created_at_dt"] = datetime.now()

        result = events_collection.insert_one(insert_data)

        return {
            "db_saved": True,
            "event_id": str(result.inserted_id),
            "db_message": "낙상 알림 이벤트가 MongoDB에 저장되었습니다.",
        }

    except Exception as e:
        return {
            "db_saved": False,
            "db_message": f"MongoDB 저장 실패: {e}",
        }


def serialize_mongo_doc(doc: dict) -> dict:
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])

    if "created_at_dt" in doc and hasattr(doc["created_at_dt"], "strftime"):
        doc["created_at_dt"] = doc["created_at_dt"].strftime("%Y-%m-%d %H:%M:%S")

    return doc


# =========================================================
# Startup / Shutdown
# =========================================================

def startup_fall_dashboard():
    global mongo_client
    global events_collection

    if MongoClient is None:
        mongo_client = None
        events_collection = None
        print("[MongoDB] pymongo가 설치되지 않았습니다. 저장 없이 예측만 동작합니다.")
    else:
        try:
            mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
            mongo_client.admin.command("ping")

            db = mongo_client[MONGO_DB_NAME]
            events_collection = db[MONGO_COLLECTION_NAME]

            events_collection.create_index([("created_at_dt", DESCENDING)])
            events_collection.create_index([("created_at", DESCENDING)])
            events_collection.create_index([("status", 1)])
            events_collection.create_index([("alert", 1)])
            events_collection.create_index([("file_name", 1)])

            print(f"[MongoDB] 연결 성공: {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")

        except Exception as e:
            mongo_client = None
            events_collection = None
            print(f"[MongoDB] 연결 실패: {e}")
            print("[MongoDB] 저장 기능 없이 낙상 예측 API만 동작합니다.")

    model_path, meta_path = find_model_and_meta_paths(raise_if_missing=False)

    if model_path is not None:
        print(f"[FALL MODEL] 사용 모델: {model_path}")
    else:
        print("[FALL MODEL] 사용 가능한 낙상 모델 파일을 찾지 못했습니다.")

    if meta_path is not None:
        print(f"[FALL MODEL] 메타 파일: {meta_path}")
    else:
        print("[FALL MODEL] 메타 파일 없음")


def shutdown_fall_dashboard():
    global mongo_client

    if mongo_client is not None:
        mongo_client.close()
        print("[MongoDB] 연결 종료")


# =========================================================
# API
# =========================================================

@router.get("/fall/health")
def health():
    model_path, meta_path = find_model_and_meta_paths(raise_if_missing=False)

    return {
        "status": "ok",
        "api": "fall-running",
        "message": "FallDashboard API is running",

        "fps": ASSUMED_FPS,
        "realtime_frame_window": REALTIME_FRAME_WINDOW,
        "realtime_window_seconds": REALTIME_WINDOW_SECONDS,
        "display_alert_threshold": ALERT_THRESHOLD,

        "rf_dir_exists": RF_DIR.exists(),
        "model_exists": model_path is not None,
        "meta_exists": meta_path is not None,
        "mongo_connected": events_collection is not None,

        "model_path": str(model_path) if model_path is not None else None,
        "meta_path": str(meta_path) if meta_path is not None else None,

        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "rf_dir": str(RF_DIR),
        "model_dir": str(MODEL_DIR),
    }


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    save_event: bool = Query(True),
):
    file_name = file.filename or "uploaded.csv"

    if not file_name.lower().endswith(".csv"):
        return {
            "status": "Error",
            "alert": False,
            "message": "CSV 파일만 업로드할 수 있습니다.",
            "db_saved": False,
            "file_name": file_name,
            "filename": file_name,
            "fall_prob": 0.0,
            "fall_probability": 0.0,
            "fall_risk_percent": 0.0,
            "risk_score": 0.0,
            "speed_max": 0.0,
            "height_drop": 0.0,
            "movement_after": 0.0,
            "created_at": now_text(),
        }

    tmp_path = None
    chunk_paths = []

    try:
        model, meta = load_model_and_meta()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        chunks = split_csv_by_frame_window(tmp_path, REALTIME_FRAME_WINDOW)
        chunk_paths = [item["tmp_path"] for item in chunks]

        if not chunks:
            raise ValueError("CSV를 10프레임 단위로 나눌 수 없습니다.")

        window_results = []

        for chunk in chunks:
            frame_meta = {
                "frame_start": chunk["frame_start"],
                "frame_end": chunk["frame_end"],
                "frame_count": chunk["frame_count"],
                "point_count": chunk["point_count"],
                "assumed_fps": ASSUMED_FPS,
                "realtime_frame_window": REALTIME_FRAME_WINDOW,
                "realtime_window_seconds": REALTIME_WINDOW_SECONDS,
            }

            chunk_result = predict_single_csv(
                csv_path=chunk["tmp_path"],
                file_name=file_name,
                model=model,
                meta=meta,
                frame_meta_override=frame_meta,
            )

            chunk_result["chunk_index"] = chunk["chunk_index"]
            window_results.append(chunk_result)

        result = build_aggregate_result(file_name, window_results)

        if save_event:
            db_result = save_event_to_mongodb(result)
        else:
            db_result = {
                "db_saved": False,
                "db_message": "실시간 구간 분석 중에는 DB에 저장하지 않습니다. 최종 결과만 저장합니다.",
            }

        result.update(db_result)

        return result

    except Exception as e:
        print("[FALL PREDICT ERROR]", e)

        return {
            "status": "Error",
            "alert": False,
            "message": f"예측 처리 중 오류가 발생했습니다: {e}",
            "db_saved": False,
            "file_name": file_name,
            "filename": file_name,

            "fall_prob": 0.0,
            "fall_probability": 0.0,
            "raw_model_fall_prob": 0.0,
            "fall_risk": 0.0,
            "fall_risk_percent": 0.0,
            "risk_score": 0.0,

            "speed_max": 0.0,
            "max_speed": 0.0,
            "height_drop": 0.0,
            "movement_after": 0.0,
            "movement_after_cm": 0.0,

            "frame_start": 0,
            "frame_end": 0,
            "frame_count": 0,
            "point_count": 0,

            "action_case": "unknown",
            "action_label": "분석 실패",
            "behavior_pattern": "예측 처리 중 오류가 발생하여 행동 패턴을 분석하지 못했습니다.",
            "likely_cause": "CSV 형식, 모델 파일, feature_extractor를 확인해야 합니다.",
            "risk_reason": [str(e)],
            "recommendations": ["백엔드 로그와 CSV 컬럼명을 확인하세요."],

            "created_at": now_text(),
        }

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        for path in chunk_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


@router.get("/events")
def get_events(limit: int = 30):
    if events_collection is None:
        return {
            "mongo_connected": False,
            "events": [],
            "count": 0,
            "message": "MongoDB가 연결되지 않았습니다.",
        }

    limit = max(1, min(limit, 100))

    docs = (
        events_collection.find({"status": "Fall Alert"})
        .sort("created_at_dt", DESCENDING)
        .limit(limit)
    )

    events = [serialize_mongo_doc(doc) for doc in docs]

    return {
        "mongo_connected": True,
        "events": events,
        "count": len(events),
    }


@router.get("/stats")
def get_stats():
    if events_collection is None:
        return {
            "mongo_connected": False,
            "total_events": 0,
            "fall_alert_count": 0,
            "exception_count": 0,
            "message": "MongoDB가 연결되지 않았습니다.",
        }

    fall_alert_count = events_collection.count_documents({"status": "Fall Alert"})

    return {
        "mongo_connected": True,
        "total_events": fall_alert_count,
        "fall_alert_count": fall_alert_count,
        "exception_count": 0,
    }


@router.delete("/events")
def delete_all_events():
    if events_collection is None:
        return {
            "deleted_count": 0,
            "message": "MongoDB가 연결되지 않았습니다.",
        }

    result = events_collection.delete_many({})

    return {
        "deleted_count": result.deleted_count,
        "message": "이벤트 로그를 삭제했습니다.",
    }