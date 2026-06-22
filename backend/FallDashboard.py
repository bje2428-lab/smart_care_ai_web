from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File

try:
    from pymongo import MongoClient, DESCENDING
except Exception:
    MongoClient = None
    DESCENDING = -1


# =========================================================
# 프로젝트 경로 설정
# backend/FallDashboard.py 기준
# ROOT_DIR    = C:\smart_care_ai_web
# BACKEND_DIR = C:\smart_care_ai_web\backend
# RF_DIR      = C:\smart_care_ai_web\rf
# MODEL_DIR   = C:\smart_care_ai_web\models
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
RF_DIR = ROOT_DIR / "rf"
MODEL_DIR = ROOT_DIR / "models"


# rf 패키지 import를 위해 루트와 rf 둘 다 등록
for path in [ROOT_DIR, RF_DIR, BACKEND_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:
    from rf.feature_extractor import extract_features_from_csv
    from rf.predict_csv import postprocess_fall_result, get_fall_probability
except ModuleNotFoundError:
    from feature_extractor import extract_features_from_csv
    from predict_csv import postprocess_fall_result, get_fall_probability


# =========================================================
# Router
# main.py에서 include_router로 연결됨
# 최종 주소:
# GET    /health
# POST   /predict
# GET    /events
# GET    /stats
# DELETE /events
# =========================================================

router = APIRouter(tags=["Fall Detection"])


# =========================================================
# 낙상 모델 경로 설정
# =========================================================

MODEL_CANDIDATES = [
    MODEL_DIR / "mmwave_rf_smote_model.pkl",
    MODEL_DIR / "mmwave_rf_model.pkl",
    MODEL_DIR / "mmwave_rf_aug_model.pkl",
]


# =========================================================
# MongoDB 설정
# =========================================================

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "smart_care_ai")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "detection_events")


mongo_client = None
events_collection = None


# =========================================================
# 공통 유틸 함수
# =========================================================

def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def pick_number(*sources, keys=None, default: float = 0.0) -> float:
    """
    여러 dict에서 같은 의미의 값을 찾아 float로 반환.
    프론트에서 0만 보이는 문제를 줄이기 위해 다양한 key 이름을 같이 탐색.
    """
    keys = keys or []

    for source in sources:
        if not isinstance(source, dict):
            continue

        for key in keys:
            if key in source and source.get(key) is not None:
                return to_float(source.get(key), default)

    return default


def normalize_status(result: dict, fall_prob: float, model_threshold: float) -> str:
    """
    postprocess_fall_result 결과의 status가 Error로 들어와도
    실제 메시지/alert/확률 기준으로 프론트 표시용 상태를 보정.
    """
    raw_status = str(result.get("status", "")).strip()
    message = str(result.get("message", "")).strip()

    valid_statuses = ["Fall Alert", "Exception", "Normal"]

    if raw_status in valid_statuses:
        return raw_status

    if result.get("alert") is True:
        return "Fall Alert"

    if "Fall Alert" in message or "낙상 알림" in message or "알림 발생" in message:
        return "Fall Alert"

    if "Exception" in message or "예외" in message:
        return "Exception"

    if "정상" in message or "정상 행동" in message:
        return "Normal"

    if fall_prob >= model_threshold:
        return "Fall Alert"

    return "Normal"


def build_normalized_frontend_result(
    result: dict,
    feat: dict,
    fall_prob: float,
    pred_label,
    model_threshold: float,
    threshold_pred_label: str,
    file_name: str,
    meta: dict,
) -> dict:
    """
    프론트가 읽을 수 있도록 key 이름을 여러 형태로 같이 넣어줌.
    - status가 Error로 잘못 보이는 문제 보정
    - fall_prob / fall_probability / fall_risk_percent 같이 제공
    - speed_max / height_drop / movement_after 같이 제공
    - file_name / filename 같이 제공
    """
    if not isinstance(result, dict):
        result = {}

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

    status = normalize_status(result, fall_prob, model_threshold)
    alert = status == "Fall Alert"

    if status == "Fall Alert":
        message = result.get(
            "message",
            "낙상으로 판단되었습니다. Fall Alert 이벤트를 MongoDB에 저장합니다.",
        )
        alert_message = "알림 발생"
    elif status == "Exception":
        message = result.get(
            "message",
            "예외 행동으로 판단되었습니다. 낙상 알림 기준에는 도달하지 않아 DB에는 저장하지 않습니다.",
        )
        alert_message = "알림 없음"
    else:
        message = result.get(
            "message",
            "정상 행동으로 판단되었습니다. DB에는 저장하지 않습니다.",
        )
        alert_message = "알림 없음"

    normalized = dict(result)

    normalized.update(
        {
            # 상태
            "status": status,
            "alert": alert,
            "alert_message": alert_message,
            "message": message,

            # 파일명: 프론트 호환용으로 둘 다 제공
            "file_name": file_name,
            "filename": file_name,

            # 낙상 확률: 프론트 호환용으로 여러 이름 제공
            "fall_prob": round(fall_prob, 4),
            "fall_probability": round(fall_prob, 4),
            "raw_model_fall_prob": round(fall_prob, 4),
            "fall_risk": round(fall_prob, 4),
            "fall_risk_percent": fall_percent,
            "risk_score": fall_percent,

            # 핵심 지표
            "speed_max": round(speed_max, 4),
            "max_speed": round(speed_max, 4),
            "height_drop": round(height_drop, 4),
            "movement_after": round(movement_after, 4),
            "movement_after_cm": round(movement_after * 100, 2),

            # 모델 정보
            "model_pred_label": str(pred_label),
            "model_threshold": round(float(model_threshold), 4),
            "threshold_pred_label": threshold_pred_label,
            "loaded_model_path": meta.get("_loaded_model_path"),
            "loaded_meta_path": meta.get("_loaded_meta_path"),

            # 프론트 그래프/표시용
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

            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    return normalized


# =========================================================
# 서버 시작 / 종료 함수
# main.py에서 호출함
# =========================================================

def startup_fall_dashboard():
    global mongo_client
    global events_collection

    if MongoClient is None:
        mongo_client = None
        events_collection = None
        print("[MongoDB] pymongo가 설치되지 않았습니다. 저장 기능 없이 예측 API만 동작합니다.")
    else:
        try:
            mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
            mongo_client.admin.command("ping")

            db = mongo_client[MONGO_DB_NAME]
            events_collection = db[MONGO_COLLECTION_NAME]

            events_collection.create_index([("created_at", DESCENDING)])
            events_collection.create_index([("status", 1)])
            events_collection.create_index([("alert", 1)])

            print(f"[MongoDB] 연결 성공: {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")

        except Exception as e:
            mongo_client = None
            events_collection = None
            print(f"[MongoDB] 연결 실패: {e}")
            print("[MongoDB] 저장 기능 없이 낙상 예측 API만 동작합니다.")

    try:
        model_path, meta_path = find_model_and_meta_paths(raise_if_missing=False)

        if model_path is not None:
            print(f"[FALL MODEL] 사용 모델: {model_path}")
        else:
            print("[FALL MODEL] 사용 가능한 낙상 모델 파일을 찾지 못했습니다.")

        if meta_path is not None:
            print(f"[FALL MODEL] 메타 파일: {meta_path}")
        else:
            print("[FALL MODEL] 메타 파일 없음. 모델 속성 또는 feature_extractor 기준으로 동작합니다.")

    except Exception as e:
        print(f"[FALL MODEL] 모델 확인 중 오류: {e}")


def shutdown_fall_dashboard():
    global mongo_client

    if mongo_client is not None:
        mongo_client.close()
        print("[MongoDB] 연결 종료")


# =========================================================
# 모델 로딩 함수
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
            "아래 경로 중 하나에 모델 파일이 있어야 합니다.\n\n"
            f"{checked_paths}"
        )

    return None, None


def load_model_and_meta():
    model_path, meta_path = find_model_and_meta_paths(raise_if_missing=True)

    model = joblib.load(model_path)

    meta = {}

    if meta_path is not None and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta["_loaded_model_path"] = str(model_path)
    meta["_loaded_meta_path"] = str(meta_path) if meta_path is not None else None

    return model, meta


def get_feature_names(model, meta: dict, feat: dict) -> list[str]:
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

    return 0.5


# =========================================================
# 프론트 표시용 센서 설명
# =========================================================

def build_sensor_feature_table(feat: dict) -> list[dict]:
    descriptions = {
        "x": {
            "name": "x",
            "label": "좌우 위치",
            "meaning": "레이더 기준 사람/물체가 좌우로 얼마나 이동했는지 나타내는 값",
        },
        "y": {
            "name": "y",
            "label": "전후 거리",
            "meaning": "레이더와 대상 사이의 앞뒤 거리 변화를 나타내는 값",
        },
        "z": {
            "name": "z",
            "label": "높이",
            "meaning": "대상의 높이 변화. 낙상이나 앉기처럼 자세가 낮아질 때 크게 변할 수 있음",
        },
        "v": {
            "name": "v",
            "label": "속도",
            "meaning": "대상의 움직임 속도. 순간적으로 빠른 움직임이 있으면 값이 커질 수 있음",
        },
    }

    rows = []

    for key, info in descriptions.items():
        rows.append(
            {
                "name": info["name"],
                "label": info["label"],
                "meaning": info["meaning"],
                "mean": round(float(feat.get(f"{key}_mean", 0.0)), 4),
                "min": round(float(feat.get(f"{key}_min", 0.0)), 4),
                "max": round(float(feat.get(f"{key}_max", 0.0)), 4),
                "range": round(float(feat.get(f"{key}_range", 0.0)), 4),
            }
        )

    return rows


# =========================================================
# MongoDB 저장 함수
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
# 낙상 상태 확인 API
# =========================================================

@router.get("/health")
def health():
    model_path, meta_path = find_model_and_meta_paths(raise_if_missing=False)

    return {
        "status": "ok",
        "api": "fall-running",
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


# =========================================================
# 낙상 예측 API
# =========================================================

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        return {
            "status": "Error",
            "alert": False,
            "message": "CSV 파일만 업로드할 수 있습니다.",
            "db_saved": False,
        }

    tmp_path = None

    try:
        model, meta = load_model_and_meta()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        # 1. CSV에서 feature 추출
        feat = extract_features_from_csv(tmp_path)

        # 2. 학습 때 사용한 feature 순서 맞추기
        feature_names = get_feature_names(model, meta, feat)

        X = pd.DataFrame([feat])

        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0

        X = X[feature_names].fillna(0.0)

        # 3. Fall 확률 계산
        fall_prob, pred_label = get_fall_probability(model, X)

        # 4. threshold 정보
        model_threshold = get_model_threshold(model, meta)
        threshold_pred_label = "Fall" if float(fall_prob) >= model_threshold else "Normal"

        # 5. 후처리
        post_result = postprocess_fall_result(
            fall_prob=fall_prob,
            features=feat,
            file_name=file.filename,
        )

        # 6. 프론트 표시용 결과 보정
        result = build_normalized_frontend_result(
            result=post_result,
            feat=feat,
            fall_prob=fall_prob,
            pred_label=pred_label,
            model_threshold=model_threshold,
            threshold_pred_label=threshold_pred_label,
            file_name=file.filename,
            meta=meta,
        )

        # 7. Fall Alert일 때만 MongoDB 저장
        db_result = save_event_to_mongodb(result)
        result.update(db_result)

        return result

    except Exception as e:
        return {
            "status": "Error",
            "alert": False,
            "message": str(e),
            "db_saved": False,
            "file_name": file.filename,
            "filename": file.filename,
            "fall_prob": 0.0,
            "fall_probability": 0.0,
            "fall_risk": 0.0,
            "fall_risk_percent": 0.0,
            "risk_score": 0.0,
            "speed_max": 0.0,
            "max_speed": 0.0,
            "height_drop": 0.0,
            "movement_after": 0.0,
            "movement_after_cm": 0.0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


# =========================================================
# 낙상 이벤트 조회 API
# =========================================================

@router.get("/events")
def get_events(limit: int = 30):
    if events_collection is None:
        return {
            "mongo_connected": False,
            "events": [],
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