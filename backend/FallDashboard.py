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
from pymongo import MongoClient, DESCENDING


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

if str(RF_DIR) not in sys.path:
    sys.path.insert(0, str(RF_DIR))

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
# pkl 파일은 전부 C:\smart_care_ai_web\models 안에 둠
# 우선순위:
# 1. mmwave_rf_smote_model.pkl
# 2. mmwave_rf_model.pkl
# 3. mmwave_rf_aug_model.pkl
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
# 서버 시작 / 종료 함수
# main.py에서 호출함
# =========================================================

def startup_fall_dashboard():
    """
    main.py의 startup_event에서 호출됩니다.
    MongoDB 연결과 낙상 모델 경로 확인을 수행합니다.
    """
    global mongo_client
    global events_collection

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
    """
    main.py의 shutdown_event에서 호출됩니다.
    """
    global mongo_client

    if mongo_client is not None:
        mongo_client.close()
        print("[MongoDB] 연결 종료")


# =========================================================
# 모델 로딩 함수
# =========================================================

def find_model_and_meta_paths(raise_if_missing: bool = True):
    """
    사용할 낙상 모델 파일과 메타 파일을 찾습니다.
    우선 루트 models 폴더에서 SMOTE 모델을 찾고,
    없으면 기존 모델을 찾습니다.
    """
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
    """
    학습된 RandomForest 또는 SMOTE Pipeline 모델과 메타 정보를 불러옵니다.
    """
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
    """
    학습 때 사용한 feature 순서를 가져옵니다.

    우선순위:
    1. meta["feature_columns"]
    2. meta["feature_names"]
    3. model.feature_columns_
    4. model.feature_names_in_
    5. 현재 추출된 feature 이름
    """
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
    """
    학습 때 저장한 best threshold를 가져옵니다.
    없으면 기본값 0.5를 사용합니다.
    """
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
    """
    프론트엔드에서 x, y, z, v를 표로 보여주기 위한 데이터입니다.
    feature_extractor.py에서 만든 통계값을 사용합니다.
    """
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
    """
    MongoDB 저장 정책:

    Normal      -> 저장 안 함
    Exception   -> 저장 안 함
    Fall Alert  -> 저장함
    """
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
    """
    MongoDB ObjectId와 datetime을 프론트에서 보기 좋은 문자열로 변환합니다.
    """
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
        "model_exists": model_path is not None,
        "meta_exists": meta_path is not None,
        "mongo_connected": events_collection is not None,
        "model_path": str(model_path) if model_path is not None else None,
        "meta_path": str(meta_path) if meta_path is not None else None,
        "model_dir": str(MODEL_DIR),
    }


# =========================================================
# 낙상 예측 API
# =========================================================

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    CSV 파일 업로드
    → 낙상 예측
    → 예외처리
    → Fall Alert일 때만 MongoDB 저장
    → x, y, z, v 설명/통계값 포함 반환
    """
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
        result = postprocess_fall_result(
            fall_prob=fall_prob,
            features=feat,
            file_name=file.filename,
        )

        # 6. 프론트 표시용 값 추가
        result.update(
            {
                "file_name": file.filename,
                "raw_model_fall_prob": round(float(fall_prob), 4),
                "model_pred_label": str(pred_label),
                "model_threshold": round(float(model_threshold), 4),
                "threshold_pred_label": threshold_pred_label,
                "loaded_model_path": meta.get("_loaded_model_path"),
                "loaded_meta_path": meta.get("_loaded_meta_path"),
                "sensor_features": result.get(
                    "sensor_features",
                    build_sensor_feature_table(feat),
                ),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
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
    """
    MongoDB에 저장된 Fall Alert 이벤트 목록 조회.
    Exception은 저장하지 않으므로 여기에는 낙상 알림만 표시됩니다.
    """
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
    """
    대시보드 카드용 통계.
    DB에는 Fall Alert만 저장하는 정책입니다.
    """
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
    """
    테스트 중 쌓인 낙상 이벤트를 전체 삭제합니다.
    """
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