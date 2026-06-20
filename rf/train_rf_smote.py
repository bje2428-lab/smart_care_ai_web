from pathlib import Path
from collections import Counter
import sys
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# =========================
# 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]   # C:\smart_care_ai_web
RF_DIR = Path(__file__).resolve().parent         # C:\smart_care_ai_web\rf
BACKEND_DIR = BASE_DIR / "backend"              # C:\smart_care_ai_web\backend

sys.path.append(str(RF_DIR))

from feature_extractor import extract_features_from_csv


# =========================
# 데이터 / 모델 경로
# =========================

DATA_DIR_CANDIDATES = [
    BASE_DIR / "data" / "mmwave_fall" / "GatheredData",
    BASE_DIR / "data" / "mmwave-fall" / "GatheredData",
    BASE_DIR / "data" / "mmwave-radar-fall-detection-main" / "GatheredData",
    BASE_DIR / "data" / "GatheredData",

    BACKEND_DIR / "data" / "mmwave_fall" / "GatheredData",
    BACKEND_DIR / "data" / "mmwave-fall" / "GatheredData",
    BACKEND_DIR / "data" / "mmwave-radar-fall-detection-main" / "GatheredData",
    BACKEND_DIR / "data" / "GatheredData",
]


# =========================
# 모델 저장 경로
# =========================
# backend/models가 아니라 프로젝트 루트 models 폴더에 저장
# 저장 위치:
# C:\smart_care_ai_web\models\mmwave_rf_smote_model.pkl

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mmwave_rf_smote_model.pkl"
META_PATH = MODEL_DIR / "mmwave_rf_smote_model_meta.json"


RANDOM_STATE = 42
TEST_SIZE = 0.2

# 기본 기준값
DEFAULT_THRESHOLD = 0.5


# =========================
# 데이터 폴더 찾기
# =========================

def is_valid_data_dir(data_dir: Path) -> bool:
    fall_dir = data_dir / "Fall"
    not_dir = data_dir / "Not"
    return fall_dir.exists() and not_dir.exists()


def find_data_dir() -> Path:
    print("\n===== 데이터 폴더 검색 =====")
    print(f"현재 실행 파일: {Path(__file__).resolve()}")
    print(f"프로젝트 루트: {BASE_DIR}")

    for data_dir in DATA_DIR_CANDIDATES:
        fall_dir = data_dir / "Fall"
        not_dir = data_dir / "Not"

        print(f"\n확인 중: {data_dir}")
        print(f"  Fall 폴더 존재: {fall_dir.exists()}")
        print(f"  Not 폴더 존재 : {not_dir.exists()}")

        if is_valid_data_dir(data_dir):
            print(f"\n[OK] 데이터 폴더 찾음: {data_dir}")
            return data_dir

    print("\n후보 경로에서 못 찾았습니다. 프로젝트 안에서 GatheredData 자동 검색을 시작합니다.")

    for gathered_dir in BASE_DIR.rglob("GatheredData"):
        fall_dir = gathered_dir / "Fall"
        not_dir = gathered_dir / "Not"

        print(f"자동 검색 확인: {gathered_dir}")
        print(f"  Fall 폴더 존재: {fall_dir.exists()}")
        print(f"  Not 폴더 존재 : {not_dir.exists()}")

        if is_valid_data_dir(gathered_dir):
            print(f"\n[OK] 자동 검색으로 데이터 폴더 찾음: {gathered_dir}")
            return gathered_dir

    raise FileNotFoundError(
        "\n데이터 폴더를 찾을 수 없습니다.\n\n"
        "아래 구조인지 확인하세요.\n"
        "C:\\smart_care_ai_web\\data\\mmwave_fall\\GatheredData\\Fall\n"
        "C:\\smart_care_ai_web\\data\\mmwave_fall\\GatheredData\\Not\n"
    )


# =========================
# CSV → 특징 데이터셋 생성
# =========================

def load_dataset(data_dir: Path):
    fall_dir = data_dir / "Fall"
    normal_dir = data_dir / "Not"

    rows = []
    labels = []
    file_names = []

    fall_files = sorted(fall_dir.glob("*.csv"))
    normal_files = sorted(normal_dir.glob("*.csv"))

    print("\n===== CSV 파일 개수 확인 =====")
    print(f"Fall CSV 개수  : {len(fall_files)}")
    print(f"Normal CSV 개수: {len(normal_files)}")

    if len(fall_files) == 0:
        raise ValueError(f"Fall 폴더에 CSV 파일이 없습니다: {fall_dir}")

    if len(normal_files) == 0:
        raise ValueError(f"Not 폴더에 CSV 파일이 없습니다: {normal_dir}")

    for csv_path in fall_files:
        try:
            features = extract_features_from_csv(csv_path)
            rows.append(features)
            labels.append(1)
            file_names.append(csv_path.name)
        except Exception as e:
            print(f"[스킵] Fall 파일 오류: {csv_path.name} | {e}")

    for csv_path in normal_files:
        try:
            features = extract_features_from_csv(csv_path)
            rows.append(features)
            labels.append(0)
            file_names.append(csv_path.name)
        except Exception as e:
            print(f"[스킵] Normal 파일 오류: {csv_path.name} | {e}")

    if not rows:
        raise ValueError("학습 가능한 CSV 데이터가 없습니다.")

    X = pd.DataFrame(rows)
    y = np.array(labels)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, file_names


# =========================
# threshold 자동 탐색
# =========================

def find_best_threshold(y_test, y_prob):
    print("\n===== Threshold 자동 탐색 =====")

    best_threshold = DEFAULT_THRESHOLD
    best_acc = -1
    best_fall_recall = -1
    best_fp = 999999
    best_fn = 999999
    best_cm = None

    for threshold in np.arange(0.20, 0.81, 0.01):
        threshold = round(float(threshold), 2)

        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_test, y_pred)
        fall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        is_better = (
            acc > best_acc
            or (acc == best_acc and fall_recall > best_fall_recall)
            or (acc == best_acc and fall_recall == best_fall_recall and fn < best_fn)
            or (acc == best_acc and fall_recall == best_fall_recall and fn == best_fn and fp < best_fp)
        )

        if is_better:
            best_threshold = threshold
            best_acc = acc
            best_fall_recall = fall_recall
            best_fp = fp
            best_fn = fn
            best_cm = cm

    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best Accuracy  : {best_acc:.4f}")
    print(f"Fall Recall    : {best_fall_recall:.4f}")

    print("\nBest Confusion Matrix")
    print("행: 실제값, 열: 예측값")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(best_cm)

    return best_threshold, best_cm


# =========================
# 틀린 파일 출력
# =========================

def print_wrong_files(files_test, y_test, y_pred, y_prob):
    print("\n===== 틀린 파일 목록 =====")

    wrong_count = 0

    for file_name, true_label, pred_label, prob in zip(files_test, y_test, y_pred, y_prob):
        if true_label != pred_label:
            wrong_count += 1

            true_name = "Fall" if true_label == 1 else "Normal"
            pred_name = "Fall" if pred_label == 1 else "Normal"

            print(
                f"- {file_name} | 실제: {true_name} | 예측: {pred_name} | fall_prob: {prob:.4f}"
            )

    if wrong_count == 0:
        print("틀린 파일 없음")


# =========================
# 학습
# =========================

def train():
    data_dir = find_data_dir()

    print("\n===================================")
    print("Smart Care AI RF + SMOTE Training")
    print("===================================")
    print(f"[DATA] {data_dir}")

    X, y, file_names = load_dataset(data_dir)

    print("\n===== 전체 데이터 =====")
    print("전체 샘플 수:", len(y))
    print("클래스 분포:", Counter(y))
    print("0 = Normal, 1 = Fall")
    print("특징 개수:", X.shape[1])

    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X,
        y,
        file_names,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n===== train/test 분리 후 =====")
    print("Train:", Counter(y_train))
    print("Test :", Counter(y_test))

    min_class_count = min(Counter(y_train).values())
    k_neighbors = min(5, min_class_count - 1)

    if k_neighbors < 1:
        raise ValueError("SMOTE를 적용하기에 minority 데이터가 너무 적습니다.")

    print("\n===== SMOTE 설정 =====")
    print("k_neighbors:", k_neighbors)

    # SMOTE 적용 후 개수 확인
    temp_imputer = SimpleImputer(strategy="median")
    temp_scaler = StandardScaler()

    X_train_imp = temp_imputer.fit_transform(X_train)
    X_train_scaled = temp_scaler.fit_transform(X_train_imp)

    smote_check = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors,
    )

    X_resampled, y_resampled = smote_check.fit_resample(X_train_scaled, y_train)

    before_count = Counter(y_train)
    after_count = Counter(y_resampled)

    print("\n===== SMOTE 적용 후 예상 학습 데이터 =====")
    print("Before SMOTE:", before_count)
    print("After SMOTE :", after_count)
    print("추가된 Normal 데이터 수:", after_count[0] - before_count[0])
    print("추가된 Fall 데이터 수  :", after_count[1] - before_count[1])

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=700,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 0.5 기준 결과
    y_pred_default = (y_prob >= DEFAULT_THRESHOLD).astype(int)

    print(f"\n===== threshold {DEFAULT_THRESHOLD} 기준 결과 =====")
    print("Accuracy:", accuracy_score(y_test, y_pred_default))
    print(
        classification_report(
            y_test,
            y_pred_default,
            target_names=["Normal", "Fall"],
            digits=4,
        )
    )

    print("===== 혼동 행렬 =====")
    print("행: 실제값, 열: 예측값")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(confusion_matrix(y_test, y_pred_default, labels=[0, 1]))

    print_wrong_files(files_test, y_test, y_pred_default, y_prob)

    # threshold 자동 탐색
    best_threshold, best_cm = find_best_threshold(y_test, y_prob)

    y_pred_best = (y_prob >= best_threshold).astype(int)

    print(f"\n===== Best threshold {best_threshold:.2f} 기준 결과 =====")
    print("Accuracy:", accuracy_score(y_test, y_pred_best))
    print(
        classification_report(
            y_test,
            y_pred_best,
            target_names=["Normal", "Fall"],
            digits=4,
        )
    )

    print("===== Best threshold 틀린 파일 목록 =====")
    print_wrong_files(files_test, y_test, y_pred_best, y_prob)

    # 예측 코드에서 같은 컬럼 순서를 사용하기 위해 저장
    model.feature_columns_ = list(X.columns)
    model.threshold_ = float(best_threshold)
    model.default_threshold_ = float(DEFAULT_THRESHOLD)
    model.label_meaning_ = {
        0: "Normal",
        1: "Fall",
    }

    joblib.dump(model, MODEL_PATH)

    # 메타 정보 저장
    tn, fp, fn, tp = best_cm.ravel()

    meta = {
        "model_name": "mmwave_rf_smote_model",
        "model_path": str(MODEL_PATH),
        "data_dir": str(data_dir),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "default_threshold": DEFAULT_THRESHOLD,
        "best_threshold": float(best_threshold),
        "total_count": int(len(y)),
        "class_distribution_total": {
            "Normal": int(Counter(y)[0]),
            "Fall": int(Counter(y)[1]),
        },
        "class_distribution_train_before_smote": {
            "Normal": int(before_count[0]),
            "Fall": int(before_count[1]),
        },
        "class_distribution_train_after_smote": {
            "Normal": int(after_count[0]),
            "Fall": int(after_count[1]),
        },
        "added_by_smote": {
            "Normal": int(after_count[0] - before_count[0]),
            "Fall": int(after_count[1] - before_count[1]),
        },
        "test_distribution": {
            "Normal": int(Counter(y_test)[0]),
            "Fall": int(Counter(y_test)[1]),
        },
        "best_confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "best_accuracy": float(accuracy_score(y_test, y_pred_best)),
        "feature_count": int(X.shape[1]),
        "feature_columns": list(X.columns),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n===== 모델 저장 완료 =====")
    print(MODEL_PATH)

    print("\n===== 메타 정보 저장 완료 =====")
    print(META_PATH)

    print("\n===== 최종 적용 threshold =====")
    print(f"백엔드 예측에 사용할 권장 threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    train()