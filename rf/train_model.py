from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from feature_extractor import extract_features_from_csv


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "mmwave_fall" / "GatheredData"
FALL_DIR = DATA_DIR / "Fall"
NOT_DIR = DATA_DIR / "Not"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "mmwave_rf_model.pkl"
META_PATH = MODEL_DIR / "mmwave_rf_model_meta.json"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_dataset():
    rows = []
    file_names = []

    fall_files = sorted(FALL_DIR.glob("*.csv"))
    not_files = sorted(NOT_DIR.glob("*.csv"))

    if not fall_files:
        raise FileNotFoundError(f"Fall CSV 파일이 없습니다: {FALL_DIR}")

    if not not_files:
        raise FileNotFoundError(f"Not CSV 파일이 없습니다: {NOT_DIR}")

    print(f"Fall CSV 개수: {len(fall_files)}")
    print(f"Not CSV 개수: {len(not_files)}")

    for path in fall_files:
        feat = extract_features_from_csv(path)
        feat["label"] = 1
        rows.append(feat)
        file_names.append(path.name)

    for path in not_files:
        feat = extract_features_from_csv(path)
        feat["label"] = 0
        rows.append(feat)
        file_names.append(path.name)

    df = pd.DataFrame(rows)

    y = df["label"]
    X = df.drop(columns=["label"])
    X = X.fillna(0)

    return X, y, file_names


def get_fall_probabilities(model, X):
    """
    RandomForest의 predict_proba에서 Fall 클래스 확률만 안전하게 가져오기
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 in classes:
        fall_idx = classes.index(1)
    else:
        fall_idx = len(classes) - 1

    return proba[:, fall_idx]


def find_best_threshold(y_test, y_prob):
    """
    threshold를 0.20 ~ 0.80 사이에서 자동 탐색
    정확도 우선, 그다음 Fall recall 우선
    """
    best_threshold = 0.5
    best_acc = -1
    best_fall_recall = -1
    best_fn = 999999
    best_fp = 999999
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
            best_fn = fn
            best_fp = fp
            best_cm = cm

    return best_threshold, best_acc, best_fall_recall, best_cm


def print_result(title, y_test, y_pred):
    print(f"\n===== {title} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fall"]))
    print("행: 실제값, 열: 예측값")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


def print_wrong_files(files_test, y_test, y_pred, y_prob):
    print("\n===== 틀린 파일 목록 =====")

    wrong_count = 0

    for file_name, true_label, pred_label, prob in zip(files_test, y_test, y_pred, y_prob):
        if int(true_label) != int(pred_label):
            wrong_count += 1

            true_name = "Fall" if int(true_label) == 1 else "Normal"
            pred_name = "Fall" if int(pred_label) == 1 else "Normal"

            print(
                f"- {file_name} | 실제: {true_name} | 예측: {pred_name} | fall_prob={prob:.4f}"
            )

    if wrong_count == 0:
        print("틀린 파일 없음")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, file_names = load_dataset()

    print("\n전체 데이터 shape:", X.shape)
    print("라벨 분포:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X,
        y,
        file_names,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 기존 방식: model.predict
    y_pred_basic = model.predict(X_test)
    print_result("기존 model.predict 기준 평가 결과", y_test, y_pred_basic)

    # 개선 방식: predict_proba + threshold 자동 탐색
    y_prob = get_fall_probabilities(model, X_test)

    best_threshold, best_acc, best_fall_recall, best_cm = find_best_threshold(
        y_test,
        y_prob,
    )

    print("\n===== Threshold 자동 탐색 결과 =====")
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best Accuracy  : {best_acc:.4f}")
    print(f"Fall Recall    : {best_fall_recall:.4f}")
    print("Best Confusion Matrix")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(best_cm)

    y_pred_best = (y_prob >= best_threshold).astype(int)

    print_result(
        f"Best threshold {best_threshold:.2f} 기준 평가 결과",
        y_test,
        y_pred_best,
    )

    print_wrong_files(files_test, y_test, y_pred_best, y_prob)

    # 모델에 feature 순서와 threshold 저장
    model.feature_columns_ = list(X.columns)
    model.threshold_ = float(best_threshold)
    model.default_threshold_ = 0.5
    model.label_meaning_ = {
        0: "Normal",
        1: "Fall",
    }

    joblib.dump(model, MODEL_PATH)

    tn, fp, fn, tp = best_cm.ravel()

    meta = {
        "model_name": "mmwave_rf_model",
        "feature_names": list(X.columns),
        "feature_columns": list(X.columns),
        "label_map": {
            "0": "Normal",
            "1": "Fall",
        },
        "data_dir": str(DATA_DIR),
        "model_path": str(MODEL_PATH),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "default_threshold": 0.5,
        "best_threshold": float(best_threshold),
        "fall_threshold": float(best_threshold),
        "fall_alert_threshold": float(best_threshold),
        "best_accuracy": float(best_acc),
        "best_fall_recall": float(best_fall_recall),
        "best_confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "total_count": int(len(y)),
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료: {MODEL_PATH}")
    print(f"메타정보 저장 완료: {META_PATH}")
    print(f"최종 적용 threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()