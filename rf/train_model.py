from pathlib import Path
import json
import joblib
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


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rows = []

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
        feat["file_name"] = path.name
        rows.append(feat)

    for path in not_files:
        feat = extract_features_from_csv(path)
        feat["label"] = 0
        feat["file_name"] = path.name
        rows.append(feat)

    df = pd.DataFrame(rows)

    y = df["label"]
    X = df.drop(columns=["label", "file_name"])
    X = X.fillna(0)

    return X, y


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    print("전체 데이터 shape:", X.shape)
    print("라벨 분포:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== 평가 결과 =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fall"]))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_names": list(X.columns),
        "label_map": {
            "0": "Normal",
            "1": "Fall"
        },
        "data_dir": str(DATA_DIR),
        "model_path": str(MODEL_PATH),
        "fall_threshold": 0.60,
        "fall_alert_threshold": 0.70
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료: {MODEL_PATH}")
    print(f"메타정보 저장 완료: {META_PATH}")


if __name__ == "__main__":
    main()
