from pathlib import Path
import sys
import json
import joblib
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# =========================
# import 경로 문제 해결
# =========================

# 현재 파일 위치: C:\smart_care_ai\rf\evaluate_model.py
CURRENT_DIR = Path(__file__).resolve().parent

# 프로젝트 루트: C:\smart_care_ai
ROOT_DIR = CURRENT_DIR.parent

# Python이 C:\smart_care_ai 기준으로 rf 패키지를 찾게 함
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from rf.feature_extractor import extract_features_from_csv
from rf.predict_csv import postprocess_fall_result, get_fall_probability


# =========================
# 경로 설정
# =========================

DATA_DIR = ROOT_DIR / "data" / "mmwave_fall" / "GatheredData"
FALL_DIR = DATA_DIR / "Fall"
NOT_DIR = DATA_DIR / "Not"

MODEL_CANDIDATES = [
    ROOT_DIR / "models" / "mmwave_rf_aug_model.pkl",
    ROOT_DIR / "models" / "mmwave_rf_model.pkl",
    ROOT_DIR / "rf" / "models" / "mmwave_rf_aug_model.pkl",
    ROOT_DIR / "rf" / "models" / "mmwave_rf_model.pkl",
]

META_CANDIDATES = [
    ROOT_DIR / "models" / "mmwave_rf_aug_model_meta.json",
    ROOT_DIR / "models" / "mmwave_rf_model_meta.json",
    ROOT_DIR / "rf" / "models" / "mmwave_rf_aug_model_meta.json",
    ROOT_DIR / "rf" / "models" / "mmwave_rf_model_meta.json",
]

REPORT_DIR = ROOT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "eval_results.csv"


# =========================
# 모델 / 메타 파일 찾기
# =========================

def find_existing_path(paths: list[Path], name: str) -> Path:
    for path in paths:
        if path.exists():
            return path

    checked = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(
        f"{name} 파일을 찾을 수 없습니다.\n확인한 경로:\n{checked}"
    )


def load_model_and_meta():
    model_path = find_existing_path(MODEL_CANDIDATES, "모델")
    model = joblib.load(model_path)

    meta = {}

    for meta_path in META_CANDIDATES:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"메타 파일 사용: {meta_path}")
            break

    print(f"모델 파일 사용: {model_path}")

    return model, meta, model_path


def get_feature_names(model, meta: dict, feat: dict) -> list[str]:
    """
    학습 때 사용한 feature 순서를 가져옵니다.
    우선순위:
    1. meta["feature_names"]
    2. model.feature_names_in_
    3. 현재 추출된 feature 이름
    """
    if meta and "feature_names" in meta:
        return list(meta["feature_names"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return sorted(feat.keys())


# =========================
# CSV 1개 예측
# =========================

def predict_with_loaded_model(csv_path: Path, model, meta: dict) -> dict:
    feat = extract_features_from_csv(csv_path)

    feature_names = get_feature_names(model, meta, feat)

    X = pd.DataFrame([feat])

    # 학습 때 사용한 feature가 현재 CSV에 없으면 0으로 채움
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0

    # 학습 순서와 동일하게 정렬
    X = X[feature_names].fillna(0.0)

    # Fall 확률 계산
    fall_prob, pred_label = get_fall_probability(model, X)

    # 최신 후처리 기준 적용
    result = postprocess_fall_result(
        fall_prob=fall_prob,
        features=feat,
        file_name=csv_path.name,
    )

    result.update({
        "file_name": csv_path.name,
        "file_path": str(csv_path),
        "raw_model_fall_prob": round(float(fall_prob), 4),
        "model_pred_label": str(pred_label),
        "speed_max": round(float(result.get("speed_max", feat.get("speed_max", 0.0))), 4),
        "height_drop": round(float(result.get("height_drop", feat.get("height_drop", 0.0))), 4),
        "movement_after": round(float(result.get("movement_after", feat.get("movement_after", 0.0))), 4),
    })

    return result


# =========================
# 평가 실행
# =========================

def main():
    if not FALL_DIR.exists():
        raise FileNotFoundError(f"Fall 폴더가 없습니다: {FALL_DIR}")

    if not NOT_DIR.exists():
        raise FileNotFoundError(f"Not 폴더가 없습니다: {NOT_DIR}")

    fall_files = sorted(FALL_DIR.glob("*.csv"))
    not_files = sorted(NOT_DIR.glob("*.csv"))

    if not fall_files:
        raise FileNotFoundError(f"Fall CSV 파일이 없습니다: {FALL_DIR}")

    if not not_files:
        raise FileNotFoundError(f"Not CSV 파일이 없습니다: {NOT_DIR}")

    model, meta, model_path = load_model_and_meta()

    dataset = []

    # 실제 정답:
    # Fall 폴더 = 1
    # Not 폴더 = 0
    for path in fall_files:
        dataset.append((path, 1, "Fall"))

    for path in not_files:
        dataset.append((path, 0, "Normal"))

    print()
    print("===== 평가 시작 =====")
    print(f"Fall CSV 개수: {len(fall_files)}")
    print(f"Normal CSV 개수: {len(not_files)}")
    print(f"전체 CSV 개수: {len(dataset)}")
    print()

    rows = []
    y_true = []
    y_pred = []

    for csv_path, true_label, true_name in dataset:
        try:
            result = predict_with_loaded_model(csv_path, model, meta)

            # Fall Alert이면 낙상으로 판단
            pred_label = 1 if result.get("alert") is True else 0
            pred_name = "Fall" if pred_label == 1 else "Normal"

            is_correct = true_label == pred_label

            y_true.append(true_label)
            y_pred.append(pred_label)

            rows.append({
                "file_name": csv_path.name,
                "true_folder": true_name,
                "true_label": true_label,
                "pred_status": result.get("status"),
                "pred_alert": result.get("alert"),
                "pred_label": pred_label,
                "pred_name": pred_name,
                "correct": is_correct,
                "fall_prob": result.get("fall_prob"),
                "raw_fall_prob": result.get("raw_fall_prob"),
                "raw_model_fall_prob": result.get("raw_model_fall_prob"),
                "speed_max": result.get("speed_max"),
                "height_drop": result.get("height_drop"),
                "movement_after": result.get("movement_after"),
                "model_pred_label": result.get("model_pred_label"),
                "message": result.get("message"),
                "file_path": str(csv_path),
            })

        except Exception as e:
            y_true.append(true_label)
            y_pred.append(0)

            rows.append({
                "file_name": csv_path.name,
                "true_folder": true_name,
                "true_label": true_label,
                "pred_status": "Error",
                "pred_alert": False,
                "pred_label": 0,
                "pred_name": "Error",
                "correct": False,
                "fall_prob": None,
                "raw_fall_prob": None,
                "raw_model_fall_prob": None,
                "speed_max": None,
                "height_drop": None,
                "movement_after": None,
                "model_pred_label": None,
                "message": str(e),
                "file_path": str(csv_path),
            })

    result_df = pd.DataFrame(rows)

    total = len(result_df)
    correct_count = int(result_df["correct"].sum())
    wrong_count = total - correct_count
    accuracy = accuracy_score(y_true, y_pred)

    print("===== 최종 결과 =====")
    print(f"전체 개수: {total}")
    print(f"맞은 개수: {correct_count}")
    print(f"틀린 개수: {wrong_count}")
    print(f"정확도: {accuracy:.4f}")
    print()

    print("===== 분류 리포트 =====")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Fall"],
            zero_division=0,
        )
    )

    print("===== 혼동 행렬 =====")
    print("행: 실제값, 열: 예측값")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(confusion_matrix(y_true, y_pred))
    print()

    wrong_df = result_df[result_df["correct"] == False]

    print("===== 틀린 파일 목록 =====")
    if len(wrong_df) == 0:
        print("틀린 파일 없음")
    else:
        for _, row in wrong_df.iterrows():
            print(
                f"- {row['file_name']} | 실제: {row['true_folder']} | "
                f"예측: {row['pred_status']} | fall_prob={row['fall_prob']} | "
                f"raw_model_prob={row['raw_model_fall_prob']} | "
                f"speed_max={row['speed_max']} | "
                f"height_drop={row['height_drop']} | "
                f"movement_after={row['movement_after']}"
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    print()
    print(f"상세 결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    main()