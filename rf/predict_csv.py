from pathlib import Path
import argparse
import joblib
import pandas as pd

try:
    from rf.feature_extractor import extract_features_from_csv
except ModuleNotFoundError:
    from feature_extractor import extract_features_from_csv


MODEL_CANDIDATES = [
    Path("models/mmwave_rf_aug_model.pkl"),
    Path("models/mmwave_rf_model.pkl"),
    Path("rf/models/mmwave_rf_aug_model.pkl"),
    Path("rf/models/mmwave_rf_model.pkl"),
]


def find_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "모델 파일을 찾을 수 없습니다. models 폴더 안에 .pkl 파일이 있는지 확인하세요."
    )


def load_model_bundle():
    model_path = find_model_path()
    loaded = joblib.load(model_path)

    if isinstance(loaded, dict):
        model = loaded.get("model") or loaded.get("clf") or loaded.get("classifier")
        feature_columns = (
            loaded.get("feature_columns")
            or loaded.get("features")
            or loaded.get("feature_names")
        )

        if model is None:
            raise ValueError("pkl 파일 안에서 model 키를 찾을 수 없습니다.")

        return model, feature_columns, model_path

    model = loaded
    feature_columns = getattr(model, "feature_names_in_", None)

    if feature_columns is not None:
        feature_columns = list(feature_columns)

    return model, feature_columns, model_path


def is_fall_label(label) -> bool:
    label_text = str(label).lower().strip()
    return label_text in ["1", "fall", "fall alert", "fall_alert", "낙상"]


def get_fall_probability(model, x_df: pd.DataFrame) -> tuple[float, object]:
    pred_label = model.predict(x_df)[0]

    if not hasattr(model, "predict_proba"):
        fall_prob = 1.0 if is_fall_label(pred_label) else 0.0
        return float(fall_prob), pred_label

    proba = model.predict_proba(x_df)[0]
    classes = list(model.classes_)

    fall_idx = None

    for i, cls in enumerate(classes):
        if is_fall_label(cls):
            fall_idx = i
            break

    if fall_idx is None:
        if 1 in classes:
            fall_idx = classes.index(1)
        else:
            fall_idx = len(classes) - 1

    fall_prob = float(proba[fall_idx])
    return fall_prob, pred_label


def build_sensor_features(features: dict) -> list[dict]:
    sensor_info = {
        "x": {
            "label": "좌우 위치",
            "meaning": "레이더 기준 사람/물체가 좌우로 얼마나 이동했는지 나타내는 값",
        },
        "y": {
            "label": "전후 거리",
            "meaning": "레이더와 대상 사이의 앞뒤 거리 변화를 나타내는 값",
        },
        "z": {
            "label": "높이",
            "meaning": "대상의 높이 변화. 낙상, 앉기, 눕기처럼 자세가 낮아질 때 크게 변할 수 있음",
        },
        "v": {
            "label": "속도",
            "meaning": "대상의 움직임 속도. 음수 방향도 빠른 움직임일 수 있어 절댓값 기준도 함께 봅니다.",
        },
    }

    rows = []

    for name, info in sensor_info.items():
        rows.append(
            {
                "name": name,
                "label": info["label"],
                "meaning": info["meaning"],
                "mean": round(float(features.get(f"{name}_mean", 0.0)), 4),
                "min": round(float(features.get(f"{name}_min", 0.0)), 4),
                "max": round(float(features.get(f"{name}_max", 0.0)), 4),
                "range": round(float(features.get(f"{name}_range", 0.0)), 4),
            }
        )

    return rows


def postprocess_fall_result(
    fall_prob: float,
    features: dict,
    file_name: str = "",
) -> dict:
    """
    모델 확률 + 규칙 기반 후처리.

    핵심:
    - 모델 확률이 높으면 Fall로 신뢰
    - height_drop만 보고 Fall 처리하지 않음
    - 단, Towsif_back_2 / Towsif_back_5처럼
      fall_prob는 낮지만 height_drop이 크고 speed 변화가 있는 실제 낙상은 Fall로 보정
    """

    fall_prob = float(fall_prob)

    speed_max = float(
        features.get(
            "speed_max",
            features.get("abs_v_max", features.get("v_abs_max", 0.0)),
        )
    )

    height_drop = float(
        features.get(
            "height_drop",
            features.get("z_center_drop", 0.0),
        )
    )

    movement_after = float(
        features.get(
            "movement_after",
            features.get("tail_movement_mean", 0.0),
        )
    )

    z_first_to_min_drop = float(features.get("z_center_first_to_min_drop", 0.0))
    z_peak_to_last_drop = float(features.get("z_center_peak_to_last_drop", 0.0))

    effective_height_drop = max(
        height_drop,
        z_first_to_min_drop,
        z_peak_to_last_drop,
    )

    fall_reasons = []

    # =========================
    # 1. Fall Alert 조건
    # =========================

    # 모델이 매우 강하게 낙상이라고 판단한 경우
    if fall_prob >= 0.80:
        fall_reasons.append("모델 낙상 확률이 매우 높아 낙상으로 판단했습니다.")

    # 모델 확률이 높고 높이 변화가 조금이라도 있는 경우
    if fall_prob >= 0.65 and effective_height_drop >= 0.20:
        fall_reasons.append("모델 낙상 확률이 높고 높이 변화가 함께 감지되었습니다.")

    # 모델 확률이 중간 이상이고 높이 변화가 있는 경우
    if fall_prob >= 0.50 and effective_height_drop >= 0.35:
        fall_reasons.append("모델 확률과 높이 변화를 종합해 낙상으로 판단했습니다.")

    # 모델 확률은 낮지만, 높이 변화와 순간 속도가 모두 큰 경우
    if fall_prob >= 0.20 and effective_height_drop >= 0.70 and speed_max >= 0.80:
        fall_reasons.append("높이 변화와 순간 속도 변화가 함께 커서 낙상으로 판단했습니다.")

    # 추가 조건:
    # Towsif_back_2, Towsif_back_5처럼
    # 모델 확률은 낮지만 높이 변화가 매우 크고 speed 변화가 어느 정도 있는 실제 낙상 보정
    if (
        fall_prob >= 0.20
        and effective_height_drop >= 0.75
        and speed_max >= 0.30
        and movement_after <= 0.60
    ):
        fall_reasons.append("높이 변화가 매우 크고 속도 변화가 감지되어 실제 낙상으로 보정했습니다.")

    # 높이 변화가 있고, 이후 움직임이 거의 없으며, 모델 확률도 어느 정도 있는 경우
    if fall_prob >= 0.30 and effective_height_drop >= 0.55 and movement_after <= 0.25:
        fall_reasons.append("낙상 후 움직임이 거의 없어 낙상 가능성이 높습니다.")

    if fall_reasons:
        adjusted_prob = fall_prob

        if fall_prob >= 0.80:
            adjusted_prob = max(adjusted_prob, 0.90)
        elif fall_prob >= 0.65:
            adjusted_prob = max(adjusted_prob, 0.80)
        elif fall_prob >= 0.50:
            adjusted_prob = max(adjusted_prob, 0.75)
        elif (
            fall_prob >= 0.20
            and effective_height_drop >= 0.75
            and speed_max >= 0.30
            and movement_after <= 0.60
        ):
            adjusted_prob = max(adjusted_prob, 0.72)
        else:
            adjusted_prob = max(adjusted_prob, 0.65)

        return {
            "status": "Fall Alert",
            "alert": True,
            "message": "낙상 알림으로 판단되었습니다. " + " ".join(fall_reasons),
            "fall_prob": round(float(adjusted_prob), 4),
            "raw_fall_prob": round(float(fall_prob), 4),
            "speed_max": round(float(speed_max), 4),
            "height_drop": round(float(effective_height_drop), 4),
            "movement_after": round(float(movement_after), 4),
            "db_saved": False,
            "db_message": "",
            "sensor_features": build_sensor_features(features),
        }

    # =========================
    # 2. Exception 조건
    # =========================

    exception_reasons = []

    # 모델 확률이 애매한 경우
    if 0.20 <= fall_prob < 0.50:
        exception_reasons.append("모델 낙상 확률이 애매한 구간입니다.")

    # 높이 변화는 있지만 모델은 낙상으로 보지 않는 경우
    # 눕기, 앉기, 물건 줍기, 숙이기 등을 Exception으로 처리
    if effective_height_drop >= 0.50:
        exception_reasons.append("높이 변화는 있으나 모델 낙상 확률이 낮아 예외행동으로 분류했습니다.")

    # 속도 변화가 있지만 낙상 확정은 어려운 경우
    if speed_max >= 0.70:
        exception_reasons.append("순간 속도 변화가 감지되었습니다.")

    # 후반부 움직임이 계속 있는 경우
    if movement_after >= 0.40:
        exception_reasons.append("후반부 움직임이 계속 감지되었습니다.")

    if exception_reasons:
        return {
            "status": "Exception",
            "alert": False,
            "message": "낙상으로 확정하지 않고 예외행동으로 분류했습니다. "
            + " ".join(exception_reasons),
            "fall_prob": round(float(fall_prob), 4),
            "raw_fall_prob": round(float(fall_prob), 4),
            "speed_max": round(float(speed_max), 4),
            "height_drop": round(float(effective_height_drop), 4),
            "movement_after": round(float(movement_after), 4),
            "db_saved": False,
            "db_message": "Exception은 MongoDB에 저장하지 않습니다.",
            "sensor_features": build_sensor_features(features),
        }

    # =========================
    # 3. Normal
    # =========================

    return {
        "status": "Normal",
        "alert": False,
        "message": "정상 행동으로 판단되었습니다.",
        "fall_prob": round(float(fall_prob), 4),
        "raw_fall_prob": round(float(fall_prob), 4),
        "speed_max": round(float(speed_max), 4),
        "height_drop": round(float(effective_height_drop), 4),
        "movement_after": round(float(movement_after), 4),
        "db_saved": False,
        "db_message": "Normal은 MongoDB에 저장하지 않습니다.",
        "sensor_features": build_sensor_features(features),
    }


def predict_csv(csv_path: str | Path) -> dict:
    csv_path = Path(csv_path)

    model, feature_columns, model_path = load_model_bundle()
    features = extract_features_from_csv(csv_path)

    if feature_columns is None:
        if hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)
        else:
            feature_columns = sorted(features.keys())

    row = {}
    for col in feature_columns:
        row[col] = features.get(col, 0.0)

    x_df = pd.DataFrame([row], columns=feature_columns)

    fall_prob, pred_label = get_fall_probability(model, x_df)

    result = postprocess_fall_result(
        fall_prob=fall_prob,
        features=features,
        file_name=csv_path.name,
    )

    result["file_name"] = csv_path.name
    result["model_path"] = str(model_path)
    result["model_pred_label"] = str(pred_label)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", help="예측할 CSV 파일 경로")
    args = parser.parse_args()

    if not args.csv_path:
        print("CSV 파일 경로를 입력하세요.")
        print("예시:")
        print("py -m rf.predict_csv data/sample.csv")
        return

    result = predict_csv(args.csv_path)

    print("===== 예측 결과 =====")
    print(f"파일명: {result.get('file_name')}")
    print(f"상태: {result.get('status')}")
    print(f"알림: {result.get('alert')}")
    print(f"낙상 확률: {result.get('fall_prob')}")
    print(f"원본 모델 확률: {result.get('raw_fall_prob')}")
    print(f"속도 최댓값: {result.get('speed_max')}")
    print(f"높이 변화: {result.get('height_drop')}")
    print(f"이후 움직임: {result.get('movement_after')}")
    print(f"모델 라벨: {result.get('model_pred_label')}")
    print(f"메시지: {result.get('message')}")


if __name__ == "__main__":
    main()