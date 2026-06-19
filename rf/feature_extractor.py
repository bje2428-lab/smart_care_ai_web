from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_BASE_COLS = ["x", "y", "z", "v"]
OPTIONAL_COLS = ["frame", "DetObj#", "snr", "noise"]


def _read_csv_safely(csv_path: Path) -> pd.DataFrame:
    """
    CSV를 안전하게 읽고 컬럼명을 정리합니다.
    """
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        low = col.lower().strip()

        if low in ["detobj", "detobj#"]:
            rename_map[col] = "DetObj#"
        elif low in ["frameid", "frame_id"]:
            rename_map[col] = "frame"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    필요한 컬럼을 숫자형으로 변환합니다.
    변환 불가능한 값은 NaN 처리합니다.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_stats(features: dict, prefix: str, values) -> None:
    """
    평균, 표준편차, 최소, 최대, 중앙값, 사분위수, 범위 통계 추가
    """
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()

    if len(arr) == 0:
        features[f"{prefix}_mean"] = 0.0
        features[f"{prefix}_std"] = 0.0
        features[f"{prefix}_min"] = 0.0
        features[f"{prefix}_max"] = 0.0
        features[f"{prefix}_median"] = 0.0
        features[f"{prefix}_q25"] = 0.0
        features[f"{prefix}_q75"] = 0.0
        features[f"{prefix}_range"] = 0.0
        return

    features[f"{prefix}_mean"] = float(arr.mean())
    features[f"{prefix}_std"] = float(arr.std(ddof=0))
    features[f"{prefix}_min"] = float(arr.min())
    features[f"{prefix}_max"] = float(arr.max())
    features[f"{prefix}_median"] = float(arr.median())
    features[f"{prefix}_q25"] = float(arr.quantile(0.25))
    features[f"{prefix}_q75"] = float(arr.quantile(0.75))
    features[f"{prefix}_range"] = float(arr.max() - arr.min())


def _calc_tail_movement(center_move: pd.Series) -> tuple[float, float]:
    """
    낙상 후 움직임을 보기 위한 후반부 이동량 계산
    """
    if len(center_move) == 0:
        return 0.0, 0.0

    tail_n = max(2, int(len(center_move) * 0.2))
    tail_move = center_move.tail(tail_n)

    return float(tail_move.mean()), float(tail_move.max())


def extract_features_from_csv(csv_path: str | Path) -> dict:
    """
    mmWave CSV 1개를 고정 길이 feature dict로 변환합니다.

    예상 컬럼:
    frame, DetObj#, x, y, z, v, snr, noise

    핵심 낙상 판단 특징:
    - speed_max: v 절댓값 기준 최대 속도
    - height_drop: z 중심값의 전체 변화량
    - z_center_peak_to_last_drop: 가장 높았던 시점에서 마지막 시점까지 떨어진 정도
    - z_center_first_to_min_drop: 처음 높이에서 최저 높이까지 떨어진 정도
    - movement_after: 후반부 움직임 평균
    """
    csv_path = Path(csv_path)
    df = _read_csv_safely(csv_path)

    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} 파일에 필수 컬럼이 없습니다: {missing}")

    numeric_cols = list(set(REQUIRED_BASE_COLS + OPTIONAL_COLS) & set(df.columns))
    df = _safe_numeric(df, numeric_cols)
    df = df.dropna(subset=REQUIRED_BASE_COLS)

    if len(df) == 0:
        raise ValueError(f"{csv_path.name} 파일에 유효한 수치 데이터가 없습니다.")

    features = {}

    # 전체 포인트 수
    features["total_points"] = float(len(df))

    # frame 없으면 임시 frame 생성
    if "frame" in df.columns:
        features["frame_count"] = float(df["frame"].nunique())
    else:
        df["frame"] = np.arange(len(df))
        features["frame_count"] = float(df["frame"].nunique())

    # DetObj# 통계
    if "DetObj#" in df.columns:
        _add_stats(features, "detobj", df["DetObj#"])
    else:
        _add_stats(features, "detobj", [0.0])

    # 기본 컬럼 통계
    for col in ["x", "y", "z", "v", "snr", "noise"]:
        if col in df.columns:
            _add_stats(features, col, df[col])
        else:
            _add_stats(features, col, [0.0])

    # 속도는 방향이 있으므로 절댓값도 반드시 사용
    abs_v = df["v"].abs()
    _add_stats(features, "abs_v", abs_v)

    # 3차원 거리
    radial_dist = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    _add_stats(features, "radial_dist", radial_dist)

    # frame별 중심점 계산
    frame_center = (
        df.groupby("frame")[["x", "y", "z", "v"]]
        .mean()
        .sort_index()
        .reset_index(drop=True)
    )

    if len(frame_center) >= 2:
        dx = frame_center["x"].diff().fillna(0)
        dy = frame_center["y"].diff().fillna(0)
        dz = frame_center["z"].diff().fillna(0)

        center_move = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        _add_stats(features, "center_move", center_move)

        z_series = frame_center["z"]

        z_max = float(z_series.max())
        z_min = float(z_series.min())
        z_first = float(z_series.iloc[0])
        z_last = float(z_series.iloc[-1])

        # 전체 높이 변화량
        z_center_drop = z_max - z_min

        # 처음 위치에서 최저점까지 떨어진 정도
        z_center_first_to_min_drop = z_first - z_min

        # 가장 높은 위치에서 마지막 위치까지 떨어진 정도
        z_center_peak_to_last_drop = z_max - z_last

        # 마지막 - 처음
        z_center_last_minus_first = z_last - z_first

        features["z_center_drop"] = float(z_center_drop)
        features["z_center_first_to_min_drop"] = float(z_center_first_to_min_drop)
        features["z_center_peak_to_last_drop"] = float(z_center_peak_to_last_drop)
        features["z_center_last_minus_first"] = float(z_center_last_minus_first)

        # z 변화 속도
        z_diff = z_series.diff().fillna(0)
        _add_stats(features, "z_diff", z_diff)
        _add_stats(features, "abs_z_diff", z_diff.abs())

        # 후반부 움직임
        tail_movement_mean, tail_movement_max = _calc_tail_movement(center_move)
        features["tail_movement_mean"] = tail_movement_mean
        features["tail_movement_max"] = tail_movement_max

        # 후반부 높이 평균
        tail_n = max(2, int(len(z_series) * 0.2))
        tail_z = z_series.tail(tail_n)
        features["tail_z_mean"] = float(tail_z.mean())
        features["tail_z_min"] = float(tail_z.min())
        features["tail_z_max"] = float(tail_z.max())

    else:
        _add_stats(features, "center_move", [0.0])
        _add_stats(features, "z_diff", [0.0])
        _add_stats(features, "abs_z_diff", [0.0])

        features["z_center_drop"] = 0.0
        features["z_center_first_to_min_drop"] = 0.0
        features["z_center_peak_to_last_drop"] = 0.0
        features["z_center_last_minus_first"] = 0.0
        features["tail_movement_mean"] = 0.0
        features["tail_movement_max"] = 0.0
        features["tail_z_mean"] = 0.0
        features["tail_z_min"] = 0.0
        features["tail_z_max"] = 0.0

    # 백엔드/프론트에서 바로 쓰는 대표값
    features["speed_max"] = float(abs_v.max())
    features["height_drop"] = float(features.get("z_center_drop", 0.0))
    features["movement_after"] = float(features.get("tail_movement_mean", 0.0))

    # 후처리용 낙상 강도 점수
    speed_max = features["speed_max"]
    height_drop = features["height_drop"]
    movement_after = features["movement_after"]

    rule_score = 0.0

    if height_drop >= 0.5:
        rule_score += 0.3
    if height_drop >= 0.8:
        rule_score += 0.3
    if speed_max >= 0.8:
        rule_score += 0.2
    if speed_max >= 1.5:
        rule_score += 0.2

    # 낙상 후 움직임이 적으면 더 위험하게 판단
    if movement_after <= 0.25 and height_drop >= 0.5:
        rule_score += 0.1

    features["rule_fall_score"] = float(min(rule_score, 1.0))

    # 규칙 기반 낙상 후보 여부
    features["rule_fall_candidate"] = float(
        1.0 if height_drop >= 0.8 and speed_max >= 0.8 else 0.0
    )

    return features