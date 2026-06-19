from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# =========================================================
# Not / Exception 데이터
# 낙상처럼 보이지만 실제로는 낙상이 아닌 행동
# 저장 위치: GatheredData/Not
# =========================================================
ACTIONS = {
    "sit_fast": "갑자기 앉기",
    "chair_sit": "의자에 앉기",
    "bed_lie": "침대에 눕기",
    "pick_object": "물건 줍기",
    "floor_sit_slow": "천천히 바닥에 앉기",

    "stand_up_fast": "갑자기 일어나기",
    "sit_to_stand": "앉았다가 일어나기",
    "lie_to_sit": "누워 있다가 앉기",
    "sit_to_lie": "앉은 상태에서 눕기",
    "roll_on_bed": "침대에서 뒤척이기",
    "turn_in_bed": "침대에서 몸 돌리기",
    "blanket_movement": "이불 움직임",
    "lean_on_table": "책상/식탁에 기대기",
    "reach_down": "아래 물건 집으려고 숙이기",
    "kneel_down": "무릎 꿇기",
    "clean_floor": "바닥 청소 동작",
    "exercise_squat": "운동 스쿼트",
    "stretching": "스트레칭",
    "fast_turn": "빠르게 방향 전환",
    "walk_fast_stop": "빠르게 걷다가 멈춤",

    "chair_moved": "의자 움직임",
    "door_open_close": "문 열고 닫힘",
    "curtain_movement": "커튼 움직임",
    "fan_noise": "선풍기/공기 흐름 잡음",
    "pet_movement": "반려동물 움직임",
    "object_drop": "물건 떨어짐",
    "blanket_falling": "이불이 떨어짐",
    "clothes_movement": "옷가지 움직임",

    "sensor_front_view": "센서 정면 설치 상황",
    "sensor_side_view": "센서 측면 설치 상황",
    "sensor_high_position": "센서 높은 위치 설치 상황",
    "sensor_low_position": "센서 낮은 위치 설치 상황",
    "near_sensor": "센서 가까운 위치",
    "far_sensor": "센서 먼 위치",
    "low_snr": "신호 약함",
    "high_noise": "노이즈 많음",
    "few_points": "감지 포인트 적음",
    "many_points": "감지 포인트 많음",
}


# =========================================================
# Fall 데이터
# 실제 낙상 행동
# 저장 위치: GatheredData/Fall
# =========================================================
FALL_ACTIONS = {
    "fall_forward": "앞으로 낙상",
    "fall_backward": "뒤로 낙상",
    "fall_left": "왼쪽으로 낙상",
    "fall_right": "오른쪽으로 낙상",
    "fall_from_chair": "의자에서 낙상",
    "fall_from_bed": "침대에서 낙상",
    "fall_after_walking": "걷다가 낙상",
    "fall_after_standing": "서 있다가 낙상",
    "fall_with_low_movement_after": "낙상 후 움직임 적음",
    "fall_with_small_speed": "속도는 작지만 높이 변화 큰 낙상",

    "fall_slip_forward": "미끄러져 앞으로 넘어짐",
    "fall_slip_backward": "미끄러져 뒤로 넘어짐",
    "fall_trip_forward": "발이 걸려 앞으로 넘어짐",
    "fall_side_from_standing": "서 있다가 옆으로 넘어짐",
    "fall_slow_collapse": "힘이 빠져 천천히 무너짐",
    "fall_faint_down": "실신처럼 아래로 무너짐",
    "fall_from_sofa": "소파에서 떨어짐",
    "fall_near_wall": "벽 근처에서 쓰러짐",
    "fall_with_turn": "몸을 돌리다가 낙상",
    "fall_with_partial_recovery": "버티려 했지만 결국 낙상",
}


ENVIRONMENT_ACTIONS = {
    "chair_moved",
    "door_open_close",
    "curtain_movement",
    "fan_noise",
    "pet_movement",
    "object_drop",
    "blanket_movement",
    "blanket_falling",
    "clothes_movement",
}

SENSOR_CONDITIONS = {
    "sensor_front_view",
    "sensor_side_view",
    "sensor_high_position",
    "sensor_low_position",
    "near_sensor",
    "far_sensor",
    "low_snr",
    "high_noise",
    "few_points",
    "many_points",
}


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def to_frame_array(value, frames: int, name: str, action: str):
    """
    x, y, z 중 하나가 숫자 1개로 만들어져도
    frames 길이 배열로 자동 변환합니다.
    """
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        return np.full(frames, float(arr))

    arr = arr.reshape(-1)

    if len(arr) == frames:
        return arr

    if len(arr) == 1:
        return np.full(frames, float(arr[0]))

    raise ValueError(
        f"{action} action에서 {name} 길이가 frames와 맞지 않습니다. "
        f"{name}_len={len(arr)}, frames={frames}"
    )


def is_environment_action(action: str) -> bool:
    return action in ENVIRONMENT_ACTIONS


def is_sensor_condition(action: str) -> bool:
    return action in SENSOR_CONDITIONS


# =========================================================
# Not / Exception 중심점 생성
# =========================================================
def make_center_curve(action: str, frames: int, rng: np.random.Generator):
    t = np.linspace(0, 1, frames)

    x0 = rng.normal(0.0, 0.10)
    y0 = rng.normal(1.4, 0.15)
    z0 = rng.normal(1.15, 0.05)

    if action == "near_sensor":
        y0 = rng.normal(0.7, 0.08)
    elif action == "far_sensor":
        y0 = rng.normal(2.5, 0.20)

    if action == "sensor_high_position":
        z0 = rng.normal(0.95, 0.05)
    elif action == "sensor_low_position":
        z0 = rng.normal(1.35, 0.05)

    if action == "sit_fast":
        drop = rng.normal(0.42, 0.04)
        z = z0 - drop * sigmoid((t - 0.32) * 18)
        x = x0 + rng.normal(0, 0.015, frames).cumsum() * 0.05
        y = y0 + 0.05 * smoothstep((t - 0.2) / 0.5)

    elif action == "chair_sit":
        drop = rng.normal(0.35, 0.04)
        z = z0 - drop * smoothstep((t - 0.15) / 0.65)
        x = x0 + 0.03 * np.sin(t * np.pi)
        y = y0 + 0.10 * smoothstep((t - 0.1) / 0.7)

    elif action == "bed_lie":
        drop = rng.normal(0.65, 0.05)
        z = z0 - drop * smoothstep((t - 0.1) / 0.8)
        x = x0 + 0.35 * smoothstep((t - 0.15) / 0.7)
        y = y0 + 0.15 * smoothstep((t - 0.2) / 0.6)

    elif action == "pick_object":
        dip = rng.normal(0.55, 0.05)
        bell = np.exp(-((t - 0.5) ** 2) / 0.035)
        z = z0 - dip * bell
        x = x0 + 0.05 * np.sin(2 * np.pi * t)
        y = y0 + 0.08 * np.sin(np.pi * t)

    elif action == "floor_sit_slow":
        drop = rng.normal(0.72, 0.05)
        z = z0 - drop * smoothstep(t)
        x = x0 + 0.08 * smoothstep((t - 0.2) / 0.7)
        y = y0 + 0.05 * smoothstep((t - 0.2) / 0.7)

    elif action == "stand_up_fast":
        rise = rng.normal(0.45, 0.04)
        z = z0 - 0.35 + rise * sigmoid((t - 0.35) * 18)
        x = x0 + 0.03 * np.sin(np.pi * t)
        y = y0 + 0.04 * smoothstep((t - 0.2) / 0.6)

    elif action == "sit_to_stand":
        rise = rng.normal(0.40, 0.04)
        z = z0 - 0.32 + rise * smoothstep((t - 0.15) / 0.75)
        x = x0 + 0.04 * np.sin(np.pi * t)
        y = y0 + 0.06 * smoothstep((t - 0.1) / 0.8)

    elif action == "lie_to_sit":
        rise = rng.normal(0.55, 0.05)
        z = z0 - 0.55 + rise * smoothstep((t - 0.15) / 0.75)
        x = x0 + 0.25 * smoothstep((t - 0.1) / 0.7)
        y = y0 + 0.12 * smoothstep((t - 0.15) / 0.7)

    elif action == "sit_to_lie":
        drop = rng.normal(0.55, 0.05)
        z = z0 - drop * smoothstep((t - 0.15) / 0.75)
        x = x0 + 0.28 * smoothstep((t - 0.1) / 0.7)
        y = y0 + 0.14 * smoothstep((t - 0.15) / 0.7)

    elif action == "roll_on_bed":
        z = np.full(frames, rng.normal(0.45, 0.04))
        x = x0 + 0.25 * np.sin(2 * np.pi * t)
        y = y0 + 0.08 * np.sin(4 * np.pi * t)

    elif action == "turn_in_bed":
        z = np.full(frames, rng.normal(0.48, 0.04))
        x = x0 + 0.18 * np.sin(2 * np.pi * t)
        y = y0 + 0.18 * np.cos(2 * np.pi * t)

    elif action == "blanket_movement":
        z = np.full(frames, rng.normal(0.45, 0.05)) + 0.05 * np.sin(4 * np.pi * t)
        x = x0 + 0.35 * np.sin(2 * np.pi * t)
        y = y0 + 0.15 * np.cos(2 * np.pi * t)

    elif action == "lean_on_table":
        drop = rng.normal(0.25, 0.03)
        z = z0 - drop * smoothstep((t - 0.2) / 0.55)
        x = x0 + 0.08 * smoothstep((t - 0.2) / 0.6)
        y = y0 + 0.18 * smoothstep((t - 0.2) / 0.6)

    elif action == "reach_down":
        dip = rng.normal(0.45, 0.05)
        bell = np.exp(-((t - 0.48) ** 2) / 0.04)
        z = z0 - dip * bell
        x = x0 + 0.08 * np.sin(np.pi * t)
        y = y0 + 0.18 * np.sin(np.pi * t)

    elif action == "kneel_down":
        drop = rng.normal(0.55, 0.05)
        z = z0 - drop * smoothstep((t - 0.15) / 0.65)
        x = x0 + 0.05 * np.sin(np.pi * t)
        y = y0 + 0.08 * smoothstep((t - 0.2) / 0.6)

    elif action == "clean_floor":
        z = np.full(frames, z0 - rng.normal(0.45, 0.04))
        x = x0 + 0.35 * np.sin(3 * np.pi * t)
        y = y0 + 0.25 * np.sin(2 * np.pi * t)

    elif action == "exercise_squat":
        dip = rng.normal(0.55, 0.05)
        z = z0 - dip * (np.sin(np.pi * t) ** 2)
        x = x0 + 0.04 * np.sin(2 * np.pi * t)
        y = y0 + 0.05 * np.sin(2 * np.pi * t)

    elif action == "stretching":
        z = z0 + 0.10 * np.sin(2 * np.pi * t)
        x = x0 + 0.20 * np.sin(np.pi * t)
        y = y0 + 0.08 * np.cos(np.pi * t)

    elif action == "fast_turn":
        z = z0 + 0.03 * np.sin(2 * np.pi * t)
        x = x0 + 0.18 * np.sin(2 * np.pi * t)
        y = y0 + 0.18 * np.cos(2 * np.pi * t)

    elif action == "walk_fast_stop":
        z = z0 + 0.03 * np.sin(8 * np.pi * t)
        x = x0 + 0.05 * np.sin(4 * np.pi * t)
        y = y0 + 0.75 * smoothstep(t)
        y[int(frames * 0.7):] = y[int(frames * 0.7)]

    elif action == "chair_moved":
        z = np.full(frames, rng.normal(0.55, 0.05))
        x = x0 + 0.50 * smoothstep((t - 0.2) / 0.55)
        y = y0 + 0.15 * np.sin(np.pi * t)

    elif action == "door_open_close":
        z = np.full(frames, rng.normal(1.0, 0.05))
        angle = smoothstep(t) * np.pi / 2
        x = x0 + 0.55 * np.sin(angle)
        y = y0 + 0.25 * np.cos(angle)

    elif action == "curtain_movement":
        z = z0 + 0.18 * np.sin(6 * np.pi * t)
        x = x0 + 0.35 * np.sin(4 * np.pi * t)
        y = y0 + 0.08 * np.sin(3 * np.pi * t)

    elif action == "fan_noise":
        z = z0 + rng.normal(0, 0.05, frames)
        x = x0 + rng.normal(0, 0.05, frames)
        y = y0 + rng.normal(0, 0.05, frames)

    elif action == "pet_movement":
        z = np.full(frames, rng.normal(0.35, 0.05))
        x = x0 + 0.50 * np.sin(2 * np.pi * t)
        y = y0 + 0.65 * smoothstep(t)

    elif action == "object_drop":
        z_start = rng.normal(0.90, 0.06)
        z_end = rng.normal(0.15, 0.03)
        p = sigmoid((t - 0.45) * 20)
        z = z_start - (z_start - z_end) * p
        x = x0 + 0.05 * np.sin(np.pi * t)
        y = y0 + 0.05 * np.sin(np.pi * t)

    elif action == "blanket_falling":
        z_start = rng.normal(0.75, 0.05)
        z_end = rng.normal(0.35, 0.05)
        p = smoothstep((t - 0.2) / 0.6)
        z = z_start - (z_start - z_end) * p
        x = x0 + 0.35 * np.sin(np.pi * t)
        y = y0 + 0.15 * np.sin(2 * np.pi * t)

    elif action == "clothes_movement":
        z = z0 + 0.20 * np.sin(3 * np.pi * t)
        x = x0 + 0.25 * np.sin(5 * np.pi * t)
        y = y0 + 0.10 * np.cos(2 * np.pi * t)

    elif is_sensor_condition(action):
        z = z0 + 0.04 * np.sin(2 * np.pi * t)
        x = x0 + 0.10 * np.sin(np.pi * t)
        y = y0 + 0.15 * smoothstep(t)

    else:
        raise ValueError(f"알 수 없는 action입니다: {action}")

    x = to_frame_array(x, frames, "x", action)
    y = to_frame_array(y, frames, "y", action)
    z = to_frame_array(z, frames, "z", action)

    center = np.column_stack([x, y, z])

    diff = np.diff(center, axis=0, prepend=center[:1])
    speed = np.linalg.norm(diff, axis=1) * frames / 30.0

    if action in ["fast_turn", "walk_fast_stop", "object_drop"]:
        speed = np.clip(speed, 0, 1.35)
    else:
        speed = np.clip(speed, 0, 1.10)

    return center, speed


# =========================================================
# Fall 중심점 생성
# =========================================================
def make_fall_center_curve(fall_action: str, frames: int, rng: np.random.Generator):
    x0 = rng.normal(0.0, 0.12)
    y0 = rng.normal(1.5, 0.18)

    if fall_action == "fall_from_bed":
        z_start = rng.normal(0.85, 0.06)
        z_end = rng.normal(0.32, 0.04)
    elif fall_action == "fall_from_chair":
        z_start = rng.normal(1.05, 0.07)
        z_end = rng.normal(0.28, 0.05)
    elif fall_action == "fall_from_sofa":
        z_start = rng.normal(0.95, 0.07)
        z_end = rng.normal(0.25, 0.05)
    else:
        z_start = rng.normal(1.45, 0.08)
        z_end = rng.normal(0.22, 0.05)

    if fall_action in ["fall_slow_collapse", "fall_faint_down", "fall_with_small_speed"]:
        fall_start_ratio = rng.uniform(0.25, 0.40)
        fall_duration_ratio = rng.uniform(0.25, 0.38)
    else:
        fall_start_ratio = rng.uniform(0.30, 0.48)
        fall_duration_ratio = rng.uniform(0.10, 0.20)

    fall_start = int(frames * fall_start_ratio)
    fall_end = min(frames - 1, fall_start + int(frames * fall_duration_ratio))

    dx_total = 0.0
    dy_total = 0.0
    turn_offset = np.zeros(frames)

    if fall_action in ["fall_forward", "fall_slip_forward", "fall_trip_forward"]:
        dy_total = rng.uniform(0.75, 1.25)

    elif fall_action in ["fall_backward", "fall_slip_backward"]:
        dy_total = -rng.uniform(0.75, 1.25)

    elif fall_action in ["fall_left", "fall_side_from_standing"]:
        dx_total = -rng.uniform(0.75, 1.25)

    elif fall_action == "fall_right":
        dx_total = rng.uniform(0.75, 1.25)

    elif fall_action == "fall_from_chair":
        dx_total = rng.uniform(-0.35, 0.35)
        dy_total = rng.uniform(0.35, 0.75)

    elif fall_action == "fall_from_bed":
        dx_total = rng.uniform(-0.45, 0.45)
        dy_total = rng.uniform(0.25, 0.60)

    elif fall_action == "fall_after_walking":
        dy_total = rng.uniform(0.65, 1.10)

    elif fall_action == "fall_after_standing":
        dx_total = rng.uniform(-0.45, 0.45)
        dy_total = rng.uniform(0.45, 0.85)

    elif fall_action == "fall_with_low_movement_after":
        dx_total = rng.uniform(-0.35, 0.35)
        dy_total = rng.uniform(0.55, 0.95)

    elif fall_action == "fall_with_small_speed":
        dx_total = rng.uniform(-0.25, 0.25)
        dy_total = rng.uniform(0.35, 0.65)

    elif fall_action == "fall_slow_collapse":
        dx_total = rng.uniform(-0.20, 0.20)
        dy_total = rng.uniform(0.10, 0.35)

    elif fall_action == "fall_faint_down":
        dx_total = rng.uniform(-0.15, 0.15)
        dy_total = rng.uniform(-0.15, 0.15)

    elif fall_action == "fall_from_sofa":
        dx_total = rng.uniform(-0.45, 0.45)
        dy_total = rng.uniform(0.30, 0.70)

    elif fall_action == "fall_near_wall":
        dx_total = rng.uniform(-0.35, 0.35)
        dy_total = rng.uniform(0.35, 0.75)
        y0 = rng.normal(0.8, 0.08)

    elif fall_action == "fall_with_turn":
        dx_total = rng.uniform(-0.60, 0.60)
        dy_total = rng.uniform(0.55, 0.95)
        turn_offset = 0.25 * np.sin(np.linspace(0, np.pi, frames))

    elif fall_action == "fall_with_partial_recovery":
        dx_total = rng.uniform(-0.45, 0.45)
        dy_total = rng.uniform(0.55, 0.95)

    else:
        raise ValueError(f"알 수 없는 fall_action입니다: {fall_action}")

    center = []

    for frame_idx in range(frames):
        if frame_idx < fall_start:
            progress = 0.0
        elif frame_idx > fall_end:
            progress = 1.0
        else:
            progress = (frame_idx - fall_start) / max(1, fall_end - fall_start)

        if fall_action in ["fall_with_small_speed", "fall_slow_collapse", "fall_faint_down"]:
            p = smoothstep(progress)
        else:
            p = sigmoid((progress - 0.5) * 12)

        walk_offset = 0.0
        if fall_action == "fall_after_walking" and frame_idx < fall_start:
            walk_offset = 0.012 * frame_idx

        x = x0 + dx_total * p + turn_offset[frame_idx]
        y = y0 + dy_total * p + walk_offset
        z = z_start - (z_start - z_end) * p

        if fall_action in ["fall_slip_forward", "fall_slip_backward"] and frame_idx > fall_end:
            y += rng.normal(0, 0.06)

        if frame_idx > fall_end:
            if fall_action == "fall_with_low_movement_after":
                after_noise = 0.004
            elif fall_action == "fall_with_partial_recovery":
                after_noise = 0.08
            elif fall_action in ["fall_from_bed", "fall_from_sofa"]:
                after_noise = 0.012
            else:
                after_noise = 0.025

            x += rng.normal(0, after_noise)
            y += rng.normal(0, after_noise)
            z += rng.normal(0, after_noise)

        center.append([x, y, z])

    center = np.array(center)

    diff = np.diff(center, axis=0, prepend=center[:1])
    speed = np.linalg.norm(diff, axis=1) * frames / 20.0

    if fall_action in ["fall_with_small_speed", "fall_slow_collapse"]:
        speed = np.clip(speed, 0, 0.85)
    elif fall_action == "fall_faint_down":
        speed = np.clip(speed, 0, 1.10)
    else:
        speed = np.clip(speed, 0, 2.35)

    return center, speed, fall_start, fall_end


def get_exception_point_count(action: str, rng: np.random.Generator) -> int:
    if action == "few_points":
        return int(rng.integers(3, 8))
    if action == "many_points":
        return int(rng.integers(36, 64))
    if is_environment_action(action):
        return int(rng.integers(5, 18))
    return int(rng.integers(12, 36))


def get_exception_snr_noise(action: str, n_points: int, rng: np.random.Generator):
    if action == "low_snr":
        snrs = rng.normal(5.5, 1.0, n_points)
        noises = rng.normal(2.8, 0.7, n_points)
    elif action == "high_noise":
        snrs = rng.normal(10.0, 2.5, n_points)
        noises = rng.normal(5.0, 1.0, n_points)
    elif action == "far_sensor":
        snrs = rng.normal(8.0, 2.0, n_points)
        noises = rng.normal(3.0, 0.8, n_points)
    elif action == "near_sensor":
        snrs = rng.normal(18.0, 3.0, n_points)
        noises = rng.normal(2.0, 0.5, n_points)
    elif is_environment_action(action):
        snrs = rng.normal(10.0, 3.0, n_points)
        noises = rng.normal(3.2, 0.8, n_points)
    else:
        snrs = rng.normal(13.0, 3.0, n_points)
        noises = rng.normal(2.5, 0.6, n_points)

    return snrs, noises


def generate_one_csv(action: str, save_path: Path, rng: np.random.Generator):
    frames = int(rng.integers(70, 120))
    center, speed = make_center_curve(action, frames, rng)

    rows = []

    for frame_idx in range(frames):
        n_points = get_exception_point_count(action, rng)

        cx, cy, cz = center[frame_idx]

        if action in ["bed_lie", "sit_to_lie", "lie_to_sit", "roll_on_bed", "turn_in_bed"]:
            spread_x = 0.22 + 0.10 * frame_idx / frames
            spread_y = 0.16 + 0.05 * frame_idx / frames
            spread_z = 0.10
        elif action in ["floor_sit_slow", "kneel_down", "exercise_squat", "clean_floor"]:
            spread_x = 0.16
            spread_y = 0.13
            spread_z = 0.12
        elif action in ["pick_object", "reach_down", "lean_on_table"]:
            spread_x = 0.13
            spread_y = 0.12
            spread_z = 0.18
        elif is_environment_action(action):
            spread_x = 0.08
            spread_y = 0.08
            spread_z = 0.08
        else:
            spread_x = 0.12
            spread_y = 0.11
            spread_z = 0.14

        xs = rng.normal(cx, spread_x, n_points)
        ys = rng.normal(cy, spread_y, n_points)
        zs = rng.normal(cz, spread_z, n_points)

        signs = rng.choice([-1, 1], size=n_points)
        vs = signs * rng.normal(speed[frame_idx], 0.08, n_points)

        if action in ["fast_turn", "walk_fast_stop"]:
            vs = np.clip(vs, -1.35, 1.35)
        else:
            vs = np.clip(vs, -1.15, 1.15)

        snrs, noises = get_exception_snr_noise(action, n_points, rng)

        for i in range(n_points):
            rows.append({
                "frame": frame_idx,
                "DetObj#": n_points,
                "x": round(float(xs[i]), 5),
                "y": round(float(ys[i]), 5),
                "z": round(float(zs[i]), 5),
                "v": round(float(vs[i]), 5),
                "snr": round(float(snrs[i]), 5),
                "noise": round(float(noises[i]), 5),
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def generate_one_fall_csv(fall_action: str, save_path: Path, rng: np.random.Generator):
    frames = int(rng.integers(75, 130))
    center, speed, fall_start, fall_end = make_fall_center_curve(fall_action, frames, rng)

    rows = []

    for frame_idx in range(frames):
        n_points = int(rng.integers(14, 38))

        cx, cy, cz = center[frame_idx]

        if frame_idx < fall_start:
            spread_x = 0.12
            spread_y = 0.12
            spread_z = 0.16
        elif fall_start <= frame_idx <= fall_end:
            spread_x = 0.18
            spread_y = 0.18
            spread_z = 0.20
        else:
            spread_x = 0.28
            spread_y = 0.22
            spread_z = 0.08

        if fall_action in ["fall_from_bed", "fall_from_sofa"]:
            spread_x += 0.08
            spread_y += 0.04

        if fall_action == "fall_from_chair":
            spread_x += 0.04
            spread_y += 0.04

        xs = rng.normal(cx, spread_x, n_points)
        ys = rng.normal(cy, spread_y, n_points)
        zs = rng.normal(cz, spread_z, n_points)

        signs = rng.choice([-1, 1], size=n_points)

        if fall_start <= frame_idx <= fall_end:
            if fall_action in ["fall_with_small_speed", "fall_slow_collapse"]:
                v_base = speed[frame_idx] + rng.normal(0.05, 0.06, n_points)
            else:
                v_base = speed[frame_idx] + rng.normal(0.35, 0.18, n_points)
        else:
            v_base = speed[frame_idx] + rng.normal(0.04, 0.06, n_points)

        vs = signs * v_base

        if fall_action in ["fall_with_small_speed", "fall_slow_collapse"]:
            vs = np.clip(vs, -0.9, 0.9)
        else:
            vs = np.clip(vs, -2.4, 2.4)

        snrs = rng.normal(14.0, 3.5, n_points)
        noises = rng.normal(2.4, 0.65, n_points)

        for i in range(n_points):
            rows.append({
                "frame": frame_idx,
                "DetObj#": n_points,
                "x": round(float(xs[i]), 5),
                "y": round(float(ys[i]), 5),
                "z": round(float(zs[i]), 5),
                "v": round(float(vs[i]), 5),
                "snr": round(float(snrs[i]), 5),
                "noise": round(float(noises[i]), 5),
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description="mmWave 예외행동 + 낙상 보강 CSV 생성기"
    )

    parser.add_argument(
        "--root",
        type=str,
        default=r"C:\smart_care_ai",
        help="프로젝트 루트 경로. 기본값: C:\\smart_care_ai",
    )

    parser.add_argument(
        "--samples-per-action",
        type=int,
        default=10,
        help="예외행동별 생성할 CSV 개수. 기본값: 10",
    )

    parser.add_argument(
        "--fall-samples-per-action",
        type=int,
        default=10,
        help="낙상 유형별 생성할 CSV 개수. 기본값: 10",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )

    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "exception", "fall"],
        help="생성 범위 선택: all, exception, fall",
    )

    args = parser.parse_args()
    root = Path(args.root)

    not_dir = root / "data" / "mmwave_fall" / "GatheredData" / "Not"
    fall_dir = root / "data" / "mmwave_fall" / "GatheredData" / "Fall"

    not_dir.mkdir(parents=True, exist_ok=True)
    fall_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    total_exception = 0
    total_fall = 0

    print("CSV 생성 시작")
    print(f"프로젝트 루트: {root}")
    print(f"Not 저장 위치: {not_dir}")
    print(f"Fall 저장 위치: {fall_dir}")
    print()

    if args.only in ["all", "exception"]:
        print("예외행동 / Not CSV 생성 시작")

        for action, kor_name in ACTIONS.items():
            for idx in range(1, args.samples_per_action + 1):
                file_name = f"exception_{action}_{idx:03d}.csv"
                save_path = not_dir / file_name
                generate_one_csv(action, save_path, rng)
                total_exception += 1

            print(f"- {kor_name}({action}) {args.samples_per_action}개 생성 완료")

        print()

    if args.only in ["all", "fall"]:
        print("Fall CSV 생성 시작")

        for fall_action, kor_name in FALL_ACTIONS.items():
            for idx in range(1, args.fall_samples_per_action + 1):
                file_name = f"{fall_action}_extra_{idx:03d}.csv"
                save_path = fall_dir / file_name
                generate_one_fall_csv(fall_action, save_path, rng)
                total_fall += 1

            print(f"- {kor_name}({fall_action}) {args.fall_samples_per_action}개 생성 완료")

        print()

    print("CSV 생성 완료")
    print(f"예외행동 Not CSV 생성 개수: {total_exception}")
    print(f"Fall CSV 생성 개수: {total_fall}")
    print()
    print("이제 아래 명령으로 모델을 다시 학습하세요.")
    print(r"cd C:\smart_care_ai\rf")
    print(r"..\backend\venv\Scripts\python.exe train_model.py")
    print()
    print("평가 명령:")
    print(r"..\backend\venv\Scripts\python.exe evaluate_model.py")


if __name__ == "__main__":
    main()
