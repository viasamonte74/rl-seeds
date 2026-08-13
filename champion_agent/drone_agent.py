from __future__ import annotations
import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from scipy import ndimage
from pathlib import Path
import onnxruntime as ort
from typing import Callable, Optional
CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ
MAX_SPEED = 6.0
ACQUISITION_VERTICAL_MAX_ACCEL = 30.0
ACQUISITION_VERTICAL_BOOST_S = 0.5
TARGET_SEARCH_AGL = 14.0
MAX_SEARCH_VZ = 3.0
MAX_INTERCEPT_VZ = 5.0
AGL_SMOOTH_ALPHA = 0.2
CLUE_JUMP_THRESHOLD_M = 1.0
CLUE_VELOCITY_EMA = 0.18
CLUE_ANCHOR_COUNT = 8
CLUE_PREDICTION_HORIZON_S = 0.1
LOCAL_SEARCH_ENTER_M = 20.0
R_EFFECTIVE = LOCAL_SEARCH_ENTER_M
USABLE_CAMERA_HALF_FOV_DEG = 35.0
DEPTH_TARGET_FOV_DEG = 90.28
DEPTH_TARGET_POOL = 2
DEPTH_TARGET_THRESHOLD = 0.9
DEPTH_TARGET_CLUE_GATE_M = 20.0
DEPTH_TARGET_MIN_PIXELS = 2
DEPTH_TARGET_MAX_COMPONENT_FRAC = 0.2
DEPTH_TARGET_DISTANT_COMPONENT_MIN_RANGE_M = 5.0
DEPTH_TARGET_DISTANT_MAX_COMPONENT_FRAC = 0.01
DEPTH_TARGET_CONTRAST_WINDOW = 9
DEPTH_TARGET_CONTRAST_MIN_M = 1.0
DEPTH_TARGET_NORMAL_MERGED_MIN_FRAC = 0.01
DEPTH_TARGET_NORMAL_MERGED_MIN_WIDTH_FRAC = 0.75
DEPTH_TARGET_NORMAL_MERGED_MIN_DEPTH_SPAN_M = 5.0
DEPTH_TARGET_NEAR_MERGED_MAX_RANGE_M = 3.0
DEPTH_TARGET_TRACK_ROI_MAX_RANGE_M = 35.0
DEPTH_TARGET_TRACK_ROI_UNCONDITIONAL_RANGE_M = 20.0
DEPTH_TARGET_TRACK_ROI_EXTENDED_MAX_TARGET_Z_M = 10.0
DEPTH_TARGET_TRACK_ROI_EXTENDED_MIN_CLUE_ERROR_M = 7.5
DEPTH_TARGET_TRACK_ROI_MAX_MEMORY_S = 1.0
DEPTH_TARGET_TRACK_ROI_POSITION_GATE_M = 0.9
DEPTH_TARGET_TRACK_ROI_DEPTH_TOLERANCE_M = 0.65
DEPTH_TARGET_TRACK_ROI_MIN_HALF_PX = 16.0
DEPTH_TARGET_TRACK_ROI_MAX_HALF_PX = 256.0
DEPTH_TARGET_ABSOLUTE_NEAR_MAX_RANGE_M = 3.5
DEPTH_TARGET_ABSOLUTE_NEAR_Z_GATE_M = 1.0
TARGET_BODY_RADIUS_M = 0.18
TARGET_POSITION_FILTER_ALPHA = 0.72
TARGET_CONFIRM_HITS = 3
TARGET_CONFIRM_WINDOW = 5
TARGET_Z_EMERGENCY_DROP_M = 10.0
TARGET_Z_GROUND_FLOOR_M = 2.0
TAKEOFF_CLEARANCE_AGL_M = 1.5
TAKEOFF_CLEARANCE_CLIMB_MPS = 2.0
INITIAL_DIRECT_CHASE_S = 0.5
INITIAL_DIRECT_CHASE_MIN_RANGE_M = 48.0
TRACK_DROPOUT_HOLD_S = 0.04
TRACK_ASSOCIATION_GATE_M = 2.0
TRACK_NEAR_ASSOCIATION_GATE_M = 3.0
TRACK_NEAR_ASSOCIATION_MEMORY_S = 0.5
TRACK_STALL_MIN_RANGE_M = 40.0
TRACK_STALL_WINDOW_S = 0.75
TRACK_STALL_MIN_PROGRESS_M = 0.5
TRACK_STALL_RECOVERY_S = 0.5
REACQUIRE_TIMEOUT_S = 1.0
REACQUIRE_FAR_TIMEOUT_S = 2.0
REACQUIRE_SPEED_MPS = 3.0
REACQUIRE_CAUTION_RANGE_M = 10.0
REACQUIRE_PREDICTION_CAP_S = 0.5
REACQUIRE_FAR_PREDICTION_CAP_S = 2.0
REACQUIRE_INITIAL_YAW_BIAS_DEG = 20.0
REACQUIRE_YAW_SWEEP_RATE_DEG_S = 90.0
REACQUIRE_MAX_YAW_BIAS_DEG = 65.0
VELOCITY_HISTORY_FRAMES = 35
TARGET_FIT_RECENCY_S = 0.35
LEGACY_MOTION_FIT_FRAMES = 12
WEIGHTED_FIT_UPWARD_GAP_MIN_M = 8.0
WEIGHTED_FIT_UPWARD_GAP_MAX_M = 10.0
TARGET_VELOCITY_CLAMP = 5.5
TARGET_ACCELERATION_CLAMP = 12.0
CHASE_SLOWDOWN_RANGE_M = 2.0
CHASE_MIN_SPEED_MPS = 5.0
PIXEL_PREDICTION_ALPHA = 1.0
PIXEL_PREDICTION_NEAR_ALPHA = 0.5
PIXEL_PREDICTION_NEAR_RANGE_M = 3.0
PIXEL_PREDICTION_MAX_DELTA_PX = float('inf')
HIGH_CLOSING_PIXEL_ALPHA = 2.0
HIGH_CLOSING_PIXEL_ACQUIRE_MIN_RANGE_M = 40.0
HIGH_CLOSING_PIXEL_ACQUIRE_MAX_RANGE_M = 55.0
HIGH_CLOSING_PIXEL_ACQUIRE_CENTER_HALF_FRAC = 0.03
LEAD_PURE_PURSUIT_FAR_M = 10.0
LEAD_PURE_PURSUIT_NEAR_M = 3.0
LEAD_WEIGHT_NEAR = 0.2
TERMINAL_HOMING_RANGE_M = 3.0
TERMINAL_RECENTER_RANGE_M = 2.0
TERMINAL_RECENTER_EXIT_RANGE_M = 2.5
TERMINAL_RECENTER_ENTER_ANGLE_DEG = 11.4
TERMINAL_RECENTER_EXIT_ANGLE_DEG = 7.6
TERMINAL_RECENTER_PREDICTIVE_RANGE_M = 3.0
TERMINAL_RECENTER_PREDICTIVE_ANGLE_DEG = 5.0
TERMINAL_RECENTER_CLOSING_SPEED_MPS = 0.0
TERMINAL_RECENTER_TANGENT_SCALE = 1.0
TERMINAL_RECENTER_STOP_DECEL_MPS2 = 4.0
TERMINAL_RECENTER_MIN_ENTRY_CLOSING_MPS = 1.0
TERMINAL_RECENTER_TIMEOUT_S = 0.75
PARALLEL_CHASE_ALIGNMENT_DEG = 20.0
PARALLEL_CHASE_MIN_DRONE_SPEED_MPS = 2.0
PARALLEL_CHASE_MIN_TARGET_SPEED_MPS = 2.0
TERMINAL_RECENTER_LOST_HOLD_S = 0.15
TERMINAL_RECENTER_STOP_LOST_HOLD_S = 0.15
TERMINAL_RECENTER_CENTER_FRAMES = 5
TERMINAL_RECENTER_CPA_HORIZON_S = 0.5
TERMINAL_CAPTURE_CORRIDOR_M = 0.12
TERMINAL_RECENTER_MISS_DISTANCE_M = 0.2
TERMINAL_RECENTER_EXIT_MISS_M = 0.1
TERMINAL_RECENTER_LEAD_HORIZON_S = 0.12
TERMINAL_RECENTER_LEAD_MAX_M = 0.3
TERMINAL_RECENTER_LEAD_RESPONSE_S = 0.5
TERMINAL_RECENTER_ERROR_GROWTH_DEG = 0.1
TERMINAL_RECENTER_LOST_YAW_OFFSET_DEG = 90.0
TERMINAL_ACTUATOR_LAG_S = 0.25
TERMINAL_ACCEL_ROLLOFF_MPS2 = 6.0
TERMINAL_HIGH_ACCEL_THRESHOLD_MPS2 = 4.0
TERMINAL_HIGH_ACCEL_SCALE = 0.5
TERMINAL_DYNAMICS_MIN_LOS_RATE_RAD_S = 0.2
TERMINAL_DYNAMICS_MID_CLOSING_MIN_MPS = 3.0
TERMINAL_DYNAMICS_MID_CLOSING_MAX_MPS = 5.2
VISIBLE_CHASE_LEAD_HORIZON_S = 0.25
RULE_AWARE_LEAD_ENTER_M = 8.0
PROPORTIONAL_NAVIGATION_GAIN = 3.0
PROPORTIONAL_NAVIGATION_ACCEL_MPS2 = 15.0
PROPORTIONAL_NAVIGATION_RESPONSE_S = 0.42
PROPORTIONAL_NAVIGATION_MAX_DIRECT_CLOSING_MPS = 6.2
PROPORTIONAL_NAVIGATION_MIN_LOS_RATE_RAD_S = 0.1
PROPORTIONAL_NAVIGATION_STALL_RANGE_M = 3.0
PROPORTIONAL_NAVIGATION_STALL_WINDOW_S = 0.5
PROPORTIONAL_NAVIGATION_STALL_MIN_PROGRESS_M = 0.25
TERMINAL_STRONG_PN_RESPONSE_S = 0.6
TERMINAL_ACCEL_LEAD_HORIZON_S = 0.25
YAW_PREDICTION_TIME = 0.1
HEAD_ON_ANGLE_DEG = 30.0
HEAD_ON_RANGE_M = 15.0
STEEP_VERTICAL_ANGLE_DEG = 45.0
TILT_RUNAWAY_MIN_DEG = 30.0
TILT_HARD_CAP_DEG = 65.0
TILT_CLIMB_WINDOW_S = 0.3
TILT_CLIMB_THRESHOLD_DEG = 15.0
TILT_BRAKE_MIN_FRACTION = 0.0
NORMAL_MAX_ACCEL = 9.0
CLOSE_MAX_ACCEL = 12.0
CHASE_CLOSE_MAX_ACCEL = 12.0
CLOSE_RANGE_M = 5.0
FINAL_CHASE_MAX_ACCEL = 30.0
FINAL_CHASE_RANGE_M = 3.0
LARGE_ACQUISITION_VERTICAL_GAP_M = 5.0
SHORT_LEAD_ACQUISITION_RANGE_M = 15.0
TARGET_Z_MEMORY_FRAMES = 15
TARGET_Z_MEMORY_BAND_M = 2.0
TARGET_Z_TENTATIVE_WINDOW_S = 2.0
TARGET_Z_TENTATIVE_HITS = 3
TARGET_Z_TENTATIVE_SPREAD_M = 0.75
TARGET_Z_TENTATIVE_MIN_M = 2.5
TARGET_Z_TENTATIVE_MAX_M = 25.5
TARGET_Z_REANCHOR_RANGE_M = 8.0
TARGET_Z_REANCHOR_HITS = 3
TARGET_Z_REANCHOR_SPREAD_M = 0.75
TERMINAL_Z_LOCK_RANGE_M = 5.0
AGL_KZ = 0.8
AGL_DEADBAND_M = 0.5
NEAR_CLUE_SINGULARITY_M = 2.0
MAX_RAY_DISTANCE_M = 20.0
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 100.0

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-09:
        return np.zeros_like(v)
    return v / n

def tilt_brake_fraction(tilt_rad: float, start_deg: float, full_deg: float, min_fraction: float) -> float:
    tilt_deg = math.degrees(tilt_rad)
    span = full_deg - start_deg
    t = 0.0 if span <= 0 else (tilt_deg - start_deg) / span
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - min_fraction) * t

def limit_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)

def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]], dtype=np.float64)

def world_to_camera(relative_world: np.ndarray, roll: float, pitch: float, yaw: float):
    rot = rotation_matrix_from_rpy(roll, pitch, yaw)
    forward_axis = rot @ np.array([1.0, 0.0, 0.0])
    up_ref = rot @ np.array([0.0, 0.0, 1.0])
    forward_axis = unit(forward_axis)
    right_axis = unit(np.cross(forward_axis, up_ref))
    if np.linalg.norm(right_axis) < 1e-06:
        right_axis = unit(np.cross(forward_axis, np.array([0.0, 1.0, 0.0])))
    up_axis = np.cross(right_axis, forward_axis)
    rel = np.asarray(relative_world, dtype=np.float64)
    forward = float(np.dot(rel, forward_axis))
    right = float(np.dot(rel, right_axis))
    up = float(np.dot(rel, up_axis))
    return (forward, right, up)

def camera_angles(relative_world: np.ndarray, roll: float, pitch: float, yaw: float):
    forward, right, up = world_to_camera(relative_world, roll, pitch, yaw)
    horizontal_angle = math.atan2(right, forward) if forward > 0 else math.pi
    vertical_angle = math.atan2(up, math.hypot(forward, right)) if forward > 0 else 0.0
    return (forward, horizontal_angle, vertical_angle)

def slew_velocity(previous: np.ndarray, requested: np.ndarray, dt: float, max_accel: float) -> np.ndarray:
    delta = np.asarray(requested, dtype=np.float64) - np.asarray(previous, dtype=np.float64)
    delta_norm = float(np.linalg.norm(delta))
    max_delta = max_accel * dt
    if delta_norm > max_delta and delta_norm > 1e-12:
        delta = delta * (max_delta / delta_norm)
    return previous + delta

class ClueFilter:

    def __init__(self, window_frames: int | None=None):
        count = CLUE_ANCHOR_COUNT if window_frames is None else window_frames
        self._anchors: deque = deque(maxlen=max(1, int(count)))
        self.reset()

    def reset(self) -> None:
        self._anchors.clear()
        self.raw: np.ndarray | None = None
        self.filtered: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self._previous_raw: np.ndarray | None = None
        self._motion = np.zeros(2, dtype=np.float64)

    def update(self, own_xy: np.ndarray, clue_offset_xy: np.ndarray):
        measurement = np.asarray(own_xy, dtype=np.float64) + np.asarray(clue_offset_xy, dtype=np.float64)
        self.raw = measurement
        if self._previous_raw is None:
            self._anchors.append((measurement.copy(), self._motion.copy()))
        else:
            step = measurement - self._previous_raw
            if float(np.linalg.norm(step)) > CLUE_JUMP_THRESHOLD_M:
                self._anchors.append((measurement.copy(), self._motion.copy()))
            else:
                observed_velocity = step / DT
                self.velocity += CLUE_VELOCITY_EMA * (observed_velocity - self.velocity)
        self._previous_raw = measurement.copy()
        self._motion += self.velocity * DT
        corrected = [p + self._motion - d for p, d in self._anchors]
        self.filtered = np.mean(np.stack(corrected), axis=0)
        return (self.raw.copy(), self.filtered.copy())

    def predict(self, horizon_s: float) -> np.ndarray:
        if self.filtered is None:
            raise RuntimeError('clue estimate is not initialized')
        return self.filtered + self.velocity * max(0.0, float(horizon_s))

def local_agl(state: np.ndarray) -> float:
    return float(state[137]) * MAX_RAY_DISTANCE_M

def _vertical_rate(error: float, max_vz: float, kz: float, deadband: float) -> float:
    if abs(error) < deadband:
        return 0.0
    return float(np.clip(kz * error, -max_vz, max_vz))

def vertical_rate_to_target_z(current_z: float, target_z: float, max_vz: float=MAX_SEARCH_VZ, kz: float=AGL_KZ, deadband: float=AGL_DEADBAND_M) -> float:
    return _vertical_rate(target_z - current_z, max_vz, kz, deadband)

def speed_budget_velocity(horizontal_direction_xy: np.ndarray, vz: float, max_speed: float=MAX_SPEED) -> np.ndarray:
    horizontal_speed = math.sqrt(max(0.0, max_speed * max_speed - vz * vz))
    d = unit(horizontal_direction_xy)
    return np.array([horizontal_speed * d[0], horizontal_speed * d[1], vz], dtype=np.float64)

def search_sector(own_xy: np.ndarray, raw_clue_xy: np.ndarray, r_effective: float=R_EFFECTIVE, usable_half_fov_rad: float=0.0):
    raw_delta_xy = np.asarray(raw_clue_xy, dtype=np.float64) - np.asarray(own_xy, dtype=np.float64)
    d = float(np.linalg.norm(raw_delta_xy))
    raw_clue_bearing = math.atan2(raw_delta_xy[1], raw_delta_xy[0])
    if d > r_effective:
        beta = math.asin(min(1.0, r_effective / d))
        scan_half_width = max(0.0, beta - usable_half_fov_rad)
    else:
        beta = math.pi
        scan_half_width = math.pi
    return (d, raw_clue_bearing, beta, scan_half_width)

def local_search_direction(own_xy: np.ndarray, raw_clue_xy: np.ndarray, radial_weight: float=0.6, tangent_weight: float=0.4) -> np.ndarray:
    radial = unit(np.asarray(raw_clue_xy, dtype=np.float64) - np.asarray(own_xy, dtype=np.float64))
    tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
    return unit(radial_weight * radial + tangent_weight * tangent)

def _min_pool_depth(depth: np.ndarray, pool: int) -> np.ndarray:
    pool = int(pool)
    height = depth.shape[0] // pool
    width = depth.shape[1] // pool
    cropped = depth[:height * pool, :width * pool]
    if pool == 2:
        return np.minimum(np.minimum(cropped[0::2, 0::2], cropped[0::2, 1::2]), np.minimum(cropped[1::2, 0::2], cropped[1::2, 1::2]))
    return cropped.reshape(height, pool, width, pool).min(axis=(1, 3))

@dataclass
class Detection:
    position: np.ndarray
    range_m: float
    horizontal_angle: float
    vertical_angle: float
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    camera_range_m: Optional[float] = None
    component_pixels: Optional[int] = None
    component_bbox_width_px: Optional[float] = None
    component_bbox_height_px: Optional[float] = None
    component_depth_span_m: Optional[float] = None
    clue_error_m: Optional[float] = None
    source: str = 'component'

def extrapolate_detection_pixel(detection: Detection, previous_pixel: Optional[tuple[float, float]], drone_pos: np.ndarray, drone_rpy: np.ndarray, alpha_override: Optional[float]=None) -> Detection:
    if previous_pixel is None or detection.pixel_x is None or detection.pixel_y is None or (detection.image_width is None) or (detection.image_height is None):
        return detection
    width = int(detection.image_width)
    height = int(detection.image_height)
    if width <= 0 or height <= 0:
        return detection
    alpha = float(alpha_override) if alpha_override is not None else float(PIXEL_PREDICTION_NEAR_ALPHA) if float(detection.range_m) < PIXEL_PREDICTION_NEAR_RANGE_M else float(PIXEL_PREDICTION_ALPHA)
    pixel_delta = np.array([float(detection.pixel_x) - float(previous_pixel[0]), float(detection.pixel_y) - float(previous_pixel[1])], dtype=np.float64)
    delta_norm = float(np.linalg.norm(pixel_delta))
    if delta_norm > PIXEL_PREDICTION_MAX_DELTA_PX:
        pixel_delta *= PIXEL_PREDICTION_MAX_DELTA_PX / max(delta_norm, 1e-12)
    predicted_x = float(np.clip(float(detection.pixel_x) + alpha * float(pixel_delta[0]), 0.0, float(width - 1)))
    predicted_y = float(np.clip(float(detection.pixel_y) + alpha * float(pixel_delta[1]), 0.0, float(height - 1)))
    focal = width * 0.5 / math.tan(math.radians(DEPTH_TARGET_FOV_DEG) * 0.5)
    body_ray = np.array([1.0, (width * 0.5 - predicted_x) / focal, (height * 0.5 - predicted_y) / focal], dtype=np.float64)
    body_ray /= np.linalg.norm(body_ray) + 1e-12
    rotation = rotation_matrix_from_rpy(float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
    position = np.asarray(drone_pos, dtype=np.float64)
    camera_position = position + 0.13 * rotation[:, 0] + 0.05 * rotation[:, 2]
    ray_range = float(detection.camera_range_m) if detection.camera_range_m is not None else float(detection.range_m)
    world_position = camera_position + rotation @ body_ray * ray_range
    relative = world_position - position
    forward, horizontal, vertical = camera_angles(relative, float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
    if forward <= 0.0:
        return detection
    return Detection(position=world_position, range_m=float(np.linalg.norm(relative)), horizontal_angle=horizontal, vertical_angle=vertical, pixel_x=predicted_x, pixel_y=predicted_y, image_width=width, image_height=height, camera_range_m=ray_range, component_pixels=detection.component_pixels, component_bbox_width_px=detection.component_bbox_width_px, component_bbox_height_px=detection.component_bbox_height_px, component_depth_span_m=detection.component_depth_span_m, clue_error_m=detection.clue_error_m, source=detection.source)

class TargetDetector:

    def detect(self, *, drone_pos: np.ndarray, drone_rpy: np.ndarray, depth_image: np.ndarray, clue_xy_est: Optional[np.ndarray]=None, track_position_est: Optional[np.ndarray]=None, target_z_est: Optional[float]=None) -> Optional[Detection]:
        raise NotImplementedError

class HybridTargetDetector(TargetDetector):

    def detect(self, *, drone_pos: np.ndarray, drone_rpy: np.ndarray, depth_image: np.ndarray, clue_xy_est: Optional[np.ndarray]=None, track_position_est: Optional[np.ndarray]=None, target_z_est: Optional[float]=None) -> Optional[Detection]:
        return self._depth_detection(drone_pos, drone_rpy, depth_image, clue_xy_est, track_position_est, target_z_est)

    def _absolute_near_detection(self, drone_pos: np.ndarray, drone_rpy: np.ndarray, depth: np.ndarray, pooled: np.ndarray, clue_xy_est: Optional[np.ndarray], track_position_est: Optional[np.ndarray], target_z_est: Optional[float]) -> Optional[Detection]:
        if target_z_est is None or not math.isfinite(float(target_z_est)):
            return None
        surface_forward = pooled * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        shallow = (pooled > 0.0) & (pooled < DEPTH_TARGET_THRESHOLD) & (surface_forward <= DEPTH_TARGET_ABSOLUTE_NEAR_MAX_RANGE_M)
        labels, count = ndimage.label(shallow)
        if not count:
            return None
        rotation = rotation_matrix_from_rpy(float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
        position = np.asarray(drone_pos, dtype=np.float64)
        camera_position = position + 0.13 * rotation[:, 0] + 0.05 * rotation[:, 2]
        focal = depth.shape[1] * 0.5 / math.tan(math.radians(DEPTH_TARGET_FOV_DEG) * 0.5)
        pool = int(DEPTH_TARGET_POOL)
        track = None if track_position_est is None else np.asarray(track_position_est, dtype=np.float64)
        best: Optional[Detection] = None
        best_score = float('inf')
        component_slices = ndimage.find_objects(labels, max_label=int(count))
        for label_id, component_slice in enumerate(component_slices, start=1):
            if component_slice is None:
                continue
            local_rows, local_cols = np.where(labels[component_slice] == label_id)
            pixels = int(len(local_rows))
            if pixels < DEPTH_TARGET_MIN_PIXELS:
                continue
            rows = local_rows + int(component_slice[0].start)
            cols = local_cols + int(component_slice[1].start)
            if pixels > int(0.02 * pooled.size):
                continue
            pixel_u = (float(cols.mean()) + 0.5) * pool
            pixel_v = (float(rows.mean()) + 0.5) * pool
            body_ray = np.array([1.0, (depth.shape[1] * 0.5 - pixel_u) / focal, (depth.shape[0] * 0.5 - pixel_v) / focal], dtype=np.float64)
            body_ray /= np.linalg.norm(body_ray) + 1e-12
            component_surface = surface_forward[rows, cols]
            measured_surface_forward = float(np.min(component_surface))
            surface_ray_range = measured_surface_forward / max(float(body_ray[0]), 1e-06)
            camera_range = surface_ray_range + TARGET_BODY_RADIUS_M
            world_position = camera_position + rotation @ body_ray * camera_range
            z_error = abs(float(world_position[2]) - float(target_z_est))
            if z_error > DEPTH_TARGET_ABSOLUTE_NEAR_Z_GATE_M:
                continue
            track_error = 0.0 if track is None else float(np.linalg.norm(world_position - track))
            if track is not None and track_error > 1.25:
                continue
            clue_error = 0.0 if clue_xy_est is None else float(np.linalg.norm(world_position[:2] - np.asarray(clue_xy_est, dtype=np.float64)))
            score = 2.0 * z_error + track_error + 0.02 * clue_error
            if score >= best_score:
                continue
            relative = world_position - position
            view_forward, horizontal, vertical = camera_angles(relative, float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
            if view_forward <= 0.0:
                continue
            best_score = score
            best = Detection(position=world_position, range_m=float(np.linalg.norm(relative)), horizontal_angle=horizontal, vertical_angle=vertical, pixel_x=pixel_u, pixel_y=pixel_v, image_width=int(depth.shape[1]), image_height=int(depth.shape[0]), camera_range_m=camera_range, component_pixels=pixels, component_bbox_width_px=float((cols.max() - cols.min() + 1) * pool), component_bbox_height_px=float((rows.max() - rows.min() + 1) * pool), component_depth_span_m=float(np.ptp(component_surface)), clue_error_m=clue_error, source='near_absolute')
        return best

    def _near_track_detection(self, drone_pos: np.ndarray, drone_rpy: np.ndarray, depth: np.ndarray, pooled: np.ndarray, clue_xy_est: Optional[np.ndarray], track_position_est: Optional[np.ndarray], target_z_est: Optional[float]) -> Optional[Detection]:
        if track_position_est is None:
            return None
        position = np.asarray(drone_pos, dtype=np.float64)
        track_position = np.asarray(track_position_est, dtype=np.float64)
        if track_position.shape != (3,):
            return None
        track_range = float(np.linalg.norm(track_position - position))
        if not 0.0 < track_range <= DEPTH_TARGET_TRACK_ROI_MAX_RANGE_M:
            return None
        rotation = rotation_matrix_from_rpy(float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
        camera_position = position + 0.13 * rotation[:, 0] + 0.05 * rotation[:, 2]
        camera_track = rotation.T @ (track_position - camera_position)
        forward = float(camera_track[0])
        if forward <= 0.05:
            return None
        focal = depth.shape[1] * 0.5 / math.tan(math.radians(DEPTH_TARGET_FOV_DEG) * 0.5)
        pixel_u_est = depth.shape[1] * 0.5 - float(camera_track[1]) / forward * focal
        pixel_v_est = depth.shape[0] * 0.5 - float(camera_track[2]) / forward * focal
        half_px = float(np.clip(focal * 0.3 / max(forward, 0.15), DEPTH_TARGET_TRACK_ROI_MIN_HALF_PX, DEPTH_TARGET_TRACK_ROI_MAX_HALF_PX))
        if pixel_u_est < -half_px or pixel_u_est >= depth.shape[1] + half_px or pixel_v_est < -half_px or (pixel_v_est >= depth.shape[0] + half_px):
            return None
        pool = int(DEPTH_TARGET_POOL)
        centre_col = pixel_u_est / pool - 0.5
        centre_row = pixel_v_est / pool - 0.5
        half_pool = max(2, int(math.ceil(half_px / pool)))
        row0 = max(0, int(math.floor(centre_row)) - half_pool)
        row1 = min(pooled.shape[0], int(math.ceil(centre_row)) + half_pool + 1)
        col0 = max(0, int(math.floor(centre_col)) - half_pool)
        col1 = min(pooled.shape[1], int(math.ceil(centre_col)) + half_pool + 1)
        if row0 >= row1 or col0 >= col1:
            return None
        camera_track_norm = float(np.linalg.norm(camera_track))
        expected_surface_forward = forward - TARGET_BODY_RADIUS_M * (forward / max(camera_track_norm, 1e-06))
        roi = pooled[row0:row1, col0:col1]
        surface_forward = roi * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        if track_range > DEPTH_TARGET_ABSOLUTE_NEAR_MAX_RANGE_M and float(np.ptp(surface_forward)) < DEPTH_TARGET_CONTRAST_MIN_M:
            return None
        depth_error = np.abs(surface_forward - expected_surface_forward)
        valid = (roi > 0.0) & (roi < DEPTH_TARGET_THRESHOLD) & (depth_error <= DEPTH_TARGET_TRACK_ROI_DEPTH_TOLERANCE_M)
        local_rows, local_cols = np.where(valid)
        if len(local_rows) < DEPTH_TARGET_MIN_PIXELS:
            return None
        rows = local_rows + row0
        cols = local_cols + col0
        row_delta_px = (rows.astype(np.float64) + 0.5) * pool - pixel_v_est
        col_delta_px = (cols.astype(np.float64) + 0.5) * pool - pixel_u_est
        sigma_px = max(6.0, half_px * 0.35)
        spatial_weight = np.exp(-0.5 * (row_delta_px * row_delta_px + col_delta_px * col_delta_px) / (sigma_px * sigma_px))
        selected_depth_error = depth_error[local_rows, local_cols]
        depth_weight = np.exp(-0.5 * (selected_depth_error / max(0.15, DEPTH_TARGET_TRACK_ROI_DEPTH_TOLERANCE_M * 0.5)) ** 2)
        weights = spatial_weight * depth_weight
        peak = float(np.max(weights))
        keep = weights >= max(1e-06, peak * 0.2)
        if int(np.count_nonzero(keep)) < DEPTH_TARGET_MIN_PIXELS:
            return None
        rows = rows[keep]
        cols = cols[keep]
        weights = weights[keep]
        selected_surface = surface_forward[local_rows, local_cols][keep]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-09:
            return None
        pixel_u = float(np.sum((cols + 0.5) * pool * weights) / weight_sum)
        pixel_v = float(np.sum((rows + 0.5) * pool * weights) / weight_sum)
        measured_surface_forward = float(np.sum(selected_surface * weights) / weight_sum)
        body_ray = np.array([1.0, (depth.shape[1] * 0.5 - pixel_u) / focal, (depth.shape[0] * 0.5 - pixel_v) / focal], dtype=np.float64)
        body_ray /= np.linalg.norm(body_ray) + 1e-12
        surface_ray_range = measured_surface_forward / max(float(body_ray[0]), 1e-06)
        camera_range = surface_ray_range + TARGET_BODY_RADIUS_M
        world_position = camera_position + rotation @ body_ray * camera_range
        if float(np.linalg.norm(world_position - track_position)) > DEPTH_TARGET_TRACK_ROI_POSITION_GATE_M:
            return None
        if not TARGET_Z_TENTATIVE_MIN_M <= float(world_position[2]) <= TARGET_Z_TENTATIVE_MAX_M:
            return None
        if target_z_est is not None and abs(float(world_position[2]) - float(target_z_est)) > DEPTH_TARGET_ABSOLUTE_NEAR_Z_GATE_M:
            return None
        relative = world_position - position
        view_forward, horizontal, vertical = camera_angles(relative, float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
        if view_forward <= 0.0:
            return None
        clue_error = 0.0 if clue_xy_est is None else float(np.linalg.norm(world_position[:2] - np.asarray(clue_xy_est, dtype=np.float64)))
        if track_range > DEPTH_TARGET_TRACK_ROI_UNCONDITIONAL_RANGE_M:
            separated_low_track = target_z_est is not None and float(target_z_est) <= DEPTH_TARGET_TRACK_ROI_EXTENDED_MAX_TARGET_Z_M and (clue_xy_est is not None) and (clue_error >= DEPTH_TARGET_TRACK_ROI_EXTENDED_MIN_CLUE_ERROR_M)
            if not separated_low_track:
                return None
        return Detection(position=world_position, range_m=float(np.linalg.norm(relative)), horizontal_angle=horizontal, vertical_angle=vertical, pixel_x=pixel_u, pixel_y=pixel_v, image_width=int(depth.shape[1]), image_height=int(depth.shape[0]), camera_range_m=camera_range, component_pixels=int(len(rows)), component_bbox_width_px=float((cols.max() - cols.min() + 1) * pool), component_bbox_height_px=float((rows.max() - rows.min() + 1) * pool), component_depth_span_m=float(np.ptp(selected_surface)), clue_error_m=clue_error, source='near_track_roi')

    def _depth_detection(self, drone_pos: np.ndarray, drone_rpy: np.ndarray, depth_image: np.ndarray, clue_xy_est: Optional[np.ndarray], track_position_est: Optional[np.ndarray], target_z_est: Optional[float]) -> Optional[Detection]:
        if depth_image is None:
            return None
        depth = np.asarray(depth_image, dtype=np.float32).squeeze()
        if depth.ndim != 2 or min(depth.shape) < DEPTH_TARGET_POOL:
            return None
        pool = DEPTH_TARGET_POOL
        height = depth.shape[0] // pool
        width = depth.shape[1] // pool
        pooled = _min_pool_depth(depth, pool)
        foreground = (pooled > 0.0) & (pooled < DEPTH_TARGET_THRESHOLD)
        label_sets: list[tuple[np.ndarray, int, bool]] = []
        labels, count = ndimage.label(foreground)
        if count:
            label_sets.append((labels, int(count), False))
        local_background = ndimage.maximum_filter(pooled, size=int(DEPTH_TARGET_CONTRAST_WINDOW), mode='nearest')
        contrast_threshold = float(DEPTH_TARGET_CONTRAST_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
        contrast_foreground = foreground & (local_background - pooled >= contrast_threshold)
        contrast_labels, contrast_count = ndimage.label(contrast_foreground)
        if contrast_count:
            label_sets.append((contrast_labels, int(contrast_count), True))
        if not label_sets:
            near_track = self._near_track_detection(drone_pos, drone_rpy, depth, pooled, clue_xy_est, track_position_est, target_z_est)
            if near_track is not None:
                return near_track
            absolute_near = self._absolute_near_detection(drone_pos, drone_rpy, depth, pooled, clue_xy_est, track_position_est, target_z_est)
            return absolute_near
        rotation = rotation_matrix_from_rpy(float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
        position = np.asarray(drone_pos, dtype=np.float64)
        camera_position = position + 0.13 * rotation[:, 0] + 0.05 * rotation[:, 2]
        focal = depth.shape[1] * 0.5 / math.tan(math.radians(DEPTH_TARGET_FOV_DEG) * 0.5)
        best: Optional[Detection] = None
        best_clue_error = DEPTH_TARGET_CLUE_GATE_M
        max_pixels = int(DEPTH_TARGET_MAX_COMPONENT_FRAC * pooled.size)
        for labels, count, is_contrast in label_sets:
            component_slices = ndimage.find_objects(labels, max_label=count)
            for label_id, component_slice in enumerate(component_slices, start=1):
                if component_slice is None:
                    continue
                local_rows, local_cols = np.where(labels[component_slice] == label_id)
                rows = local_rows + int(component_slice[0].start)
                cols = local_cols + int(component_slice[1].start)
                pixels = len(rows)
                if pixels < DEPTH_TARGET_MIN_PIXELS or pixels > max_pixels:
                    continue
                if rows.max() == height - 1 and pixels > 0.01 * pooled.size:
                    continue
                component_depth = pooled[rows, cols]
                nearest = float(component_depth.min())
                component_depth_span_m = float((component_depth.max() - component_depth.min()) * (DEPTH_MAX_M - DEPTH_MIN_M))
                component_width = int(cols.max() - cols.min() + 1)
                if pixels > DEPTH_TARGET_NORMAL_MERGED_MIN_FRAC * pooled.size and component_width > DEPTH_TARGET_NORMAL_MERGED_MIN_WIDTH_FRAC * width and (component_depth_span_m >= DEPTH_TARGET_NORMAL_MERGED_MIN_DEPTH_SPAN_M):
                    continue
                pixel_u = (float(cols.mean()) + 0.5) * pool
                pixel_v = (float(rows.mean()) + 0.5) * pool
                surface_range_m = nearest * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
                if not DEPTH_MIN_M <= surface_range_m <= DEPTH_MAX_M * 0.9:
                    continue
                body_ray = np.array([1.0, (depth.shape[1] * 0.5 - pixel_u) / focal, (depth.shape[0] * 0.5 - pixel_v) / focal], dtype=np.float64)
                body_ray /= np.linalg.norm(body_ray) + 1e-12
                surface_ray_range_m = surface_range_m / max(float(body_ray[0]), 1e-06)
                range_m = surface_ray_range_m + TARGET_BODY_RADIUS_M
                if range_m <= DEPTH_TARGET_NEAR_MERGED_MAX_RANGE_M and component_width > DEPTH_TARGET_NORMAL_MERGED_MIN_WIDTH_FRAC * width and (component_depth_span_m >= DEPTH_TARGET_NORMAL_MERGED_MIN_DEPTH_SPAN_M):
                    continue
                if range_m >= DEPTH_TARGET_DISTANT_COMPONENT_MIN_RANGE_M and pixels > DEPTH_TARGET_DISTANT_MAX_COMPONENT_FRAC * pooled.size:
                    continue
                world_position = camera_position + rotation @ body_ray * range_m
                if not TARGET_Z_TENTATIVE_MIN_M <= float(world_position[2]) <= TARGET_Z_TENTATIVE_MAX_M:
                    continue
                clue_error = 0.0 if clue_xy_est is None else float(np.linalg.norm(world_position[:2] - np.asarray(clue_xy_est, dtype=np.float64)))
                if clue_error >= best_clue_error:
                    continue
                relative = world_position - position
                forward, horizontal, vertical = camera_angles(relative, float(drone_rpy[0]), float(drone_rpy[1]), float(drone_rpy[2]))
                if forward <= 0.0:
                    continue
                best_clue_error = clue_error
                best = Detection(position=world_position, range_m=float(np.linalg.norm(relative)), horizontal_angle=horizontal, vertical_angle=vertical, pixel_x=pixel_u, pixel_y=pixel_v, image_width=int(depth.shape[1]), image_height=int(depth.shape[0]), camera_range_m=range_m, component_pixels=int(pixels), component_bbox_width_px=float((cols.max() - cols.min() + 1) * pool), component_bbox_height_px=float((rows.max() - rows.min() + 1) * pool), component_depth_span_m=component_depth_span_m, clue_error_m=float(clue_error))
            if best is not None:
                break
        near_track = self._near_track_detection(drone_pos, drone_rpy, depth, pooled, clue_xy_est, track_position_est, target_z_est)
        if near_track is not None and (best is None or float(np.linalg.norm(np.asarray(best.position, dtype=np.float64) - np.asarray(track_position_est, dtype=np.float64))) > DEPTH_TARGET_TRACK_ROI_POSITION_GATE_M):
            return near_track
        track_inconsistent = bool(best is not None and track_position_est is not None and (float(np.linalg.norm(np.asarray(best.position, dtype=np.float64) - np.asarray(track_position_est, dtype=np.float64))) > DEPTH_TARGET_TRACK_ROI_POSITION_GATE_M))
        if best is None or track_inconsistent:
            absolute_near = self._absolute_near_detection(drone_pos, drone_rpy, depth, pooled, clue_xy_est, track_position_est, target_z_est)
            if absolute_near is not None:
                return absolute_near
            if track_inconsistent:
                return None
        return best
OracleTargetDetector = HybridTargetDetector

class TargetTracker:

    def __init__(self, confirm_window: int=5, velocity_history: int=VELOCITY_HISTORY_FRAMES, velocity_clamp: float=TARGET_VELOCITY_CLAMP):
        self.confirm_window = int(confirm_window)
        self.velocity_clamp = velocity_clamp
        self._detection_history: deque = deque(maxlen=self.confirm_window)
        self._samples_t: deque = deque(maxlen=int(velocity_history))
        self._samples_pos: deque = deque(maxlen=int(velocity_history))
        self.last_position: Optional[np.ndarray] = None
        self.last_velocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self.last_acceleration: np.ndarray = np.zeros(3, dtype=np.float64)
        self.last_innovation: np.ndarray = np.zeros(3, dtype=np.float64)
        self.last_measurement_interval: float = 0.0
        self.last_detection_time: float = float('-inf')
        self.prefer_weighted_fit = False
        self._filtered_position: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._detection_history.clear()
        self._samples_t.clear()
        self._samples_pos.clear()
        self.last_position = None
        self.last_velocity = np.zeros(3, dtype=np.float64)
        self.last_acceleration = np.zeros(3, dtype=np.float64)
        self.last_innovation = np.zeros(3, dtype=np.float64)
        self.last_measurement_interval = 0.0
        self.last_detection_time = float('-inf')
        self.prefer_weighted_fit = False
        self._filtered_position = None

    def update(self, t: float, detection: Optional[Detection]) -> None:
        self._detection_history.append(detection is not None)
        if detection is not None:
            previous_detection_time = self.last_detection_time
            self.last_detection_time = t
            self._samples_t.append(t)
            self._samples_pos.append(detection.position)
            if len(self._samples_t) >= 2:
                self.last_velocity = self._estimate_velocity()
            if len(self._samples_t) >= 4:
                self.last_acceleration = self._estimate_acceleration()
            measurement = np.asarray(detection.position, dtype=np.float64)
            if self._filtered_position is None:
                self.last_innovation = np.zeros(3, dtype=np.float64)
                self.last_measurement_interval = 0.0
                self._filtered_position = measurement.copy()
            else:
                dt = max(0.0, float(t) - float(previous_detection_time))
                prediction = self._filtered_position + self.last_velocity * dt
                innovation = measurement - prediction
                self.last_innovation = innovation.copy()
                self.last_measurement_interval = dt
                self._filtered_position = prediction + TARGET_POSITION_FILTER_ALPHA * innovation
            self.last_position = self._filtered_position.copy()

    def recent_hits(self, window: int) -> int:
        history = list(self._detection_history)[-window:]
        return sum((1 for h in history if h))

    def missing_time(self, t: float) -> float:
        if self.last_position is None:
            return float('inf')
        return t - self.last_detection_time

    def predicted_position(self, t: float) -> Optional[np.ndarray]:
        if self.last_position is None:
            return None
        elapsed = max(0.0, t - self.last_detection_time)
        return self.last_position + self.last_velocity * elapsed

    def _estimate_velocity(self) -> np.ndarray:
        if self.prefer_weighted_fit:
            if len(self._samples_t) < 4:
                dt = float(self._samples_t[-1] - self._samples_t[0])
                if dt <= 1e-08:
                    return self.last_velocity
                velocity = (np.asarray(self._samples_pos[-1], dtype=np.float64) - np.asarray(self._samples_pos[0], dtype=np.float64)) / dt
                return limit_norm(velocity, self.velocity_clamp)
            coefficients = self._weighted_motion_fit()
            if coefficients is not None:
                return limit_norm(coefficients[1], self.velocity_clamp)
        times = np.asarray(self._samples_t, dtype=np.float64)[-LEGACY_MOTION_FIT_FRAMES:]
        positions = np.asarray(self._samples_pos, dtype=np.float64)[-LEGACY_MOTION_FIT_FRAMES:]
        centered_time = times - times.mean()
        denominator = float(np.dot(centered_time, centered_time))
        if denominator <= 1e-08:
            return self.last_velocity
        velocity = (centered_time[:, None] * positions).sum(axis=0) / denominator
        return limit_norm(velocity, self.velocity_clamp)

    def _estimate_acceleration(self) -> np.ndarray:
        if self.prefer_weighted_fit:
            coefficients = self._weighted_motion_fit()
            if coefficients is not None:
                return limit_norm(coefficients[2], TARGET_ACCELERATION_CLAMP)
        times = np.asarray(self._samples_t, dtype=np.float64)[-LEGACY_MOTION_FIT_FRAMES:]
        positions = np.asarray(self._samples_pos, dtype=np.float64)[-LEGACY_MOTION_FIT_FRAMES:]
        relative_time = times - times[-1]
        design = np.column_stack((np.ones_like(relative_time), relative_time, 0.5 * relative_time * relative_time))
        try:
            coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        except np.linalg.LinAlgError:
            return self.last_acceleration
        return limit_norm(coefficients[2], TARGET_ACCELERATION_CLAMP)

    def _weighted_motion_fit(self) -> Optional[np.ndarray]:
        times = np.asarray(self._samples_t, dtype=np.float64)
        positions = np.asarray(self._samples_pos, dtype=np.float64)
        relative_time = times - times[-1]
        design = np.column_stack((np.ones_like(relative_time), relative_time, 0.5 * relative_time * relative_time))
        weights = np.exp(relative_time / TARGET_FIT_RECENCY_S)
        try:
            coefficients, _, _, _ = np.linalg.lstsq(design * weights[:, None], positions * weights[:, None], rcond=None)
        except np.linalg.LinAlgError:
            return None
        return coefficients

    def drop(self) -> None:
        self._detection_history.clear()
        self._samples_t.clear()
        self._samples_pos.clear()
        self.last_position = None
        self.last_velocity = np.zeros(3, dtype=np.float64)
        self.last_acceleration = np.zeros(3, dtype=np.float64)
        self.last_innovation = np.zeros(3, dtype=np.float64)
        self.last_measurement_interval = 0.0
        self.last_detection_time = float('-inf')
        self._filtered_position = None

def calculate_intercept_velocity(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, max_speed: float=MAX_SPEED, target_velocity_clamp: float=TARGET_VELOCITY_CLAMP) -> np.ndarray:
    r = np.asarray(target_position, dtype=np.float64) - np.asarray(drone_position, dtype=np.float64)
    r_sq = float(np.dot(r, r))
    if r_sq < 1e-08:
        return np.zeros(3, dtype=np.float64)
    vt = limit_norm(np.asarray(target_velocity, dtype=np.float64), target_velocity_clamp)
    rv = float(np.dot(r, vt))
    vt_sq = float(np.dot(vt, vt))
    discriminant = rv * rv + (max_speed * max_speed - vt_sq) * r_sq
    if discriminant <= 0.0:
        return max_speed * r / math.sqrt(r_sq)
    closing_factor = (-rv + math.sqrt(discriminant)) / r_sq
    command = vt + closing_factor * r
    return limit_norm(command, max_speed)

def apply_vertical_limit(raw_intercept: np.ndarray, max_intercept_vz: float=MAX_INTERCEPT_VZ, max_speed: float=MAX_SPEED) -> np.ndarray:
    vz = float(np.clip(raw_intercept[2], -max_intercept_vz, max_intercept_vz))
    horizontal = raw_intercept[:2]
    if float(np.linalg.norm(horizontal)) < 1e-06:
        return limit_norm(raw_intercept, max_speed)
    horizontal_direction = unit(horizontal)
    horizontal_speed = math.sqrt(max(0.0, max_speed * max_speed - vz * vz))
    return np.array([horizontal_speed * horizontal_direction[0], horizontal_speed * horizontal_direction[1], vz], dtype=np.float64)

def chase_speed_limit(distance: float, max_speed: float=MAX_SPEED, slowdown_range_m: float=None, minimum_speed: float=None) -> float:
    if slowdown_range_m is None:
        slowdown_range_m = CHASE_SLOWDOWN_RANGE_M
    if minimum_speed is None:
        minimum_speed = CHASE_MIN_SPEED_MPS
    slowdown_range_m = max(1e-09, float(slowdown_range_m))
    fraction = min(1.0, max(0.0, float(distance)) / slowdown_range_m)
    return float(minimum_speed) + (float(max_speed) - float(minimum_speed)) * fraction

def direct_pursuit_command(drone_position: np.ndarray, target_position: np.ndarray, max_speed: float=MAX_SPEED) -> np.ndarray:
    delta = np.asarray(target_position, dtype=np.float64) - np.asarray(drone_position, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-09:
        return np.zeros(3, dtype=np.float64)
    return chase_speed_limit(distance, max_speed=max_speed) * delta / distance

def proportional_navigation_command(drone_position: np.ndarray, drone_velocity: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, navigation_gain: float, max_lateral_accel: float, response_horizon_s: float, max_speed: float=MAX_SPEED) -> np.ndarray:
    relative_position = np.asarray(target_position, dtype=np.float64) - np.asarray(drone_position, dtype=np.float64)
    distance = float(np.linalg.norm(relative_position))
    if distance < 1e-09:
        return np.zeros(3, dtype=np.float64)
    line_of_sight = relative_position / distance
    relative_velocity = np.asarray(target_velocity, dtype=np.float64) - np.asarray(drone_velocity, dtype=np.float64)
    radial_relative_speed = float(np.dot(relative_velocity, line_of_sight))
    closing_speed = max(0.0, -radial_relative_speed)
    los_tangent_rate = (relative_velocity - radial_relative_speed * line_of_sight) / distance
    lateral_acceleration = float(navigation_gain) * closing_speed * los_tangent_rate
    lateral_acceleration = limit_norm(lateral_acceleration, max_lateral_accel)
    base_velocity = max_speed * line_of_sight
    predicted_velocity = base_velocity + float(response_horizon_s) * lateral_acceleration
    return max_speed * unit(predicted_velocity)

def blend_lead_pure_pursuit(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, max_speed: float=MAX_SPEED) -> np.ndarray:
    drone_position = np.asarray(drone_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    lead_velocity = calculate_intercept_velocity(drone_position, target_position, target_velocity, max_speed)
    pure_pursuit = max_speed * unit(target_position - drone_position)
    distance = float(np.linalg.norm(target_position - drone_position))
    if distance >= LEAD_PURE_PURSUIT_FAR_M:
        lead_weight = 1.0
    elif distance <= LEAD_PURE_PURSUIT_NEAR_M:
        lead_weight = LEAD_WEIGHT_NEAR
    else:
        span = LEAD_PURE_PURSUIT_FAR_M - LEAD_PURE_PURSUIT_NEAR_M
        lead_weight = LEAD_WEIGHT_NEAR + (1.0 - LEAD_WEIGHT_NEAR) * (distance - LEAD_PURE_PURSUIT_NEAR_M) / span
    v_command = lead_weight * lead_velocity + (1.0 - lead_weight) * pure_pursuit
    v_command = limit_norm(v_command, max_speed)
    return apply_vertical_limit(v_command, max_speed=max_speed)

def short_horizon_visible_chase(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, target_acceleration: np.ndarray, max_speed: float=MAX_SPEED, lead_horizon_s: float=VISIBLE_CHASE_LEAD_HORIZON_S) -> np.ndarray:
    drone_position = np.asarray(drone_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_velocity = limit_norm(np.asarray(target_velocity, dtype=np.float64), TARGET_VELOCITY_CLAMP)
    target_acceleration = np.asarray(target_acceleration, dtype=np.float64)
    horizon = max(0.0, float(lead_horizon_s))
    accel_norm = float(np.linalg.norm(target_acceleration))
    rolloff = max(1e-06, float(TERMINAL_ACCEL_ROLLOFF_MPS2))
    accel_scale = 1.0 / (1.0 + (accel_norm / rolloff) ** 2)
    aim_point = target_position + target_velocity * horizon + 0.5 * accel_scale * target_acceleration * horizon * horizon
    command = max_speed * unit(aim_point - drone_position)
    return apply_vertical_limit(command, max_speed=max_speed)

def terminal_recenter_command(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, target_acceleration: np.ndarray=None, drone_velocity: np.ndarray=None, max_speed: float=MAX_SPEED, tangent_scale: float=None, closing_speed_mps: float=None) -> np.ndarray:
    relative_position = np.asarray(target_position, dtype=np.float64) - np.asarray(drone_position, dtype=np.float64)
    line_of_sight = unit(relative_position)
    velocity = limit_norm(np.asarray(target_velocity, dtype=np.float64), TARGET_VELOCITY_CLAMP)
    chaser_velocity = np.zeros(3, dtype=np.float64) if drone_velocity is None else np.asarray(drone_velocity, dtype=np.float64)
    acceleration = np.zeros(3, dtype=np.float64) if target_acceleration is None else np.asarray(target_acceleration, dtype=np.float64)
    if tangent_scale is None:
        tangent_scale = TERMINAL_RECENTER_TANGENT_SCALE
    if closing_speed_mps is None:
        closing_speed_mps = TERMINAL_RECENTER_CLOSING_SPEED_MPS
    speed = float(np.linalg.norm(velocity))
    longitudinal_deceleration = -float(np.dot(velocity, acceleration)) / speed if speed > 1e-06 else 0.0
    if longitudinal_deceleration >= TERMINAL_RECENTER_STOP_DECEL_MPS2:
        return np.zeros(3, dtype=np.float64)
    if parallel_chase_speed_lock(relative_position, chaser_velocity, velocity):
        return max_speed * line_of_sight
    target_tangent = velocity - float(np.dot(velocity, line_of_sight)) * line_of_sight
    lead_offset = limit_norm(target_tangent * TERMINAL_RECENTER_LEAD_HORIZON_S, TERMINAL_RECENTER_LEAD_MAX_M)
    lead_correction = lead_offset / max(TERMINAL_RECENTER_LEAD_RESPONSE_S, 1e-06)
    predicted_target_tangent = float(tangent_scale) * target_tangent + lead_correction
    steering_tangent = predicted_target_tangent
    if drone_velocity is not None:
        pn_command = proportional_navigation_command(drone_position, chaser_velocity, target_position, velocity, PROPORTIONAL_NAVIGATION_GAIN, PROPORTIONAL_NAVIGATION_ACCEL_MPS2, PROPORTIONAL_NAVIGATION_RESPONSE_S, max_speed)
        pn_tangent = pn_command - float(np.dot(pn_command, line_of_sight)) * line_of_sight
        if float(np.linalg.norm(pn_tangent)) > float(np.linalg.norm(predicted_target_tangent)):
            steering_tangent = pn_tangent
    command = steering_tangent + float(closing_speed_mps) * line_of_sight
    return limit_norm(command, max_speed)

def closest_approach_metrics(relative_position: np.ndarray, drone_velocity: np.ndarray, target_velocity: np.ndarray, horizon_s: float=TERMINAL_RECENTER_CPA_HORIZON_S) -> tuple[float, float, float]:
    relative_position = np.asarray(relative_position, dtype=np.float64)
    relative_velocity = np.asarray(target_velocity, dtype=np.float64) - np.asarray(drone_velocity, dtype=np.float64)
    distance = float(np.linalg.norm(relative_position))
    line_of_sight = unit(relative_position)
    relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
    if relative_speed_sq <= 1e-08:
        time_to_cpa = 0.0
    else:
        time_to_cpa = float(np.clip(-float(np.dot(relative_position, relative_velocity)) / relative_speed_sq, 0.0, max(0.0, float(horizon_s))))
    miss_distance = float(np.linalg.norm(relative_position + relative_velocity * time_to_cpa))
    closing_speed = -float(np.dot(relative_velocity, line_of_sight)) if distance > 1e-09 else 0.0
    return (time_to_cpa, miss_distance, closing_speed)

def parallel_chase_speed_lock(relative_position: np.ndarray, drone_velocity: np.ndarray, target_velocity: np.ndarray, alignment_deg: float=PARALLEL_CHASE_ALIGNMENT_DEG, min_drone_speed: float=PARALLEL_CHASE_MIN_DRONE_SPEED_MPS, min_target_speed: float=PARALLEL_CHASE_MIN_TARGET_SPEED_MPS) -> bool:
    line_of_sight = unit(np.asarray(relative_position, dtype=np.float64))
    drone_velocity = np.asarray(drone_velocity, dtype=np.float64)
    target_velocity = np.asarray(target_velocity, dtype=np.float64)
    drone_speed = float(np.linalg.norm(drone_velocity))
    target_speed = float(np.linalg.norm(target_velocity))
    if drone_speed < float(min_drone_speed) or target_speed < float(min_target_speed):
        return False
    drone_direction = drone_velocity / drone_speed
    target_direction = target_velocity / target_speed
    threshold = math.cos(math.radians(float(alignment_deg)))
    return bool(float(np.dot(drone_direction, target_direction)) >= threshold and float(np.dot(drone_direction, line_of_sight)) >= threshold and (float(np.dot(target_direction, line_of_sight)) >= threshold))

def terminal_homing_command(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, max_speed: float=MAX_SPEED, target_velocity_clamp: float=TARGET_VELOCITY_CLAMP, max_intercept_vz: float=MAX_INTERCEPT_VZ, drone_velocity: np.ndarray=None, target_acceleration: np.ndarray=None, actuator_lag_s: float=TERMINAL_ACTUATOR_LAG_S, target_accel_scale: float=None) -> np.ndarray:
    drone_position = np.asarray(drone_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_velocity = np.asarray(target_velocity, dtype=np.float64)
    drone_velocity = np.zeros(3, dtype=np.float64) if drone_velocity is None else np.asarray(drone_velocity, dtype=np.float64)
    target_acceleration = np.zeros(3, dtype=np.float64) if target_acceleration is None else np.asarray(target_acceleration, dtype=np.float64)
    lag = max(0.0, float(actuator_lag_s))
    accel_norm = float(np.linalg.norm(target_acceleration))
    rolloff = max(1e-06, float(TERMINAL_ACCEL_ROLLOFF_MPS2))
    accel_scale = 1.0 / (1.0 + (accel_norm / rolloff) ** 2) if target_accel_scale is None else float(np.clip(target_accel_scale, 0.0, 1.0))
    projected_drone = drone_position + drone_velocity * lag
    projected_target = target_position + target_velocity * lag + 0.5 * accel_scale * target_acceleration * lag * lag
    projected_target_velocity = target_velocity + accel_scale * target_acceleration * lag
    command = calculate_intercept_velocity(projected_drone, projected_target, projected_target_velocity, max_speed, target_velocity_clamp)
    return apply_vertical_limit(command, max_intercept_vz=max_intercept_vz, max_speed=max_speed)

def is_head_on(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, angle_deg: float=HEAD_ON_ANGLE_DEG, range_m: float=HEAD_ON_RANGE_M) -> bool:
    to_chaser = np.asarray(drone_position, dtype=np.float64) - np.asarray(target_position, dtype=np.float64)
    dist = float(np.linalg.norm(to_chaser))
    if dist > range_m or dist < 1e-06:
        return False
    cos_angle = float(np.dot(unit(np.asarray(target_velocity, dtype=np.float64)), unit(to_chaser)))
    return cos_angle > math.cos(math.radians(angle_deg))

def head_on_lead_command(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, max_speed: float=MAX_SPEED, target_velocity_clamp: float=TARGET_VELOCITY_CLAMP, max_intercept_vz: float=MAX_INTERCEPT_VZ) -> np.ndarray:
    drone_position = np.asarray(drone_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    vt = limit_norm(np.asarray(target_velocity, dtype=np.float64), target_velocity_clamp)
    distance = float(np.linalg.norm(target_position - drone_position))
    lead_point = target_position + vt * (2.0 * distance / max_speed)
    command = max_speed * unit(lead_point - drone_position)
    return apply_vertical_limit(command, max_intercept_vz=max_intercept_vz, max_speed=max_speed)

def is_steep_vertical(drone_position: np.ndarray, target_position: np.ndarray, angle_deg: float=STEEP_VERTICAL_ANGLE_DEG) -> bool:
    to_target = np.asarray(target_position, dtype=np.float64) - np.asarray(drone_position, dtype=np.float64)
    horizontal = float(np.linalg.norm(to_target[:2]))
    vertical = abs(float(to_target[2]))
    return vertical > horizontal * math.tan(math.radians(angle_deg))

def vertical_dive_command(drone_position: np.ndarray, target_position: np.ndarray, max_speed: float=MAX_SPEED) -> np.ndarray:
    dz = float(target_position[2]) - float(drone_position[2])
    vz = math.copysign(max_speed, dz) if abs(dz) > 1e-06 else 0.0
    return np.array([0.0, 0.0, vz], dtype=np.float64)

def yaw_toward_predicted_target(drone_position: np.ndarray, target_position: np.ndarray, target_velocity: np.ndarray, prediction_time: float=YAW_PREDICTION_TIME) -> float:
    yaw_point = np.asarray(target_position, dtype=np.float64) + prediction_time * np.asarray(target_velocity, dtype=np.float64)
    drone_position = np.asarray(drone_position, dtype=np.float64)
    yaw_target = math.atan2(yaw_point[1] - drone_position[1], yaw_point[0] - drone_position[0])
    return wrap_angle(yaw_target)

def yaw_toward_current_target(drone_position: np.ndarray, target_position: np.ndarray) -> float:
    drone_position = np.asarray(drone_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    yaw_target = math.atan2(target_position[1] - drone_position[1], target_position[0] - drone_position[0])
    return wrap_angle(yaw_target)
VISIBLE_FEATURE_FIELDS = ('distance', 'closing_speed', 'cpa_time', 'cpa_miss', 'horizontal_error', 'vertical_error', 'horizontal_error_rate', 'vertical_error_rate', 'los_rate', 'drone_speed', 'target_speed', 'target_longitudinal_deceleration', 'target_turn_rate', 'drone_target_alignment', 'drone_los_alignment', 'target_los_alignment', 'z_error', 'tilt', 'detection_present', 'missing_time', 'estimator_confidence')
EXTENDED_VISIBLE_FEATURE_FIELDS = ('time_s', 'horizontal_distance', 'target_velocity_radial', 'target_velocity_lateral', 'target_velocity_vertical', 'drone_velocity_radial', 'drone_velocity_lateral', 'drone_velocity_vertical', 'target_acceleration_radial', 'target_acceleration_lateral', 'target_acceleration_vertical', 'target_acceleration_magnitude', 'target_lateral_alignment', 'target_away_alignment', 'target_lateral_rate_0p2', 'target_lateral_rate_0p5', 'target_heading_change_0p2', 'target_heading_change_0p5', 'target_speed_change_0p2', 'target_speed_change_0p5', 'closing_speed_change_0p2', 'closing_speed_change_0p5', 'detection_duty_0p2', 'detection_duty_0p5', 'tracker_innovation_norm', 'tracker_innovation_radial', 'tracker_innovation_lateral', 'tracker_innovation_vertical', 'tracker_measurement_interval', 'track_age', 'target_displacement_from_track_start', 'drone_command_speed', 'drone_speed_error', 'command_velocity_alignment', 'roll', 'pitch', 'roll_rate', 'pitch_rate', 'yaw_rate', 'jink_frequency_estimate_hz', 'jink_sign_changes_4s', 'prior_pass_count', 'minimum_estimated_distance')
DETECTOR_VISIBLE_FEATURE_FIELDS = ('detection_range_m', 'detection_component_pixels', 'detection_bbox_width_frac', 'detection_bbox_height_frac', 'detection_component_fill', 'detection_depth_span_m', 'detection_clue_error_m', 'pixel_delta_x_frac', 'pixel_delta_y_frac', 'pixel_delta_norm_frac')
CAUSAL_VISIBLE_FEATURE_FIELDS = ('relative_velocity_radial', 'relative_velocity_lateral', 'relative_velocity_vertical', 'signed_cpa_lateral_m', 'signed_cpa_vertical_m', 'range_change_0p2', 'range_change_0p5', 'horizontal_error_change_0p2', 'horizontal_error_change_0p5', 'vertical_error_change_0p2', 'vertical_error_change_0p5', 'target_acceleration_lateral_change_0p2', 'target_acceleration_lateral_change_0p5', 'command_velocity_radial', 'command_velocity_lateral', 'command_velocity_vertical')
FIRST_DROPOUT_FEATURE_FIELDS = ('memory_distance', 'visibility_confirmed', 'time_s', 'drone_speed', 'target_speed', 'drone_target_alignment', 'drone_los_alignment', 'target_los_alignment', 'horizontal_exit_angle', 'z_error', 'far_noisy_acquisition')

class TerminalProfile:
    NORMAL_PN = 'NORMAL_PN'
    CAPTURE_PN = 'CAPTURE_PN'
    PARALLEL_FULL = 'PARALLEL_FULL'
    PN_TANGENT_BRAKE = 'PN_TANGENT_BRAKE'
    STRONG_TURN_PN = 'STRONG_TURN_PN'
    ACCEL_LEAD = 'ACCEL_LEAD'
    COLLISION_LEAD = 'COLLISION_LEAD'
    TERMINAL_HOMING = 'TERMINAL_HOMING'
    CAMERA_RAY = 'CAMERA_RAY'
    PREDICTED_CAMERA_RAY = 'PREDICTED_CAMERA_RAY'
    STOP_AND_YAW = 'STOP_AND_YAW'
    VERTICAL_ALIGN = 'VERTICAL_ALIGN'
    DIRECT_STABLE = 'DIRECT_STABLE'
    BRAKE_YAW_HOMING = 'BRAKE_YAW_HOMING'
    LATERAL_CUT = 'LATERAL_CUT'
    STAGED_LATERAL_CUT = 'STAGED_LATERAL_CUT'
    STAGED_HIGH_AUTHORITY_PN = 'STAGED_HIGH_AUTHORITY_PN'
    HIGH_AUTHORITY_PN = 'HIGH_AUTHORITY_PN'
    TUNED_PN_045 = 'TUNED_PN_045'
    MEMORY_REACQUIRE = 'MEMORY_REACQUIRE'

@dataclass(frozen=True)
class TerminalContext:
    distance: float
    closing_speed: float
    cpa_time: float
    cpa_miss: float
    horizontal_error: float
    vertical_error: float
    horizontal_error_rate: float
    vertical_error_rate: float
    los_rate: float
    drone_speed: float
    target_speed: float
    target_longitudinal_deceleration: float
    target_turn_rate: float
    drone_target_alignment: float
    drone_los_alignment: float
    target_los_alignment: float
    z_error: float
    tilt: float
    detection_present: bool
    missing_time: float
    estimator_confidence: float

    @property
    def camera_error(self) -> float:
        return max(abs(self.horizontal_error), abs(self.vertical_error))

    @property
    def camera_error_rate(self) -> float:
        return max(abs(self.horizontal_error_rate), abs(self.vertical_error_rate))

@dataclass(frozen=True)
class TerminalDecision:
    rule_id: str
    profile: str
    min_hold_s: float = 0.0
    recenter_tangent_scale: Optional[float] = None
    recenter_closing_speed_mps: Optional[float] = None
    rule_lead_accel_mps2: Optional[float] = None

@dataclass(frozen=True)
class ReacquireDecision:
    rule_id: str
    speed_mps: float
    force_speed: bool = False

@dataclass(frozen=True)
class ReacquireContext:
    memory_distance: float
    visibility_confirmed: bool
    time_s: float
    drone_speed: float
    target_speed: float
    drone_target_alignment: float
    drone_los_alignment: float
    target_los_alignment: float
    horizontal_exit_angle: float
    z_error: float
    far_noisy_acquisition: bool
_ROUTER_FIELDS = VISIBLE_FEATURE_FIELDS + EXTENDED_VISIBLE_FEATURE_FIELDS + DETECTOR_VISIBLE_FEATURE_FIELDS + CAUSAL_VISIBLE_FEATURE_FIELDS + tuple((field for field in FIRST_DROPOUT_FEATURE_FIELDS if field not in VISIBLE_FEATURE_FIELDS + EXTENDED_VISIBLE_FEATURE_FIELDS + DETECTOR_VISIBLE_FEATURE_FIELDS + CAUSAL_VISIBLE_FEATURE_FIELDS))
_ROUTER_PROFILE_NAMES = (TerminalProfile.NORMAL_PN, TerminalProfile.CAPTURE_PN, TerminalProfile.PARALLEL_FULL, TerminalProfile.PN_TANGENT_BRAKE, TerminalProfile.STRONG_TURN_PN, TerminalProfile.ACCEL_LEAD, TerminalProfile.COLLISION_LEAD, TerminalProfile.TERMINAL_HOMING, TerminalProfile.CAMERA_RAY, TerminalProfile.PREDICTED_CAMERA_RAY, TerminalProfile.STOP_AND_YAW, TerminalProfile.VERTICAL_ALIGN, TerminalProfile.DIRECT_STABLE, TerminalProfile.BRAKE_YAW_HOMING, TerminalProfile.LATERAL_CUT, TerminalProfile.STAGED_LATERAL_CUT, TerminalProfile.STAGED_HIGH_AUTHORITY_PN, TerminalProfile.HIGH_AUTHORITY_PN, TerminalProfile.TUNED_PN_045, TerminalProfile.MEMORY_REACQUIRE)

def _router_features(context, extended_context):
    extended = extended_context or {}
    values = []
    for field in _ROUTER_FIELDS:
        if hasattr(context, field):
            value = getattr(context, field)
        else:
            value = extended.get(field, float('nan'))
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)

class _ExactTerminalRouter:

    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.use_deterministic_compute', '1')
        model_path = Path(__file__).resolve().parent / 'terminal_router.onnx'
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        self._last_input = None
        self._last_output = None

    def evaluate(self, features):
        if self._last_input is not None and np.array_equal(self._last_input, features, equal_nan=True):
            return self._last_output
        raw = self._session.run(None, {'x': features})
        output = (int(np.asarray(raw[0]).item()), int(np.asarray(raw[1]).item()), float(np.asarray(raw[2]).item()), float(np.asarray(raw[3]).item()), float(np.asarray(raw[4]).item()), float(np.asarray(raw[5]).item()), int(np.asarray(raw[6]).item()), int(np.asarray(raw[7]).item()))
        self._last_input = features.copy()
        self._last_output = output
        return output
_TERMINAL_ROUTER = _ExactTerminalRouter()

def _optional_router_float(value):
    return None if math.isnan(value) else float(value)

class LearnedTerminalPolicyTable:

    def __init__(self) -> None:
        self._visible_case = None
        self._reacquire_mode: Optional[ReacquireDecision] = None

    def reset(self) -> None:
        self._visible_case = None
        self._reacquire_mode = None

    def reset_visible(self) -> None:
        self._visible_case = None

    @property
    def active_visible_rule_id(self) -> str:
        return 'L00_NONE' if self._visible_case is None else self._visible_case[0]

    @property
    def max_visible_range(self) -> float:
        return 60.0

    def select_visible(self, context, extended_context=None):
        if self._visible_case is not None and context.distance > self._visible_case[2]:
            self._visible_case = None
        if self._visible_case is None:
            result = _TERMINAL_ROUTER.evaluate(_router_features(context, extended_context))
            region, profile, activation, tangent, closing, lead, _generalized, _reacquire = result
            if region >= 0:
                self._visible_case = (f'L{region + 1:03d}', profile, activation, _optional_router_float(tangent), _optional_router_float(closing), _optional_router_float(lead))
        if self._visible_case is None:
            return None
        rule_id, profile, _activation, tangent, closing, lead = self._visible_case
        return TerminalDecision(rule_id, _ROUTER_PROFILE_NAMES[profile], recenter_tangent_scale=tangent, recenter_closing_speed_mps=closing, rule_lead_accel_mps2=lead)

    def select_reacquire(self, context: ReacquireContext) -> ReacquireDecision:
        if self._reacquire_mode is not None:
            return self._reacquire_mode
        if context.memory_distance > 3.0:
            code = _TERMINAL_ROUTER.evaluate(_router_features(context, None))[7]
            if code == 1:
                self._reacquire_mode = ReacquireDecision('R01', 6.0, True)
            elif code == 2:
                self._reacquire_mode = ReacquireDecision('R02', 3.0, True)
            else:
                self._reacquire_mode = ReacquireDecision('R00', 3.0)
            return self._reacquire_mode
        return ReacquireDecision('R00', 3.0)
GENERALIZED_TERMINAL_POLICY_ENABLED = True
GENERALIZED_TERMINAL_ENTER_M = 5.0
GENERALIZED_TERMINAL_EXIT_M = 5.25

class GeneralizedTerminalPolicy:

    def __init__(self) -> None:
        self._classified = False
        self._decision: Optional[TerminalDecision] = None
        self._rule_id = 'G00_INACTIVE'

    def reset(self) -> None:
        self._classified = False
        self._decision = None
        self._rule_id = 'G00_INACTIVE'

    @property
    def active_rule_id(self) -> str:
        return self._rule_id

    def select(self, context, extended_context=None):
        if not GENERALIZED_TERMINAL_POLICY_ENABLED:
            self.reset()
            return None
        if context.distance > GENERALIZED_TERMINAL_EXIT_M:
            self.reset()
            return None
        if self._classified:
            return self._decision
        if context.distance > GENERALIZED_TERMINAL_ENTER_M:
            return None
        code = _TERMINAL_ROUTER.evaluate(_router_features(context, extended_context))[6]
        if code < 0:
            return None
        self._classified = True
        self._rule_id = f'G{code:02d}_ROUTED'
        profile_by_code = {2: TerminalProfile.ACCEL_LEAD, 3: TerminalProfile.TERMINAL_HOMING, 4: TerminalProfile.CAMERA_RAY, 5: TerminalProfile.COLLISION_LEAD, 6: TerminalProfile.LATERAL_CUT, 7: TerminalProfile.STRONG_TURN_PN}
        profile = profile_by_code.get(code)
        self._decision = None if profile is None else TerminalDecision(self._rule_id, profile)
        return self._decision

@dataclass(frozen=True)
class _Rule:
    rule_id: str
    profile: str
    enter: Callable[[TerminalContext], bool]
    stay: Callable[[TerminalContext], bool]
    min_hold_s: float
    preemptive: bool = False

    def decision(self) -> TerminalDecision:
        return TerminalDecision(self.rule_id, self.profile, self.min_hold_s)
_CENTER_ENTER = math.radians(11.4)
_CENTER_PREDICT = math.radians(5.0)
_CAPTURE_MISS = 0.12
_UNSAFE_MISS = 0.2
_PARALLEL_COS = math.cos(math.radians(20.0))

def _capture(c: TerminalContext) -> bool:
    return c.detection_present and c.cpa_time > 0.0 and (c.cpa_miss <= _CAPTURE_MISS) and (c.closing_speed >= 1.0)

def _parallel(c: TerminalContext) -> bool:
    return c.drone_speed >= 2.0 and c.target_speed >= 2.0 and (c.drone_target_alignment >= _PARALLEL_COS) and (c.drone_los_alignment >= _PARALLEL_COS) and (c.target_los_alignment >= _PARALLEL_COS) and (c.target_longitudinal_deceleration < 4.0)

def _turning_or_reversing(c: TerminalContext) -> bool:
    return c.detection_present and c.cpa_miss >= _UNSAFE_MISS and (c.target_turn_rate >= 0.8 or (c.target_longitudinal_deceleration >= 3.0 and c.target_turn_rate >= 0.35))

def _stopping(c: TerminalContext) -> bool:
    return c.target_speed >= 0.5 and c.target_longitudinal_deceleration >= 4.0 and (c.target_turn_rate < 0.8)

def _vertical(c: TerminalContext) -> bool:
    return c.distance <= 2.5 and abs(c.z_error) >= 0.35

def _critical_frame_exit(c: TerminalContext) -> bool:
    return c.detection_present and c.camera_error >= math.radians(30.0) and (c.cpa_miss >= _UNSAFE_MISS) and (c.closing_speed >= 1.5) and (c.target_turn_rate < 1.0)

def _imminent_cpa_miss(c: TerminalContext) -> bool:
    return c.detection_present and c.distance <= 1.75 and (c.cpa_miss >= 0.25) and (c.closing_speed >= 2.0)

def _imminent_high_closing_miss(c: TerminalContext) -> bool:
    return _imminent_cpa_miss(c) and c.closing_speed >= 5.0

def _imminent_medium_closing_miss(c: TerminalContext) -> bool:
    return _imminent_cpa_miss(c) and c.closing_speed < 5.0

def _parallel_stall(c: TerminalContext) -> bool:
    return _parallel(c) and c.distance <= 2.5 and (0.0 <= c.closing_speed < 0.4) and (c.cpa_miss >= 0.4) and (4.7 <= c.target_speed <= 5.2)

def _fast_crossing(c: TerminalContext) -> bool:
    return c.cpa_miss >= _UNSAFE_MISS and c.los_rate >= 0.65

def _high_closing_miss(c: TerminalContext) -> bool:
    return c.cpa_miss >= _UNSAFE_MISS and c.closing_speed >= 3.0 and (c.camera_error >= _CENTER_PREDICT or c.camera_error_rate >= math.radians(5.0))

def _low_closing_diverging(c: TerminalContext) -> bool:
    return c.cpa_miss >= _UNSAFE_MISS and c.closing_speed < 1.0 and c.detection_present

def _boundary_safe(c: TerminalContext) -> bool:
    return c.camera_error >= _CENTER_ENTER and c.cpa_miss < _UNSAFE_MISS

def _boundary_unsafe(c: TerminalContext) -> bool:
    return c.camera_error >= _CENTER_ENTER and c.cpa_miss >= _UNSAFE_MISS

def _unreliable(c: TerminalContext) -> bool:
    return c.detection_present and c.estimator_confidence < 0.6

def _lost(c: TerminalContext) -> bool:
    return not c.detection_present and c.missing_time <= 0.15
_RULES = (_Rule('T01_BRIEF_LOSS', TerminalProfile.MEMORY_REACQUIRE, _lost, lambda c: not c.detection_present and c.missing_time <= 0.15, 0.04, True), _Rule('T02_CAPTURE_CORRIDOR', TerminalProfile.CAPTURE_PN, _capture, lambda c: c.cpa_miss <= 0.16 and c.closing_speed > 0.0, 0.04, True), _Rule('T03_VERTICAL_MISMATCH', TerminalProfile.VERTICAL_ALIGN, _vertical, lambda c: abs(c.z_error) >= 0.2 and c.distance <= 3.0, 0.08), _Rule('T03B_CRITICAL_FRAME_EXIT', TerminalProfile.STOP_AND_YAW, _critical_frame_exit, lambda c: c.camera_error >= math.radians(18.0) and c.cpa_miss >= 0.15 and (c.closing_speed > 0.25), 0.12, True), _Rule('T03C_IMMINENT_HIGH_CLOSING_MISS', TerminalProfile.CAMERA_RAY, _imminent_high_closing_miss, lambda c: c.detection_present and c.distance <= 2.0 and (c.cpa_miss >= 0.15) and (c.closing_speed >= 4.5), 0.08, True), _Rule('T03D_IMMINENT_MEDIUM_CLOSING_MISS', TerminalProfile.PREDICTED_CAMERA_RAY, _imminent_medium_closing_miss, lambda c: c.detection_present and c.distance <= 2.0 and (c.cpa_miss >= 0.15) and (0.5 <= c.closing_speed < 5.0), 0.08, True), _Rule('T04A_HIGH_CLOSING_TURN', TerminalProfile.STRONG_TURN_PN, lambda c: _turning_or_reversing(c) and c.closing_speed >= 5.0, lambda c: c.target_turn_rate >= 0.25 and c.cpa_miss >= 0.15 and (c.closing_speed >= 4.0), 0.1), _Rule('T04B_MEDIUM_CLOSING_TURN', TerminalProfile.ACCEL_LEAD, _turning_or_reversing, lambda c: c.target_turn_rate >= 0.25 and c.cpa_miss >= 0.15, 0.14), _Rule('T05_TARGET_STOPPING', TerminalProfile.STOP_AND_YAW, _stopping, lambda c: c.target_longitudinal_deceleration >= 2.0 and c.target_speed >= 0.3, 0.1), _Rule('T06A_PARALLEL_STALL', TerminalProfile.COLLISION_LEAD, _parallel_stall, lambda c: _parallel(c) and c.distance <= 3.0 and (c.closing_speed < 0.75), 0.2), _Rule('T06_PARALLEL_FLEE', TerminalProfile.PARALLEL_FULL, _parallel, _parallel, 0.1), _Rule('T07_FAST_CROSSING', TerminalProfile.PN_TANGENT_BRAKE, _fast_crossing, lambda c: c.los_rate >= 0.35 and c.cpa_miss >= 0.15, 0.12), _Rule('T08_HIGH_CLOSING_MISS', TerminalProfile.PN_TANGENT_BRAKE, _high_closing_miss, lambda c: c.closing_speed >= 1.0 and c.cpa_miss >= 0.15, 0.1), _Rule('T09_LOW_CLOSING_DIVERGING', TerminalProfile.STRONG_TURN_PN, _low_closing_diverging, lambda c: c.closing_speed < 1.5 and c.cpa_miss >= 0.15, 0.1), _Rule('T10_BOUNDARY_SAFE', TerminalProfile.CAPTURE_PN, _boundary_safe, lambda c: c.camera_error >= math.radians(7.6) and c.cpa_miss < 0.22, 0.06), _Rule('T11_BOUNDARY_UNSAFE', TerminalProfile.PN_TANGENT_BRAKE, _boundary_unsafe, lambda c: c.camera_error >= math.radians(7.6) and c.cpa_miss >= 0.15, 0.1), _Rule('T12_UNRELIABLE_TRACK', TerminalProfile.DIRECT_STABLE, _unreliable, lambda c: c.estimator_confidence < 0.75, 0.1), _Rule('T13_DEFAULT', TerminalProfile.NORMAL_PN, lambda _c: True, lambda _c: False, 0.0))

class TerminalRuleEngine:

    def __init__(self) -> None:
        self._active: Optional[_Rule] = None
        self._entered_at = float('-inf')

    def reset(self) -> None:
        self._active = None
        self._entered_at = float('-inf')

    @property
    def active_rule_id(self) -> str:
        return 'T00_INACTIVE' if self._active is None else self._active.rule_id

    def select(self, t: float, context: TerminalContext) -> TerminalDecision:
        candidate = next((rule for rule in _RULES if rule.enter(context)))
        active = self._active
        if active is not None and active.rule_id != candidate.rule_id:
            held = float(t) - self._entered_at < active.min_hold_s
            candidate_priority = _RULES.index(candidate)
            active_priority = _RULES.index(active)
            if candidate_priority >= active_priority and (not candidate.preemptive) and (held or active.stay(context)):
                candidate = active
        if active is None or active.rule_id != candidate.rule_id:
            self._active = candidate
            self._entered_at = float(t)
        return self._active.decision()
DEV_FORCE_TERMINAL_PROFILE: Optional[str] = None
DEV_FORCE_TERMINAL_RANGE_M = 3.0
DEV_FORCE_REACQUIRE_SPEED: Optional[float] = None
DEV_FORCE_NEAR_REACQUIRE_PROFILE: Optional[str] = None
DEV_LOCAL_SEARCH_RADIAL_WEIGHT: Optional[float] = None
DEV_LOCAL_SEARCH_TANGENT_WEIGHT: Optional[float] = None
DEV_LOCAL_SEARCH_YAW_SCAN_DEG: Optional[float] = None
DEV_LOCAL_SEARCH_YAW_SCAN_PERIOD_S = 4.0

class Phase:
    SEARCH_TRANSIT = 'SEARCH_TRANSIT'
    SEARCH_LOCAL = 'SEARCH_LOCAL'
    TENTATIVE_DETECTION = 'TENTATIVE_DETECTION'
    TRACK_AND_INTERCEPT = 'TRACK_AND_INTERCEPT'
    REACQUIRE = 'REACQUIRE'

def _heading_to(direction_xy: np.ndarray) -> float:
    return math.atan2(direction_xy[1], direction_xy[0])

def detection_matches_track_memory(detection: Optional[Detection], predicted_position: Optional[np.ndarray]) -> bool:
    if detection is None or predicted_position is None:
        return False
    return bool(np.linalg.norm(np.asarray(detection.position, dtype=np.float64) - np.asarray(predicted_position, dtype=np.float64)) <= TRACK_ASSOCIATION_GATE_M)

def clean_centered_predictive_acquisition(detection: Optional[Detection]) -> bool:
    if detection is None or detection.pixel_x is None or detection.image_width is None or (detection.image_width <= 0):
        return False
    half_width = float(detection.image_width) * HIGH_CLOSING_PIXEL_ACQUIRE_CENTER_HALF_FRAC
    centered = abs(float(detection.pixel_x) - 0.5 * float(detection.image_width)) <= half_width
    return bool(centered and HIGH_CLOSING_PIXEL_ACQUIRE_MIN_RANGE_M <= float(detection.range_m) <= HIGH_CLOSING_PIXEL_ACQUIRE_MAX_RANGE_M)

def recenter_lost_yaw_target(current_yaw: float, last_horizontal_angle: float) -> float:
    if abs(float(last_horizontal_angle)) < 1e-08:
        return wrap_angle(float(current_yaw))
    turn_sign = 1.0 if float(last_horizontal_angle) > 0.0 else -1.0
    return wrap_angle(float(current_yaw) - turn_sign * math.radians(TERMINAL_RECENTER_LOST_YAW_OFFSET_DEG))

def encode_action(v_command: np.ndarray, yaw_target: float) -> np.ndarray:
    v_command = np.asarray(v_command, dtype=np.float64)
    norm = float(np.linalg.norm(v_command))
    if norm > MAX_SPEED and norm > 1e-12:
        v_command = v_command * (MAX_SPEED / norm)
    speed_mps = float(np.linalg.norm(v_command))
    if speed_mps < 1e-06:
        direction = np.zeros(3, dtype=np.float64)
        speed_action = 0.0
    else:
        direction = v_command / speed_mps
        speed_action = speed_mps / MAX_SPEED
    return np.array([direction[0], direction[1], direction[2], float(np.clip(speed_action, 0.0, 1.0)), float(np.clip(wrap_angle(yaw_target) / math.pi, -1.0, 1.0))], dtype=np.float32)

class DroneFlightController:

    def __init__(self):
        self.phase = Phase.SEARCH_TRANSIT
        self.frame_index = 0
        self.clue_filter = ClueFilter()
        self.detector = OracleTargetDetector()
        self.tracker = TargetTracker(confirm_window=TARGET_CONFIRM_WINDOW)
        self.terminal_rules = TerminalRuleEngine()
        self.learned_policies = LearnedTerminalPolicyTable()
        self.generalized_terminal_policy = GeneralizedTerminalPolicy()
        self.previous_velocity_command = np.zeros(3, dtype=np.float64)
        self._last_candidate: Optional[Detection] = None
        self._tilt_history: deque = deque()
        self._smoothed_agl: Optional[float] = None
        self._terminal_active = False
        self._terminal_accel_scale = 1.0
        self._terminal_dynamics_enabled = False
        self._remembered_target_z: Optional[float] = None
        self._first_target_z: Optional[float] = None
        self._target_z_history: deque = deque(maxlen=TARGET_Z_MEMORY_FRAMES)
        self._target_z_observation_times: deque = deque(maxlen=TARGET_Z_MEMORY_FRAMES)
        self._target_z_reanchor_history: deque = deque(maxlen=TARGET_Z_REANCHOR_HITS)
        self._target_z_reanchors = 0
        self._last_detection_horizontal_angle = 0.0
        self._previous_target_pixel: Optional[tuple[float, float]] = None
        self._previous_target_pixel_frame: Optional[int] = None
        self._reacquire_started_at: Optional[float] = None
        self._reacquire_started_far = False
        self._acquisition_vertical_boost_until = float('-inf')
        self._large_vertical_acquisition = False
        self._short_visible_chase = False
        self._terminal_recenter_active = False
        self._terminal_recenter_started_at: Optional[float] = None
        self._terminal_recenter_center_frames = 0
        self._previous_camera_error: Optional[float] = None
        self._previous_horizontal_angle: Optional[float] = None
        self._previous_vertical_angle: Optional[float] = None
        self._previous_detection_t: Optional[float] = None
        self._terminal_horizontal_rate = 0.0
        self._terminal_vertical_rate = 0.0
        self._terminal_decision = TerminalDecision('T00_INACTIVE', TerminalProfile.NORMAL_PN)
        self._terminal_context: Optional[TerminalContext] = None
        self._reacquire_context: Optional[ReacquireContext] = None
        self._learned_reacquire_rule_id = 'L00_DEFAULT_REACQUIRE'
        self._terminal_recenter_events = 0
        self._terminal_cpa_time = 0.0
        self._terminal_cpa_miss = float('inf')
        self._terminal_closing_speed = 0.0
        self._parallel_chase_speed_lock = False
        self._rule_aware_lead_active = False
        self._pn_engagement_decided = False
        self._pn_engagement_active = False
        self._terminal_range_history: deque = deque()
        self._initial_direct_chase_until = float('-inf')
        self._far_noisy_acquisition = False
        self._track_range_history: deque = deque()
        self._stall_recovery_until = float('-inf')
        self._high_closing_pixel_lead_active = False
        self._high_closing_pixel_mode_decided = False
        self._motion_feature_history: deque = deque(maxlen=250)
        self._track_confirmed_at: Optional[float] = None
        self._track_start_position: Optional[np.ndarray] = None
        self._previous_feature_closing_speed: Optional[float] = None
        self._minimum_estimated_distance = float('inf')
        self._prior_pass_count = 0
        self._extended_terminal_context: Optional[dict] = None

    def reset(self) -> None:
        self.phase = Phase.SEARCH_TRANSIT
        self.frame_index = 0
        self.clue_filter.reset()
        self.tracker.reset()
        self.terminal_rules.reset()
        self.learned_policies.reset()
        self.generalized_terminal_policy.reset()
        self.previous_velocity_command = np.zeros(3, dtype=np.float64)
        self._last_candidate = None
        self._tilt_history.clear()
        self._smoothed_agl = None
        self._terminal_active = False
        self._terminal_accel_scale = 1.0
        self._terminal_dynamics_enabled = False
        self._remembered_target_z = None
        self._first_target_z = None
        self._target_z_history.clear()
        self._target_z_observation_times.clear()
        self._target_z_reanchor_history.clear()
        self._target_z_reanchors = 0
        self._last_detection_horizontal_angle = 0.0
        self._previous_target_pixel = None
        self._previous_target_pixel_frame = None
        self._reacquire_started_at = None
        self._reacquire_started_far = False
        self._acquisition_vertical_boost_until = float('-inf')
        self._large_vertical_acquisition = False
        self._short_visible_chase = False
        self._terminal_recenter_active = False
        self._terminal_recenter_started_at = None
        self._terminal_recenter_center_frames = 0
        self._previous_camera_error = None
        self._previous_horizontal_angle = None
        self._previous_vertical_angle = None
        self._previous_detection_t = None
        self._terminal_horizontal_rate = 0.0
        self._terminal_vertical_rate = 0.0
        self._terminal_decision = TerminalDecision('T00_INACTIVE', TerminalProfile.NORMAL_PN)
        self._terminal_context = None
        self._reacquire_context = None
        self._learned_reacquire_rule_id = 'L00_DEFAULT_REACQUIRE'
        self._terminal_recenter_events = 0
        self._terminal_cpa_time = 0.0
        self._terminal_cpa_miss = float('inf')
        self._terminal_closing_speed = 0.0
        self._parallel_chase_speed_lock = False
        self._rule_aware_lead_active = False
        self._pn_engagement_decided = False
        self._pn_engagement_active = False
        self._terminal_range_history.clear()
        self._initial_direct_chase_until = float('-inf')
        self._far_noisy_acquisition = False
        self._track_range_history.clear()
        self._stall_recovery_until = float('-inf')
        self._high_closing_pixel_lead_active = False
        self._high_closing_pixel_mode_decided = False
        self._motion_feature_history.clear()
        self._track_confirmed_at = None
        self._track_start_position = None
        self._previous_feature_closing_speed = None
        self._minimum_estimated_distance = float('inf')
        self._prior_pass_count = 0
        self._extended_terminal_context = None

    def act(self, observation) -> np.ndarray:
        state = np.asarray(observation['state'], dtype=np.float64)
        depth = observation.get('depth')
        t = self.frame_index * DT
        position = state[0:3]
        own_xy = position[0:2]
        rpy = state[3:6]
        agl = local_agl(state)
        clue_offset_xy = state[138:140]
        if self._smoothed_agl is None:
            self._smoothed_agl = agl
        else:
            self._smoothed_agl += AGL_SMOOTH_ALPHA * (agl - self._smoothed_agl)
        raw_clue_xy, filtered_clue_xy = self.clue_filter.update(own_xy, clue_offset_xy)
        predicted_clue_xy = self.clue_filter.predict(CLUE_PREDICTION_HORIZON_S)
        predicted_track_before_update = self.tracker.predicted_position(t)
        near_track_hint = predicted_track_before_update if predicted_track_before_update is not None and self.tracker.missing_time(t) <= DEPTH_TARGET_TRACK_ROI_MAX_MEMORY_S else None
        raw_depth_detection = self.detector.detect(drone_pos=position, drone_rpy=rpy, depth_image=depth, clue_xy_est=predicted_clue_xy, track_position_est=near_track_hint, target_z_est=self._remembered_target_z)
        detection_authorized = raw_depth_detection is not None
        rejected_near_track_jump = False
        if detection_authorized and raw_depth_detection is not None and (predicted_track_before_update is not None) and (self.tracker.missing_time(t) <= TRACK_NEAR_ASSOCIATION_MEMORY_S) and (float(np.linalg.norm(np.asarray(predicted_track_before_update, dtype=np.float64) - position)) <= 3.0) and (float(np.linalg.norm(np.asarray(raw_depth_detection.position, dtype=np.float64) - np.asarray(predicted_track_before_update, dtype=np.float64))) > TRACK_NEAR_ASSOCIATION_GATE_M):
            detection_authorized = raw_depth_detection is not None
            rejected_near_track_jump = True
        if detection_authorized and raw_depth_detection is not None:
            target_z = float(raw_depth_detection.position[2])
            if self._first_target_z is not None:
                is_close_out_of_band = bool(raw_depth_detection.range_m <= TARGET_Z_REANCHOR_RANGE_M and abs(target_z - self._first_target_z) > TARGET_Z_MEMORY_BAND_M)
                if is_close_out_of_band:
                    self._target_z_reanchor_history.append(target_z)
                    if len(self._target_z_reanchor_history) >= TARGET_Z_REANCHOR_HITS and float(np.ptp(self._target_z_reanchor_history)) <= TARGET_Z_REANCHOR_SPREAD_M:
                        reanchored_z = float(np.median(self._target_z_reanchor_history))
                        self._first_target_z = reanchored_z
                        self._remembered_target_z = reanchored_z
                        self._target_z_history.clear()
                        self._target_z_reanchor_history.clear()
                        self._target_z_reanchors += 1
                else:
                    self._target_z_reanchor_history.clear()
                target_z = float(np.clip(target_z, self._first_target_z - TARGET_Z_MEMORY_BAND_M, self._first_target_z + TARGET_Z_MEMORY_BAND_M))
            if self._first_target_z is None:
                self._update_tentative_target_height(t, target_z)
            else:
                self._target_z_history.append(target_z)
                self._remembered_target_z = float(np.median(self._target_z_history))
        depth_detection = raw_depth_detection
        detection = depth_detection if detection_authorized else None
        self.tracker.update(t, detection)
        first_detection = bool(detection is not None and self._first_target_z is None and (self.tracker.recent_hits(TARGET_CONFIRM_WINDOW) >= TARGET_CONFIRM_HITS))
        if first_detection:
            confirmed_z = float(np.median(self._target_z_history))
            self._first_target_z = confirmed_z
            self._remembered_target_z = confirmed_z
        guidance_detection = detection
        detection_pixel = None
        detection_pixel_delta = None
        if detection is not None:
            previous_pixel = self._previous_target_pixel if self._previous_target_pixel_frame == self.frame_index - 1 else None
            if detection.pixel_x is not None and detection.pixel_y is not None:
                detection_pixel = [float(detection.pixel_x), float(detection.pixel_y)]
                if previous_pixel is not None:
                    detection_pixel_delta = [float(detection.pixel_x) - float(previous_pixel[0]), float(detection.pixel_y) - float(previous_pixel[1])]
            guidance_detection = extrapolate_detection_pixel(detection, previous_pixel, position, rpy, alpha_override=HIGH_CLOSING_PIXEL_ALPHA if self._high_closing_pixel_lead_active else None)
            if detection.pixel_x is not None and detection.pixel_y is not None:
                self._previous_target_pixel = (float(detection.pixel_x), float(detection.pixel_y))
                self._previous_target_pixel_frame = self.frame_index
            else:
                self._previous_target_pixel = None
                self._previous_target_pixel_frame = None
            self._last_candidate = detection
            self._last_detection_horizontal_angle = float(detection.horizontal_angle)
            if first_detection:
                self._far_noisy_acquisition = detection.range_m >= INITIAL_DIRECT_CHASE_MIN_RANGE_M
                self._initial_direct_chase_until = t + INITIAL_DIRECT_CHASE_S if self._far_noisy_acquisition else float('-inf')
                initial_vertical_gap = float(detection.position[2] - position[2])
                self.tracker.prefer_weighted_fit = True
                self._acquisition_vertical_boost_until = t + ACQUISITION_VERTICAL_BOOST_S
                self._large_vertical_acquisition = abs(float(detection.position[2]) - float(position[2])) >= LARGE_ACQUISITION_VERTICAL_GAP_M
                self._short_visible_chase = detection.range_m < SHORT_LEAD_ACQUISITION_RANGE_M and (not self._large_vertical_acquisition)
        d_raw, raw_bearing, beta, scan_half_width = search_sector(own_xy, raw_clue_xy, R_EFFECTIVE, math.radians(USABLE_CAMERA_HALF_FOV_DEG))
        predicted_clue_dist = float(np.linalg.norm(predicted_clue_xy - own_xy))
        local_needed = predicted_clue_dist <= LOCAL_SEARCH_ENTER_M
        self._update_track_stall(t, detection)
        previous_phase = self.phase
        self.phase = self._next_phase(t, detection, local_needed, position)
        if not self._high_closing_pixel_mode_decided and previous_phase != Phase.TRACK_AND_INTERCEPT and (self.phase == Phase.TRACK_AND_INTERCEPT):
            self._high_closing_pixel_lead_active = clean_centered_predictive_acquisition(detection)
            self._high_closing_pixel_mode_decided = True
        target_distance: Optional[float] = None
        target_pos_estimate: Optional[np.ndarray] = None
        self._extended_terminal_context = None
        if self.phase == Phase.SEARCH_TRANSIT:
            v_req, yaw_req = self._search_transit_command(own_xy, position[2], self._smoothed_agl, predicted_clue_xy)
        elif self.phase == Phase.SEARCH_LOCAL:
            v_req, yaw_req = self._search_local_command(own_xy, position[2], self._smoothed_agl, predicted_clue_xy)
            if DEV_LOCAL_SEARCH_YAW_SCAN_DEG is not None:
                scan_period = max(0.25, DEV_LOCAL_SEARCH_YAW_SCAN_PERIOD_S)
                yaw_req = wrap_angle(yaw_req + math.radians(DEV_LOCAL_SEARCH_YAW_SCAN_DEG) * math.sin(2.0 * math.pi * t / scan_period))
        elif self.phase == Phase.TENTATIVE_DETECTION:
            v_req, yaw_req = self._tentative_command(own_xy, position[2], self._smoothed_agl, detection or self._last_candidate)
        elif self.phase == Phase.TRACK_AND_INTERCEPT:
            target_pos_estimate = self.tracker.predicted_position(t)
            guidance_target_velocity = np.asarray(self.tracker.last_velocity, dtype=np.float64).copy()
            guidance_target_acceleration = np.asarray(self.tracker.last_acceleration, dtype=np.float64).copy()
            command_target = target_pos_estimate
            command_target = np.asarray(command_target, dtype=np.float64).copy()
            target_distance = float(np.linalg.norm(command_target - position))
            if target_distance <= TERMINAL_Z_LOCK_RANGE_M and self._remembered_target_z is not None:
                command_target[2] = self._remembered_target_z
                guidance_target_velocity[2] = 0.0
                guidance_target_acceleration[2] = 0.0
                target_distance = float(np.linalg.norm(command_target - position))
            self._update_extended_terminal_features(t=t, detection_present=guidance_detection is not None, detection=detection, detection_pixel_delta=detection_pixel_delta, drone_pos=position, drone_velocity=state[6:9], target_pos=command_target, target_velocity=guidance_target_velocity, target_acceleration=guidance_target_acceleration, drone_rpy=rpy, drone_angular_velocity=state[9:12])
            self._update_terminal_recenter(t, guidance_detection, position, state[6:9], command_target, guidance_target_velocity)
            self._select_terminal_rule(t, guidance_detection, position, state[6:9], command_target, guidance_target_velocity, guidance_target_acceleration, rpy)
            if t <= self._initial_direct_chase_until:
                v_req = direct_pursuit_command(position, command_target)
                yaw_req = _heading_to(command_target[:2] - position[:2])
            else:
                profile_target = command_target
                if self._terminal_decision.profile == TerminalProfile.CAMERA_RAY and detection is not None:
                    profile_target = np.asarray(detection.position, dtype=np.float64).copy()
                    if self._remembered_target_z is not None:
                        profile_target[2] = self._remembered_target_z
                elif self._terminal_decision.profile == TerminalProfile.PREDICTED_CAMERA_RAY and guidance_detection is not None:
                    profile_target = np.asarray(guidance_detection.position, dtype=np.float64).copy()
                    if self._remembered_target_z is not None:
                        profile_target[2] = self._remembered_target_z
                v_req, yaw_req = self._intercept_command(position, state[6:9], profile_target, guidance_target_velocity, guidance_target_acceleration, terminal_recenter=self._terminal_recenter_active, terminal_profile=self._terminal_decision.profile)
            if self._terminal_decision.profile == TerminalProfile.CAMERA_RAY and detection is not None:
                yaw_req = wrap_angle(float(rpy[2]) - float(detection.horizontal_angle))
            elif guidance_detection is not None:
                yaw_req = wrap_angle(float(rpy[2]) - float(guidance_detection.horizontal_angle))
            elif self._terminal_recenter_active:
                yaw_req = recenter_lost_yaw_target(float(rpy[2]), self._last_detection_horizontal_angle)
        elif self.phase == Phase.REACQUIRE:
            missing_time = self.tracker.missing_time(t)
            prediction_horizon = min(missing_time, REACQUIRE_FAR_PREDICTION_CAP_S if self._reacquire_started_far else REACQUIRE_PREDICTION_CAP_S)
            target_pos_estimate = self.tracker.last_position + self.tracker.last_velocity * prediction_horizon
            if self._remembered_target_z is not None:
                target_pos_estimate = np.asarray(target_pos_estimate, dtype=np.float64).copy()
                target_pos_estimate[2] = self._remembered_target_z
            v_req, yaw_req = self._reacquire_command(t, position, state[6:9], target_pos_estimate)
            target_distance = float(np.linalg.norm(target_pos_estimate - position))
        else:
            v_req, yaw_req = (np.zeros(3, dtype=np.float64), float(rpy[2]))
        if self.phase != Phase.TRACK_AND_INTERCEPT:
            self._terminal_recenter_active = False
            self._terminal_recenter_started_at = None
            self._terminal_recenter_center_frames = 0
        v_req = self._apply_ground_guard(position[2], v_req)
        if agl < TAKEOFF_CLEARANCE_AGL_M:
            v_req = np.asarray(v_req, dtype=np.float64).copy()
            clearance_fraction = float(np.clip((TAKEOFF_CLEARANCE_AGL_M - agl) / max(TAKEOFF_CLEARANCE_AGL_M, 1e-06), 0.0, 1.0))
            v_req[2] = max(float(v_req[2]), TAKEOFF_CLEARANCE_CLIMB_MPS * clearance_fraction)
        tilt_deg = math.degrees(max(abs(float(rpy[0])), abs(float(rpy[1]))))
        self._tilt_history.append((t, tilt_deg))
        while self._tilt_history and t - self._tilt_history[0][0] > TILT_CLIMB_WINDOW_S:
            self._tilt_history.popleft()
        window_full = bool(self._tilt_history) and t - self._tilt_history[0][0] >= TILT_CLIMB_WINDOW_S * 0.9
        climbed_deg = tilt_deg - self._tilt_history[0][1] if self._tilt_history else 0.0
        sustained_climb = window_full and climbed_deg >= TILT_CLIMB_THRESHOLD_DEG and (tilt_deg >= TILT_RUNAWAY_MIN_DEG)
        if tilt_deg >= TILT_HARD_CAP_DEG:
            brake = TILT_BRAKE_MIN_FRACTION
        elif sustained_climb:
            brake = tilt_brake_fraction(math.radians(tilt_deg), TILT_RUNAWAY_MIN_DEG, TILT_HARD_CAP_DEG, TILT_BRAKE_MIN_FRACTION)
        else:
            brake = 1.0
        v_req = v_req * brake
        if self._rule_aware_lead_active:
            max_accel = CHASE_CLOSE_MAX_ACCEL if self._terminal_decision.rule_lead_accel_mps2 is None else float(self._terminal_decision.rule_lead_accel_mps2)
        elif self._terminal_recenter_active or self._parallel_chase_speed_lock:
            max_accel = FINAL_CHASE_MAX_ACCEL
        elif self._large_vertical_acquisition and target_distance is not None and (target_distance < FINAL_CHASE_RANGE_M):
            max_accel = FINAL_CHASE_MAX_ACCEL
        elif target_distance is not None and target_distance < CLOSE_RANGE_M:
            max_accel = CLOSE_MAX_ACCEL if self._terminal_recenter_active else CHASE_CLOSE_MAX_ACCEL
        else:
            max_accel = NORMAL_MAX_ACCEL
        previous_command = self.previous_velocity_command.copy()
        v_cmd = slew_velocity(previous_command, v_req, DT, max_accel)
        if self._terminal_recenter_active:
            vertical_delta = float(np.clip(float(v_req[2]) - float(previous_command[2]), -FINAL_CHASE_MAX_ACCEL * DT, FINAL_CHASE_MAX_ACCEL * DT))
            v_cmd[2] = float(previous_command[2]) + vertical_delta
        if t <= self._acquisition_vertical_boost_until:
            vertical_delta = float(np.clip(float(v_req[2]) - float(previous_command[2]), -ACQUISITION_VERTICAL_MAX_ACCEL * DT, ACQUISITION_VERTICAL_MAX_ACCEL * DT))
            v_cmd[2] = float(previous_command[2]) + vertical_delta
        self.previous_velocity_command = v_cmd
        action = encode_action(v_cmd, yaw_req)
        self.frame_index += 1
        return action

    def _update_extended_terminal_features(self, *, t: float, detection_present: bool, detection: Optional[Detection], detection_pixel_delta: Optional[list[float]], drone_pos: np.ndarray, drone_velocity: np.ndarray, target_pos: np.ndarray, target_velocity: np.ndarray, target_acceleration: np.ndarray, drone_rpy: np.ndarray, drone_angular_velocity: np.ndarray) -> None:
        drone_pos = np.asarray(drone_pos, dtype=np.float64)
        drone_velocity = np.asarray(drone_velocity, dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_velocity = np.asarray(target_velocity, dtype=np.float64)
        target_acceleration = np.asarray(target_acceleration, dtype=np.float64)
        drone_rpy = np.asarray(drone_rpy, dtype=np.float64)
        drone_angular_velocity = np.asarray(drone_angular_velocity, dtype=np.float64)
        relative = target_pos - drone_pos
        distance = float(np.linalg.norm(relative))
        horizontal_relative = relative.copy()
        horizontal_relative[2] = 0.0
        horizontal_distance = float(np.linalg.norm(horizontal_relative))
        if horizontal_distance > 1e-06:
            radial = horizontal_relative / horizontal_distance
        else:
            radial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        lateral = np.array([-radial[1], radial[0], 0.0], dtype=np.float64)
        line_of_sight = unit(relative)
        closing_speed = float(np.dot(drone_velocity - target_velocity, line_of_sight))
        target_speed = float(np.linalg.norm(target_velocity))
        target_horizontal_speed = float(np.linalg.norm(target_velocity[:2]))
        target_heading = math.atan2(float(target_velocity[1]), float(target_velocity[0])) if target_horizontal_speed > 1e-06 else 0.0
        target_lateral_alignment = float(np.dot(target_velocity, lateral) / target_speed) if target_speed > 1e-06 else 0.0
        target_away_alignment = -float(np.dot(target_velocity, radial) / target_speed) if target_speed > 1e-06 else 0.0
        if self._track_confirmed_at is None:
            self._track_confirmed_at = float(t)
            self._track_start_position = target_pos.copy()
        self._minimum_estimated_distance = min(self._minimum_estimated_distance, distance)
        if self._previous_feature_closing_speed is not None:
            if self._previous_feature_closing_speed > 0.25 and closing_speed < -0.25 and (distance <= 3.0):
                self._prior_pass_count += 1
        self._previous_feature_closing_speed = closing_speed
        sample = {'t': float(t), 'target_heading': target_heading, 'target_lateral_alignment': target_lateral_alignment, 'target_speed': target_speed, 'closing_speed': closing_speed, 'range_m': distance, 'horizontal_error': float(detection.horizontal_angle) if detection is not None else 0.0, 'vertical_error': float(detection.vertical_angle) if detection is not None else 0.0, 'target_acceleration_lateral': float(np.dot(target_acceleration, lateral)), 'detection': 1.0 if detection_present else 0.0}
        self._motion_feature_history.append(sample)

        def past_sample(window: float) -> dict:
            desired_t = t - window
            return min(self._motion_feature_history, key=lambda item: abs(float(item['t']) - desired_t))

        def scalar_change(field: str, window: float) -> float:
            return float(sample[field]) - float(past_sample(window)[field])

        def heading_change(window: float) -> float:
            return wrap_angle(target_heading - float(past_sample(window)['target_heading']))

        def detection_duty(window: float) -> float:
            observations = [item for item in self._motion_feature_history if float(item['t']) >= t - window]
            if not observations:
                return float(detection_present)
            return float(sum((float(item['detection']) for item in observations)) / len(observations))
        recent = [item for item in self._motion_feature_history if float(item['t']) >= t - 4.0]
        signs = [1 if float(item['target_lateral_alignment']) >= 0.0 else -1 for item in recent if abs(float(item['target_lateral_alignment'])) >= 0.04]
        sign_changes = sum((a != b for a, b in zip(signs, signs[1:])))
        history_span = float(recent[-1]['t']) - float(recent[0]['t']) if len(recent) >= 2 else 0.0
        jink_frequency = 0.5 * sign_changes / history_span if history_span > 0.25 else 0.0
        innovation = np.asarray(self.tracker.last_innovation, dtype=np.float64)
        command = np.asarray(self.previous_velocity_command, dtype=np.float64)
        command_speed = float(np.linalg.norm(command))
        drone_speed = float(np.linalg.norm(drone_velocity))
        command_alignment = float(np.dot(command, drone_velocity) / (command_speed * drone_speed)) if command_speed > 1e-06 and drone_speed > 1e-06 else 0.0
        relative_velocity = target_velocity - drone_velocity
        relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        signed_cpa_time = float(np.clip(-float(np.dot(relative, relative_velocity)) / relative_speed_sq, 0.0, 0.5)) if relative_speed_sq > 1e-08 else 0.0
        cpa_offset = relative + relative_velocity * signed_cpa_time
        track_displacement = float(np.linalg.norm(target_pos - self._track_start_position)) if self._track_start_position is not None else 0.0
        self._extended_terminal_context = {'time_s': float(t), 'horizontal_distance': horizontal_distance, 'target_velocity_radial': float(np.dot(target_velocity, radial)), 'target_velocity_lateral': float(np.dot(target_velocity, lateral)), 'target_velocity_vertical': float(target_velocity[2]), 'drone_velocity_radial': float(np.dot(drone_velocity, radial)), 'drone_velocity_lateral': float(np.dot(drone_velocity, lateral)), 'drone_velocity_vertical': float(drone_velocity[2]), 'target_acceleration_radial': float(np.dot(target_acceleration, radial)), 'target_acceleration_lateral': float(np.dot(target_acceleration, lateral)), 'target_acceleration_vertical': float(target_acceleration[2]), 'target_acceleration_magnitude': float(np.linalg.norm(target_acceleration)), 'target_lateral_alignment': target_lateral_alignment, 'target_away_alignment': target_away_alignment, 'target_lateral_rate_0p2': scalar_change('target_lateral_alignment', 0.2) / 0.2, 'target_lateral_rate_0p5': scalar_change('target_lateral_alignment', 0.5) / 0.5, 'target_heading_change_0p2': heading_change(0.2), 'target_heading_change_0p5': heading_change(0.5), 'target_speed_change_0p2': scalar_change('target_speed', 0.2), 'target_speed_change_0p5': scalar_change('target_speed', 0.5), 'closing_speed_change_0p2': scalar_change('closing_speed', 0.2), 'closing_speed_change_0p5': scalar_change('closing_speed', 0.5), 'detection_duty_0p2': detection_duty(0.2), 'detection_duty_0p5': detection_duty(0.5), 'tracker_innovation_norm': float(np.linalg.norm(innovation)), 'tracker_innovation_radial': float(np.dot(innovation, radial)), 'tracker_innovation_lateral': float(np.dot(innovation, lateral)), 'tracker_innovation_vertical': float(innovation[2]), 'tracker_measurement_interval': float(self.tracker.last_measurement_interval), 'track_age': max(0.0, float(t) - float(self._track_confirmed_at)), 'target_displacement_from_track_start': track_displacement, 'drone_command_speed': command_speed, 'drone_speed_error': command_speed - drone_speed, 'command_velocity_alignment': command_alignment, 'roll': float(drone_rpy[0]), 'pitch': float(drone_rpy[1]), 'roll_rate': float(drone_angular_velocity[0]), 'pitch_rate': float(drone_angular_velocity[1]), 'yaw_rate': float(drone_angular_velocity[2]), 'jink_frequency_estimate_hz': float(jink_frequency), 'jink_sign_changes_4s': int(sign_changes), 'prior_pass_count': int(self._prior_pass_count), 'minimum_estimated_distance': float(self._minimum_estimated_distance), 'relative_velocity_radial': float(np.dot(relative_velocity, radial)), 'relative_velocity_lateral': float(np.dot(relative_velocity, lateral)), 'relative_velocity_vertical': float(relative_velocity[2]), 'signed_cpa_lateral_m': float(np.dot(cpa_offset, lateral)), 'signed_cpa_vertical_m': float(cpa_offset[2]), 'range_change_0p2': scalar_change('range_m', 0.2), 'range_change_0p5': scalar_change('range_m', 0.5), 'horizontal_error_change_0p2': scalar_change('horizontal_error', 0.2), 'horizontal_error_change_0p5': scalar_change('horizontal_error', 0.5), 'vertical_error_change_0p2': scalar_change('vertical_error', 0.2), 'vertical_error_change_0p5': scalar_change('vertical_error', 0.5), 'target_acceleration_lateral_change_0p2': scalar_change('target_acceleration_lateral', 0.2), 'target_acceleration_lateral_change_0p5': scalar_change('target_acceleration_lateral', 0.5), 'command_velocity_radial': float(np.dot(command, radial)), 'command_velocity_lateral': float(np.dot(command, lateral)), 'command_velocity_vertical': float(command[2])}
        if detection is not None:
            image_width = max(1.0, float(detection.image_width or 1.0))
            image_height = max(1.0, float(detection.image_height or 1.0))
            bbox_width = float(detection.component_bbox_width_px or 0.0)
            bbox_height = float(detection.component_bbox_height_px or 0.0)
            component_pixels = float(detection.component_pixels or 0.0)
            pooled_bbox_area = max(1.0, 0.25 * bbox_width * bbox_height)
            pixel_delta = np.asarray(detection_pixel_delta or [0.0, 0.0], dtype=np.float64)
            self._extended_terminal_context.update({'detection_range_m': float(detection.range_m), 'detection_component_pixels': component_pixels, 'detection_bbox_width_frac': bbox_width / image_width, 'detection_bbox_height_frac': bbox_height / image_height, 'detection_component_fill': component_pixels / pooled_bbox_area, 'detection_depth_span_m': float(detection.component_depth_span_m or 0.0), 'detection_clue_error_m': float(detection.clue_error_m or 0.0), 'pixel_delta_x_frac': float(pixel_delta[0]) / image_width, 'pixel_delta_y_frac': float(pixel_delta[1]) / image_height, 'pixel_delta_norm_frac': float(np.linalg.norm(pixel_delta)) / math.hypot(image_width, image_height)})

    def _select_terminal_rule(self, t: float, detection: Optional[Detection], drone_pos: np.ndarray, drone_vel: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray, target_accel: np.ndarray, drone_rpy: np.ndarray) -> None:
        relative_position = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
        distance = float(np.linalg.norm(relative_position))
        selection_range = TERMINAL_RECENTER_PREDICTIVE_RANGE_M
        selection_range = max(selection_range, self.learned_policies.max_visible_range)
        if DEV_FORCE_TERMINAL_PROFILE is not None:
            selection_range = max(selection_range, DEV_FORCE_TERMINAL_RANGE_M)
        if distance > selection_range:
            self.terminal_rules.reset()
            self.learned_policies.reset_visible()
            self._terminal_decision = TerminalDecision('T00_INACTIVE', TerminalProfile.NORMAL_PN)
            self._terminal_context = None
            return
        line_of_sight = unit(relative_position)
        drone_velocity = np.asarray(drone_vel, dtype=np.float64)
        target_velocity = np.asarray(target_vel, dtype=np.float64)
        target_acceleration = np.asarray(target_accel, dtype=np.float64)
        relative_velocity = target_velocity - drone_velocity
        radial_relative_speed = float(np.dot(relative_velocity, line_of_sight))
        tangent_relative_velocity = relative_velocity - radial_relative_speed * line_of_sight
        los_rate = float(np.linalg.norm(tangent_relative_velocity) / max(distance, 1e-06))
        drone_speed = float(np.linalg.norm(drone_velocity))
        target_speed = float(np.linalg.norm(target_velocity))
        target_direction = target_velocity / target_speed if target_speed > 1e-06 else np.zeros(3, dtype=np.float64)
        longitudinal_deceleration = -float(np.dot(target_velocity, target_acceleration)) / target_speed if target_speed > 1e-06 else 0.0
        tangent_acceleration = target_acceleration - float(np.dot(target_acceleration, target_direction)) * target_direction
        target_turn_rate = float(np.linalg.norm(tangent_acceleration) / max(target_speed, 0.5))

        def alignment(a: np.ndarray, b: np.ndarray) -> float:
            a_norm = float(np.linalg.norm(a))
            b_norm = float(np.linalg.norm(b))
            if a_norm <= 1e-06 or b_norm <= 1e-06:
                return -1.0
            return float(np.dot(a, b) / (a_norm * b_norm))
        horizontal_error = float(detection.horizontal_angle) if detection is not None else float(self._last_detection_horizontal_angle)
        vertical_error = float(detection.vertical_angle) if detection is not None else 0.0
        context = TerminalContext(distance=distance, closing_speed=self._terminal_closing_speed, cpa_time=self._terminal_cpa_time, cpa_miss=self._terminal_cpa_miss, horizontal_error=horizontal_error, vertical_error=vertical_error, horizontal_error_rate=self._terminal_horizontal_rate, vertical_error_rate=self._terminal_vertical_rate, los_rate=los_rate, drone_speed=drone_speed, target_speed=target_speed, target_longitudinal_deceleration=longitudinal_deceleration, target_turn_rate=target_turn_rate, drone_target_alignment=alignment(drone_velocity, target_velocity), drone_los_alignment=alignment(drone_velocity, line_of_sight), target_los_alignment=alignment(target_velocity, line_of_sight), z_error=float(relative_position[2]), tilt=max(abs(float(drone_rpy[0])), abs(float(drone_rpy[1]))), detection_present=detection is not None, missing_time=self.tracker.missing_time(t), estimator_confidence=min(1.0, self.tracker.recent_hits(TARGET_CONFIRM_WINDOW) / max(1.0, float(TARGET_CONFIRM_WINDOW))))
        self._terminal_context = context
        previous_recenter = self._terminal_recenter_active
        if distance > TERMINAL_RECENTER_PREDICTIVE_RANGE_M:
            self.terminal_rules.reset()
            self._terminal_decision = TerminalDecision('T00_INACTIVE', TerminalProfile.NORMAL_PN)
        else:
            self._terminal_decision = self.terminal_rules.select(t, context)
        learned_decision = self.learned_policies.select_visible(context, self._extended_terminal_context)
        if learned_decision is not None:
            self._terminal_decision = learned_decision
        generalized_decision = self.generalized_terminal_policy.select(context, self._extended_terminal_context)
        if generalized_decision is not None:
            self._terminal_decision = generalized_decision
        if DEV_FORCE_TERMINAL_PROFILE is not None and context.distance <= DEV_FORCE_TERMINAL_RANGE_M:
            self._terminal_decision = TerminalDecision('D00_FORCED_PROFILE', DEV_FORCE_TERMINAL_PROFILE, rule_lead_accel_mps2=30.0 if DEV_FORCE_TERMINAL_PROFILE in (TerminalProfile.HIGH_AUTHORITY_PN, TerminalProfile.STAGED_HIGH_AUTHORITY_PN) else None)
        wants_recenter = self._terminal_decision.profile in (TerminalProfile.PN_TANGENT_BRAKE, TerminalProfile.STOP_AND_YAW)
        if self._terminal_decision.profile == TerminalProfile.MEMORY_REACQUIRE:
            wants_recenter = previous_recenter
        if wants_recenter and (not previous_recenter):
            self._terminal_recenter_active = True
            self._terminal_recenter_started_at = t
            self._terminal_recenter_center_frames = 0
            self._terminal_recenter_events += 1
        elif not wants_recenter:
            self._terminal_recenter_active = False
            self._terminal_recenter_started_at = None
            self._terminal_recenter_center_frames = 0

    def _update_terminal_recenter(self, t: float, detection: Optional[Detection], drone_pos: np.ndarray, drone_vel: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray) -> None:
        if detection is None:
            self._terminal_horizontal_rate = 0.0
            self._terminal_vertical_rate = 0.0
            still_braking = self._terminal_recenter_active and self._terminal_recenter_started_at is not None and (t - self._terminal_recenter_started_at < TERMINAL_RECENTER_TIMEOUT_S) and (self.tracker.missing_time(t) <= self._terminal_recenter_loss_hold_s())
            if still_braking:
                self._terminal_recenter_center_frames = 0
                return
            self._terminal_recenter_active = False
            self._terminal_recenter_started_at = None
            self._terminal_recenter_center_frames = 0
            self._previous_camera_error = None
            return
        horizontal_error = abs(float(detection.horizontal_angle))
        vertical_error = abs(float(detection.vertical_angle))
        if self._previous_detection_t is not None and self._previous_horizontal_angle is not None and (self._previous_vertical_angle is not None):
            detection_dt = max(1e-06, t - self._previous_detection_t)
            self._terminal_horizontal_rate = (float(detection.horizontal_angle) - self._previous_horizontal_angle) / detection_dt
            self._terminal_vertical_rate = (float(detection.vertical_angle) - self._previous_vertical_angle) / detection_dt
        else:
            self._terminal_horizontal_rate = 0.0
            self._terminal_vertical_rate = 0.0
        self._previous_horizontal_angle = float(detection.horizontal_angle)
        self._previous_vertical_angle = float(detection.vertical_angle)
        self._previous_detection_t = t
        camera_error = max(horizontal_error, vertical_error)
        enter_angle = math.radians(TERMINAL_RECENTER_ENTER_ANGLE_DEG)
        exit_angle = math.radians(TERMINAL_RECENTER_EXIT_ANGLE_DEG)
        distance = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)))
        worsening = self._previous_camera_error is not None and camera_error - self._previous_camera_error >= math.radians(TERMINAL_RECENTER_ERROR_GROWTH_DEG)
        relative_position = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
        time_to_cpa, miss_distance, closing_speed = closest_approach_metrics(relative_position, drone_vel, target_vel, TERMINAL_RECENTER_CPA_HORIZON_S)
        self._terminal_cpa_time = time_to_cpa
        self._terminal_cpa_miss = miss_distance
        self._terminal_closing_speed = closing_speed
        outside_center_box = distance < TERMINAL_RECENTER_RANGE_M and (horizontal_error > enter_angle or vertical_error > enter_angle)
        predicted_to_leave_center = distance < TERMINAL_RECENTER_PREDICTIVE_RANGE_M and (not self._high_closing_pixel_lead_active) and (camera_error > math.radians(TERMINAL_RECENTER_PREDICTIVE_ANGLE_DEG)) and worsening and (closing_speed >= TERMINAL_RECENTER_MIN_ENTRY_CLOSING_MPS)
        if self._terminal_recenter_active:
            elapsed = t - float(self._terminal_recenter_started_at)
            centered = horizontal_error <= exit_angle and vertical_error <= exit_angle and (miss_distance <= TERMINAL_RECENTER_EXIT_MISS_M)
            self._terminal_recenter_center_frames = self._terminal_recenter_center_frames + 1 if centered else 0
            if distance > TERMINAL_RECENTER_EXIT_RANGE_M or elapsed >= TERMINAL_RECENTER_TIMEOUT_S or self._terminal_recenter_center_frames >= TERMINAL_RECENTER_CENTER_FRAMES:
                self._terminal_recenter_active = False
                self._terminal_recenter_started_at = None
                self._terminal_recenter_center_frames = 0
        elif outside_center_box or predicted_to_leave_center:
            definite_miss = miss_distance >= TERMINAL_RECENTER_MISS_DISTANCE_M
            already_capturing = miss_distance <= TERMINAL_CAPTURE_CORRIDOR_M and time_to_cpa > 0.0 and (closing_speed >= TERMINAL_RECENTER_MIN_ENTRY_CLOSING_MPS)
            if definite_miss and (not already_capturing):
                self._terminal_recenter_active = True
                self._terminal_recenter_started_at = t
                self._terminal_recenter_center_frames = 0
                self._terminal_recenter_events += 1
        self._previous_camera_error = camera_error

    def _terminal_recenter_loss_hold_s(self) -> float:
        velocity = np.asarray(self.tracker.last_velocity, dtype=np.float64)
        acceleration = np.asarray(self.tracker.last_acceleration, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        longitudinal_deceleration = -float(np.dot(velocity, acceleration)) / speed if speed > 1e-06 else 0.0
        if longitudinal_deceleration >= TERMINAL_RECENTER_STOP_DECEL_MPS2:
            return TERMINAL_RECENTER_STOP_LOST_HOLD_S
        return TERMINAL_RECENTER_LOST_HOLD_S

    def _update_tentative_target_height(self, t: float, target_z: float) -> bool:
        target_z = float(target_z)
        if not TARGET_Z_TENTATIVE_MIN_M <= target_z <= TARGET_Z_TENTATIVE_MAX_M:
            return False
        self._target_z_history.append(target_z)
        self._target_z_observation_times.append(float(t))
        while self._target_z_observation_times and t - self._target_z_observation_times[0] > TARGET_Z_TENTATIVE_WINDOW_S:
            self._target_z_observation_times.popleft()
            self._target_z_history.popleft()
        if len(self._target_z_history) < TARGET_Z_TENTATIVE_HITS or float(np.ptp(self._target_z_history)) > TARGET_Z_TENTATIVE_SPREAD_M:
            return False
        self._remembered_target_z = float(np.median(self._target_z_history))
        return True

    def _apply_ground_guard(self, current_z: float, requested_velocity: np.ndarray) -> np.ndarray:
        command = np.asarray(requested_velocity, dtype=np.float64).copy()
        if self._first_target_z is None:
            return command
        safety_floor = max(TARGET_Z_GROUND_FLOOR_M, self._first_target_z - TARGET_Z_EMERGENCY_DROP_M)
        if float(current_z) >= safety_floor:
            return command
        command[2] = max(float(command[2]), vertical_rate_to_target_z(current_z, self._first_target_z))
        return command

    def _update_track_stall(self, t: float, detection: Optional[Detection]) -> None:
        if detection is None:
            return
        distance = float(detection.range_m)
        boolean_confirmation_missing = False
        if distance <= TRACK_STALL_MIN_RANGE_M or not self._far_noisy_acquisition or boolean_confirmation_missing:
            self._track_range_history.clear()
            self._stall_recovery_until = float('-inf')
            return
        self._track_range_history.append((float(t), distance))
        while self._track_range_history and t - self._track_range_history[0][0] > TRACK_STALL_WINDOW_S:
            self._track_range_history.popleft()
        if len(self._track_range_history) < 4 or self._track_range_history[-1][0] - self._track_range_history[0][0] < TRACK_STALL_WINDOW_S * 0.65:
            return
        rows = list(self._track_range_history)
        edge = min(3, max(1, len(rows) // 3))
        start_distance = float(np.median([row[1] for row in rows[:edge]]))
        end_distance = float(np.median([row[1] for row in rows[-edge:]]))
        if start_distance - end_distance < TRACK_STALL_MIN_PROGRESS_M:
            self._stall_recovery_until = max(self._stall_recovery_until, t + TRACK_STALL_RECOVERY_S)

    def _next_phase(self, t: float, detection: Optional[Detection], local_needed: bool, drone_pos: Optional[np.ndarray]=None) -> str:
        phase = self.phase
        if phase in (Phase.SEARCH_TRANSIT, Phase.SEARCH_LOCAL):
            if detection is not None:
                return Phase.TENTATIVE_DETECTION
            return Phase.SEARCH_LOCAL if local_needed else Phase.SEARCH_TRANSIT
        if phase == Phase.TENTATIVE_DETECTION:
            if self.tracker.recent_hits(TARGET_CONFIRM_WINDOW) >= TARGET_CONFIRM_HITS:
                return Phase.TRACK_AND_INTERCEPT
            if self.tracker.missing_time(t) > TARGET_CONFIRM_WINDOW * DT:
                self._last_candidate = None
                self.tracker.drop()
                return Phase.SEARCH_LOCAL if local_needed else Phase.SEARCH_TRANSIT
            return Phase.TENTATIVE_DETECTION
        if phase == Phase.TRACK_AND_INTERCEPT:
            if detection is not None:
                self._reacquire_started_at = None
                self._reacquire_started_far = False
                return Phase.TRACK_AND_INTERCEPT
            if not self._terminal_active and self.tracker.missing_time(t) <= TRACK_DROPOUT_HOLD_S and (t <= self._stall_recovery_until):
                return Phase.TRACK_AND_INTERCEPT
            if self._terminal_recenter_active and self._terminal_recenter_started_at is not None and (t - self._terminal_recenter_started_at < TERMINAL_RECENTER_TIMEOUT_S) and (self.tracker.missing_time(t) <= self._terminal_recenter_loss_hold_s()):
                return Phase.TRACK_AND_INTERCEPT
            self._reacquire_started_at = t
            track_position = self.tracker.predicted_position(t)
            self._reacquire_started_far = bool(drone_pos is not None and track_position is not None and (float(np.linalg.norm(np.asarray(track_position, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64))) > REACQUIRE_CAUTION_RANGE_M))
            return Phase.REACQUIRE
        if phase == Phase.REACQUIRE:
            if detection is not None:
                self._reacquire_started_at = None
                self._reacquire_started_far = False
                return Phase.TRACK_AND_INTERCEPT
            elapsed = 0.0 if self._reacquire_started_at is None else t - self._reacquire_started_at
            timeout = REACQUIRE_FAR_TIMEOUT_S if self._reacquire_started_far else REACQUIRE_TIMEOUT_S
            if self.tracker.last_position is not None and elapsed <= timeout:
                return Phase.REACQUIRE
            self.tracker.drop()
            self._last_candidate = None
            self._reacquire_started_at = None
            self._reacquire_started_far = False
            return Phase.SEARCH_LOCAL if local_needed else Phase.SEARCH_TRANSIT
        return Phase.SEARCH_TRANSIT

    def _search_vertical_rate(self, own_z: float, agl: float) -> float:
        if self._remembered_target_z is not None:
            return vertical_rate_to_target_z(own_z, self._remembered_target_z)
        return vertical_rate_to_target_z(agl, TARGET_SEARCH_AGL)

    def _reacquire_command(self, t: float, drone_pos: np.ndarray, drone_velocity: np.ndarray, target_pos: np.ndarray) -> tuple[np.ndarray, float]:
        toward = unit(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64))
        bearing = math.atan2(target_pos[1] - drone_pos[1], target_pos[0] - drone_pos[0])
        elapsed = 0.0 if self._reacquire_started_at is None else max(0.0, t - self._reacquire_started_at)
        memory_distance = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)))
        bridge_parallel_contact = self._terminal_decision.profile == TerminalProfile.PARALLEL_FULL and memory_distance <= 1.75 and (self.tracker.missing_time(t) <= 0.15)
        target_velocity = np.asarray(self.tracker.last_velocity, dtype=np.float64)
        drone_velocity = np.asarray(drone_velocity, dtype=np.float64)
        relative_position = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)

        def alignment(a: np.ndarray, b: np.ndarray) -> float:
            a_norm = float(np.linalg.norm(a))
            b_norm = float(np.linalg.norm(b))
            if a_norm <= 1e-06 or b_norm <= 1e-06:
                return -1.0
            return float(np.dot(a, b) / (a_norm * b_norm))
        self._reacquire_context = ReacquireContext(memory_distance=memory_distance, visibility_confirmed=False, time_s=float(t), drone_speed=float(np.linalg.norm(drone_velocity)), target_speed=float(np.linalg.norm(target_velocity)), drone_target_alignment=alignment(drone_velocity, target_velocity), drone_los_alignment=alignment(drone_velocity, relative_position), target_los_alignment=alignment(target_velocity, relative_position), horizontal_exit_angle=abs(self._last_detection_horizontal_angle), z_error=float(relative_position[2]), far_noisy_acquisition=bool(self._far_noisy_acquisition))
        learned_reacquire = self.learned_policies.select_reacquire(self._reacquire_context)
        self._learned_reacquire_rule_id = learned_reacquire.rule_id
        if DEV_FORCE_REACQUIRE_SPEED is not None:
            speed = DEV_FORCE_REACQUIRE_SPEED
        elif learned_reacquire.force_speed:
            speed = learned_reacquire.speed_mps
        elif bridge_parallel_contact or memory_distance > REACQUIRE_CAUTION_RANGE_M:
            speed = MAX_SPEED
        else:
            speed = learned_reacquire.speed_mps
        v = speed * toward
        if DEV_FORCE_NEAR_REACQUIRE_PROFILE is not None and memory_distance <= 2.0 and (self.tracker.missing_time(t) <= 0.2):
            if DEV_FORCE_NEAR_REACQUIRE_PROFILE == 'FULL_MEMORY':
                v = MAX_SPEED * toward
            elif DEV_FORCE_NEAR_REACQUIRE_PROFILE == 'HOLD_COMMAND':
                v = np.asarray(self.previous_velocity_command, dtype=np.float64).copy()
            elif DEV_FORCE_NEAR_REACQUIRE_PROFILE == 'VELOCITY_MATCH':
                target_tangent = target_velocity - float(np.dot(target_velocity, toward)) * toward
                tangent_speed = float(np.linalg.norm(target_tangent))
                radial_speed = math.sqrt(max(0.0, MAX_SPEED * MAX_SPEED - tangent_speed * tangent_speed))
                v = target_tangent + radial_speed * toward
            else:
                raise ValueError(f'unknown near reacquire profile: {DEV_FORCE_NEAR_REACQUIRE_PROFILE}')
        last_angle = self._last_detection_horizontal_angle
        if abs(last_angle) < math.radians(2.0):
            yaw = bearing
        else:
            exit_sign = 1.0 if last_angle > 0.0 else -1.0
            bias_deg = min(REACQUIRE_MAX_YAW_BIAS_DEG, REACQUIRE_INITIAL_YAW_BIAS_DEG + REACQUIRE_YAW_SWEEP_RATE_DEG_S * elapsed)
            yaw = wrap_angle(bearing - exit_sign * math.radians(bias_deg))
        return (v, yaw)

    def _search_transit_command(self, own_xy, own_z, agl, filtered_clue_xy):
        vz = self._search_vertical_rate(own_z, agl)
        horizontal_dir = unit(filtered_clue_xy - own_xy)
        v = speed_budget_velocity(horizontal_dir, vz)
        return (v, _heading_to(horizontal_dir))

    def _search_local_command(self, own_xy, own_z, agl, raw_clue_xy):
        vz = self._search_vertical_rate(own_z, agl)
        horizontal_dir = local_search_direction(own_xy, raw_clue_xy, radial_weight=0.6 if DEV_LOCAL_SEARCH_RADIAL_WEIGHT is None else float(DEV_LOCAL_SEARCH_RADIAL_WEIGHT), tangent_weight=0.4 if DEV_LOCAL_SEARCH_TANGENT_WEIGHT is None else float(DEV_LOCAL_SEARCH_TANGENT_WEIGHT))
        v = speed_budget_velocity(horizontal_dir, vz)
        return (v, _heading_to(horizontal_dir))

    def _tentative_command(self, own_xy, own_z, agl, candidate: Optional[Detection]):
        if candidate is None:
            return (speed_budget_velocity(np.zeros(2), 0.0), 0.0)
        vz = self._search_vertical_rate(own_z, agl) if self._remembered_target_z is None else vertical_rate_to_target_z(own_z, self._remembered_target_z)
        candidate_xy = candidate.position[0:2]
        horizontal_dir = unit(candidate_xy - own_xy)
        v = speed_budget_velocity(horizontal_dir, vz)
        return (v, _heading_to(horizontal_dir))

    def _intercept_command(self, drone_pos, drone_vel, target_pos, target_vel, target_accel, terminal_recenter: bool=False, terminal_profile: str=TerminalProfile.NORMAL_PN):
        self._terminal_active = False
        self._terminal_accel_scale = 1.0
        self._terminal_dynamics_enabled = False
        self._parallel_chase_speed_lock = False
        self._rule_aware_lead_active = False
        if terminal_profile == TerminalProfile.STOP_AND_YAW:
            v = np.zeros(3, dtype=np.float64)
        elif terminal_profile == TerminalProfile.BRAKE_YAW_HOMING:
            relative_position = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
            distance = float(np.linalg.norm(relative_position))
            camera_error = 0.0 if self._terminal_context is None else self._terminal_context.camera_error
            detection_present = bool(self._terminal_context is not None and self._terminal_context.detection_present)
            braking = bool(distance <= TERMINAL_RECENTER_RANGE_M and (camera_error >= math.radians(TERMINAL_RECENTER_ENTER_ANGLE_DEG) or not detection_present))
            if braking:
                v = np.zeros(3, dtype=np.float64)
                self._terminal_recenter_active = True
                if self._terminal_recenter_started_at is None:
                    self._terminal_recenter_started_at = self.frame_index * DT
            else:
                self._rule_aware_lead_active = True
                v = proportional_navigation_command(drone_pos, drone_vel, target_pos, target_vel, PROPORTIONAL_NAVIGATION_GAIN, PROPORTIONAL_NAVIGATION_ACCEL_MPS2, TERMINAL_STRONG_PN_RESPONSE_S)
        elif terminal_profile in (TerminalProfile.LATERAL_CUT, TerminalProfile.STAGED_LATERAL_CUT):
            self._rule_aware_lead_active = True
            relative_position = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
            if terminal_profile == TerminalProfile.STAGED_LATERAL_CUT and float(np.linalg.norm(relative_position)) > 5.0:
                v = proportional_navigation_command(drone_pos, drone_vel, target_pos, target_vel, PROPORTIONAL_NAVIGATION_GAIN, PROPORTIONAL_NAVIGATION_ACCEL_MPS2, PROPORTIONAL_NAVIGATION_RESPONSE_S)
            else:
                v = terminal_recenter_command(drone_pos, target_pos, target_vel, target_accel, drone_velocity=drone_vel, tangent_scale=self._terminal_decision.recenter_tangent_scale, closing_speed_mps=self._terminal_decision.recenter_closing_speed_mps)
        elif terminal_recenter and terminal_profile == TerminalProfile.NORMAL_PN:
            relative_position = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
            target_speed = float(np.linalg.norm(target_vel))
            longitudinal_deceleration = -float(np.dot(target_vel, target_accel)) / target_speed if target_speed > 1e-06 else 0.0
            self._parallel_chase_speed_lock = bool(longitudinal_deceleration < TERMINAL_RECENTER_STOP_DECEL_MPS2 and parallel_chase_speed_lock(relative_position, drone_vel, target_vel))
            v = terminal_recenter_command(drone_pos, target_pos, target_vel, target_accel, drone_velocity=drone_vel, tangent_scale=self._terminal_decision.recenter_tangent_scale, closing_speed_mps=self._terminal_decision.recenter_closing_speed_mps)
            distance = float(np.linalg.norm(relative_position))
            if not self._parallel_chase_speed_lock:
                v = limit_norm(v, chase_speed_limit(distance))
        elif terminal_profile == TerminalProfile.PARALLEL_FULL:
            v = MAX_SPEED * unit(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64))
            self._parallel_chase_speed_lock = True
        elif terminal_profile == TerminalProfile.STAGED_HIGH_AUTHORITY_PN:
            distance = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)))
            if distance > 5.0:
                if not self._pn_engagement_decided:
                    line_of_sight = unit(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64))
                    relative_velocity = np.asarray(target_vel, dtype=np.float64) - np.asarray(drone_vel, dtype=np.float64)
                    closing_speed = -float(np.dot(relative_velocity, line_of_sight))
                    radial_relative_speed = float(np.dot(relative_velocity, line_of_sight))
                    tangent_relative_velocity = relative_velocity - radial_relative_speed * line_of_sight
                    line_of_sight_rate = float(np.linalg.norm(tangent_relative_velocity) / max(distance, 1e-06))
                    self._pn_engagement_active = bool(closing_speed <= PROPORTIONAL_NAVIGATION_MAX_DIRECT_CLOSING_MPS or line_of_sight_rate >= PROPORTIONAL_NAVIGATION_MIN_LOS_RATE_RAD_S)
                    self._pn_engagement_decided = True
                if self._pn_engagement_active:
                    self._rule_aware_lead_active = True
                    v = proportional_navigation_command(drone_pos, drone_vel, target_pos, target_vel, 4.0, 25.0, 0.42)
                else:
                    v = direct_pursuit_command(drone_pos, target_pos)
            else:
                self._rule_aware_lead_active = True
                v = proportional_navigation_command(drone_pos, drone_vel, target_pos, target_vel, 4.0, 25.0, 0.9)
        elif terminal_profile in (TerminalProfile.CAPTURE_PN, TerminalProfile.STRONG_TURN_PN, TerminalProfile.HIGH_AUTHORITY_PN, TerminalProfile.TUNED_PN_045):
            self._rule_aware_lead_active = True
            if terminal_profile in (TerminalProfile.HIGH_AUTHORITY_PN,):
                distance = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)))
                navigation_gain = 4.0
                max_lateral_accel = 25.0
                response = 0.9
            elif terminal_profile == TerminalProfile.TUNED_PN_045:
                navigation_gain = 3.0
                max_lateral_accel = 15.0
                response = 0.45
            else:
                navigation_gain = PROPORTIONAL_NAVIGATION_GAIN
                max_lateral_accel = PROPORTIONAL_NAVIGATION_ACCEL_MPS2
                response = TERMINAL_STRONG_PN_RESPONSE_S if terminal_profile == TerminalProfile.STRONG_TURN_PN else PROPORTIONAL_NAVIGATION_RESPONSE_S
            v = proportional_navigation_command(drone_pos, drone_vel, target_pos, target_vel, navigation_gain, max_lateral_accel, response)
        elif terminal_profile == TerminalProfile.ACCEL_LEAD:
            self._rule_aware_lead_active = True
            v = short_horizon_visible_chase(drone_pos, target_pos, target_vel, target_accel, lead_horizon_s=TERMINAL_ACCEL_LEAD_HORIZON_S)
        elif terminal_profile == TerminalProfile.COLLISION_LEAD:
            self._rule_aware_lead_active = True
            v = calculate_intercept_velocity(drone_pos, target_pos, target_vel)
        elif terminal_profile == TerminalProfile.TERMINAL_HOMING:
            self._rule_aware_lead_active = True
            v = terminal_homing_command(drone_pos, target_pos, target_vel, drone_velocity=drone_vel, target_acceleration=target_accel)
        elif terminal_profile in (TerminalProfile.DIRECT_STABLE, TerminalProfile.VERTICAL_ALIGN, TerminalProfile.CAMERA_RAY, TerminalProfile.PREDICTED_CAMERA_RAY):
            v = direct_pursuit_command(drone_pos, target_pos)
        else:
            distance = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)))
            if distance <= RULE_AWARE_LEAD_ENTER_M and (not self._pn_engagement_decided):
                line_of_sight = unit(np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64))
                relative_velocity = np.asarray(target_vel, dtype=np.float64) - np.asarray(drone_vel, dtype=np.float64)
                closing_speed = -float(np.dot(relative_velocity, line_of_sight))
                radial_relative_speed = float(np.dot(relative_velocity, line_of_sight))
                tangent_relative_velocity = relative_velocity - radial_relative_speed * line_of_sight
                line_of_sight_rate = float(np.linalg.norm(tangent_relative_velocity) / max(distance, 1e-06))
                self._pn_engagement_active = bool(closing_speed <= PROPORTIONAL_NAVIGATION_MAX_DIRECT_CLOSING_MPS or line_of_sight_rate >= PROPORTIONAL_NAVIGATION_MIN_LOS_RATE_RAD_S)
                self._pn_engagement_decided = True
            if self._pn_engagement_decided and (not self._pn_engagement_active) and (distance <= PROPORTIONAL_NAVIGATION_STALL_RANGE_M):
                now = self.frame_index * DT
                self._terminal_range_history.append((now, distance))
                while self._terminal_range_history and now - self._terminal_range_history[0][0] > PROPORTIONAL_NAVIGATION_STALL_WINDOW_S:
                    self._terminal_range_history.popleft()
                if self._terminal_range_history and now - self._terminal_range_history[0][0] >= 0.9 * PROPORTIONAL_NAVIGATION_STALL_WINDOW_S and (self._terminal_range_history[0][1] - distance < PROPORTIONAL_NAVIGATION_STALL_MIN_PROGRESS_M):
                    self._pn_engagement_active = True
            elif distance > PROPORTIONAL_NAVIGATION_STALL_RANGE_M:
                self._terminal_range_history.clear()
            if distance <= RULE_AWARE_LEAD_ENTER_M and self._pn_engagement_active:
                self._rule_aware_lead_active = True
                v = proportional_navigation_command(drone_pos, drone_vel, target_pos, target_vel, PROPORTIONAL_NAVIGATION_GAIN, PROPORTIONAL_NAVIGATION_ACCEL_MPS2, PROPORTIONAL_NAVIGATION_RESPONSE_S)
            else:
                v = direct_pursuit_command(drone_pos, target_pos)
        yaw = yaw_toward_current_target(drone_pos, target_pos)
        return (v, yaw)
