"""Shared training harness for the per-family baseline starters.

Each RL/<family_id>/train.py calls train_family(): PPO is trained on real
generator tasks, saved, packaged into a validator-ready submission.zip, and
smoke-tested against the family's policy contract.

The policy sees a min-pooled depth frame plus the full state vector; rgb is
never requested. Interceptor uses 256×256 nearest-surface pool so a few-pixel
target at 60–100 m is not deleted by stride downsampling. Multi-drone families
train one shared policy by exposing each drone as one slot of a vectorized
environment over a single shared simulation.

Interceptor (cf_interceptor) uses longer rollouts, a larger MLP, a short-gap
curriculum, and a three-gate action reward: clue when far, true target plus
closing speed inside ~15 m, then lead-aim plus last-metre commit inside ~5 m
so the 15 cm ram is not a fly-by. Crash/tilt is penalised in training only.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from swarm.challenge_families import get_challenge_family
from swarm.constants import (
    BENCHMARK_TOTAL_SEED_COUNT,
    INTERCEPTOR_ALT_MIN_M,
    INTERCEPTOR_MAX_START_DISTANCE_M,
    INTERCEPTOR_MIN_START_DISTANCE_M,
    INTERCEPTOR_MINER_SPEED,
    SIM_DT,
    SWARM_MAX_DRONES,
    SWARM_MIN_DRONES,
)
from swarm.domain_model import get_policy_interface_contract
from swarm.policy_interface import resolve_policy_interface_version, smoke_test_policy_package
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.task_gen import random_task, screening_task

from agent_template import InterceptorFeaturesExtractor

_MAX_MAP_SEED = 2**32 - 1

POLICY_DEPTH_SIZE = 64
# Interceptor HD depth is 1024² so a 36 cm target is a few pixels at 60–100 m.
# Stride-16 into 64² drops those pixels. 256² + nearest-surface pool keeps them.
INTERCEPTOR_DEPTH_SIZE = 256
DEFAULT_TIMESTEPS = 50_000
DEFAULT_INTERCEPTOR_TIMESTEPS = 500_000
DEFAULT_SWARM_DRONES = 4
_MAX_TASK_RESAMPLES = 200

# Stage 0 is 3–5 m so a ram can appear in the buffer. Promote when the
# recent window actually catches; last stage is the validator 60–100 m band.
_INTERCEPTOR_CURRICULUM_GAPS = (
    (3.0, 5.0),
    (8.0, 12.0),
    (15.0, 30.0),
    (30.0, 50.0),
    (45.0, 70.0),
    (INTERCEPTOR_MIN_START_DISTANCE_M, INTERCEPTOR_MAX_START_DISTANCE_M),
)
_CURRICULUM_WINDOW = 20
_CURRICULUM_PROMOTE_MIN = 15
_CURRICULUM_PROMOTE_RATE = 0.10
_CURRICULUM_PROMOTE_CATCHES = 2

# Commanded yaw/dir toward the clue, not actual velocity. 0.02/step so a few
# seconds of "face it and go" is a clear return. Catch bonus >> 60 s of
# alignment so hovering-pointed is not better than intercepting.
_INTERCEPTOR_ALIGN_SCALE = 0.02
_INTERCEPTOR_CATCH_BONUS = 80.0
_INTERCEPTOR_CRASH_PENALTY = 40.0
_INTERCEPTOR_CRASH_REASONS = frozenset({"TILT", "OBSTACLE_COLLISION"})
# Inside this range the 10–40 m clue is the wrong aim point; switch to the
# true target (flee trigger is 12 m).
_INTERCEPTOR_NEAR_M = 15.0
# Last-mile ram: lead the jinking evader and make the final metres matter.
_INTERCEPTOR_RAM_M = 5.0
_INTERCEPTOR_LEAD_SEC = 0.6
_INTERCEPTOR_COMMIT_SCALE = 0.5

_DEFAULT_PPO = dict(
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    learning_rate=3e-4,
    max_grad_norm=0.5,
)


# Chase episodes are 60 s × 50 Hz = 3,000 steps. n_steps=512 only sees ~10 s
# of a pursuit; 2048 covers ~40 s so GAE can credit a catch.
# target_kl early-stops extra epochs when the policy has already moved; it does
# not change observations or reward. lr/n_epochs are lower so KL stays ~0.01.
_INTERCEPTOR_PPO = dict(
    n_steps=2048,
    batch_size=256,
    n_epochs=4,
    gamma=0.995,
    gae_lambda=0.95,
    ent_coef=0.005,
    learning_rate=1e-4,
    max_grad_norm=0.5,
    target_kl=0.03,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        features_extractor_class=InterceptorFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    ),
)


def interceptor_curriculum_gap(stage: int) -> tuple[float, float]:
    """Chase-gap band for a curriculum stage. Clips to the last (validator) band."""
    last = len(_INTERCEPTOR_CURRICULUM_GAPS) - 1
    idx = max(0, min(int(stage), last))
    return _INTERCEPTOR_CURRICULUM_GAPS[idx]


def should_promote_curriculum(
    caught_window: list[bool] | deque,
    *,
    min_episodes: int = _CURRICULUM_PROMOTE_MIN,
    rate: float = _CURRICULUM_PROMOTE_RATE,
    min_catches: int = _CURRICULUM_PROMOTE_CATCHES,
) -> bool:
    """Promote when the recent window has both enough episodes and real catches."""
    n = len(caught_window)
    if n < min_episodes:
        return False
    catches = int(sum(bool(x) for x in caught_window))
    return catches >= min_catches and (catches / n) >= rate


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def action_offset_align_reward(
    action: np.ndarray,
    offset: np.ndarray,
    *,
    scale: float = _INTERCEPTOR_ALIGN_SCALE,
) -> float:
    """Reward commanded yaw (XY) and dir toward ``offset`` (2D clue or 3D target)."""
    act = np.asarray(action, dtype=np.float64).reshape(-1)
    off = np.asarray(offset, dtype=np.float64).reshape(-1)
    if act.shape[0] < 5 or off.shape[0] < 2:
        return 0.0
    off_xy = off[:2]
    off_xy_n = float(np.linalg.norm(off_xy))
    if off_xy_n < 1e-3:
        yaw_align = 0.0
    else:
        yaw_cmd = float(np.clip(act[4], -1.0, 1.0) * np.pi)
        desired = float(np.arctan2(off_xy[1], off_xy[0]))
        yaw_align = float(np.cos(_wrap_pi(yaw_cmd - desired)))
    if off.shape[0] >= 3:
        dir_vec = act[:3]
        aim = off[:3]
    else:
        dir_vec = act[:2]
        aim = off_xy
    dir_n = float(np.linalg.norm(dir_vec))
    aim_n = float(np.linalg.norm(aim))
    if dir_n < 1e-3 or aim_n < 1e-3:
        dir_align = 0.0
    else:
        dir_align = float(np.clip(np.dot(dir_vec, aim) / (dir_n * aim_n), -1.0, 1.0))
    speed = float(np.clip(act[3], 0.0, 1.0))
    return float(scale * 0.5 * (yaw_align + speed * dir_align))


def clue_action_align_reward(
    action: np.ndarray,
    state: np.ndarray,
    *,
    scale: float = _INTERCEPTOR_ALIGN_SCALE,
) -> float:
    """Far-range: align the action with the observed clue ``state[-2:]``."""
    vec = np.asarray(state, dtype=np.float64).reshape(-1)
    if vec.shape[0] < 2:
        return 0.0
    return action_offset_align_reward(action, vec[-2:], scale=scale)


def close_range_reward(
    prev_dist: float | None,
    dist: float | None,
    *,
    scale: float = _INTERCEPTOR_ALIGN_SCALE,
    max_step_m: float = INTERCEPTOR_MINER_SPEED * SIM_DT,
) -> float:
    """Near-range: reward live chaser→target closing, scaled to ±align scale."""
    if prev_dist is None or dist is None:
        return 0.0
    if not (np.isfinite(prev_dist) and np.isfinite(dist)) or max_step_m <= 0.0:
        return 0.0
    return float(scale * np.clip((prev_dist - dist) / max_step_m, -1.0, 1.0))


def lead_aim_offset(
    delta: np.ndarray,
    target_vel: np.ndarray | None,
    *,
    lead_sec: float = _INTERCEPTOR_LEAD_SEC,
) -> np.ndarray:
    """Aim at where the target will be, not where it is (pure pursuit lags a jink)."""
    off = np.asarray(delta, dtype=np.float64).reshape(-1)
    if target_vel is None or lead_sec <= 0.0:
        return off
    vel = np.asarray(target_vel, dtype=np.float64).reshape(-1)
    if vel.size < off.size:
        vel = np.pad(vel, (0, off.size - vel.size))
    return off + float(lead_sec) * vel[: off.size]


def last_metre_commit_reward(
    prev_dist: float | None,
    dist: float | None,
    *,
    inner_m: float = _INTERCEPTOR_RAM_M,
    scale: float = _INTERCEPTOR_COMMIT_SCALE,
) -> float:
    """Potential on 1/dist inside ``inner_m``. Hovering does not farm; only closing pays."""
    if prev_dist is None or dist is None:
        return 0.0
    if not (np.isfinite(prev_dist) and np.isfinite(dist)):
        return 0.0
    if min(float(prev_dist), float(dist)) >= inner_m:
        return 0.0

    def _phi(d: float) -> float:
        clipped = min(max(float(d), 0.0), inner_m)
        return 1.0 / (clipped + 0.1) - 1.0 / (inner_m + 0.1)

    return float(scale * (_phi(float(dist)) - _phi(float(prev_dist))))


def interceptor_terminal_bonus(
    *,
    done: bool,
    caught: bool,
    failure_reason: str | None,
) -> float:
    """Train-only terminal: +catch or −crash/tilt. Timeout and target-crash are 0."""
    if not done:
        return 0.0
    if caught:
        return float(_INTERCEPTOR_CATCH_BONUS)
    if str(failure_reason or "") in _INTERCEPTOR_CRASH_REASONS:
        return float(-_INTERCEPTOR_CRASH_PENALTY)
    return 0.0


def _policy_depth_size(family_id: str) -> int:
    if family_id == "cf_interceptor":
        return INTERCEPTOR_DEPTH_SIZE
    return POLICY_DEPTH_SIZE


def downsample_depth(depth: np.ndarray, size: int) -> np.ndarray:
    """Block-pool depth to size×size×1. Must match RL/agent_template.py.

    Env depth is 0=near, 1=far. Uniform blocks use min-pool (nearest surface)
    so a few-pixel target against sky survives 1024→256. Mixed ground+sky
    blocks would otherwise report ground and erase an airborne target; those
    keep the nearest pixel in the far cluster instead. Other families (64²)
    always min-pool so nearby obstacles are not dropped.
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = int(d.shape[0]), int(d.shape[1])
    if h == size and w == size:
        out = d
    elif h < size or w < size:
        ys = (np.arange(size) * h // size).clip(0, max(0, h - 1))
        xs = (np.arange(size) * w // size).clip(0, max(0, w - 1))
        out = d[ys][:, xs]
    else:
        bh, bw = h // size, w // size
        cropped = d[: size * bh, : size * bw]
        blocks = cropped.reshape(size, bh, size, bw).transpose(0, 2, 1, 3)
        flat = blocks.reshape(size, size, bh * bw)
        out = flat.min(axis=-1)
        if size >= INTERCEPTOR_DEPTH_SIZE and bh * bw >= 4:
            p10 = np.percentile(flat, 10, axis=-1)
            p90 = np.percentile(flat, 90, axis=-1)
            mixed = (p90 - p10) > 0.25
            if np.any(mixed):
                mid = p10 + 0.5 * (p90 - p10)
                far_only = np.where(flat >= mid[..., None], flat, 1.0)
                out = np.where(mixed, far_only.min(axis=-1), out)
    return np.ascontiguousarray(out.reshape(size, size, 1), dtype=np.float32)


def _policy_view(depth: np.ndarray, state: np.ndarray, size: int) -> dict[str, np.ndarray]:
    return {
        "depth": downsample_depth(depth, size),
        "state": np.asarray(state, dtype=np.float32),
    }


def _unique_map_seeds(count: int, rng: random.Random) -> list[int]:
    """Validator-sized unique map seeds in [0, 2**32-1]."""
    if count <= 0:
        return []
    seen: set[int] = set()
    seeds: list[int] = []
    while len(seeds) < count:
        value = rng.randint(0, _MAX_MAP_SEED)
        if value not in seen:
            seen.add(value)
            seeds.append(value)
    return seeds


def _load_map_seed_file(path: Path) -> list[int]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("seeds", payload.get("map_seeds"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must be a JSON list of ints, or {{'seeds': [...]}}")
    seeds = [int(s) for s in payload]
    if any(s < 0 or s > _MAX_MAP_SEED for s in seeds):
        raise ValueError(f"{path} contains a seed outside [0, {_MAX_MAP_SEED}]")
    return seeds


class FamilyVecEnv(VecEnv):
    """One shared simulation exposed as NUM_DRONES single-drone slots.

    Interceptor episodes cycle a map-seed pool through a short-gap curriculum
    (3–5 m until catches appear, then out to the validator 60–100 m bands).
    ``--seed`` only seeds PPO and this pool; it is not a single map.
    """

    def __init__(
        self,
        family_id: str,
        *,
        seed: int,
        n_drones: int | None = None,
        map_seeds: list[int] | None = None,
        curriculum: bool = True,
    ):
        self._family_id = family_id
        self._family = get_challenge_family(family_id)
        self._rng = random.Random(seed)
        self._forced_drones = n_drones
        self._map_seeds = list(map_seeds) if map_seeds else None
        self._seed_cursor = 0
        self._last_map_seed: int | None = None
        self._curriculum_enabled = bool(curriculum) and family_id == "cf_interceptor"
        last_stage = len(_INTERCEPTOR_CURRICULUM_GAPS) - 1
        self._curriculum_stage = 0 if self._curriculum_enabled else last_stage
        self._caught_window: deque[bool] = deque(maxlen=_CURRICULUM_WINDOW)
        self._env, first_obs = self._build_episode()
        self._depth_size = _policy_depth_size(family_id)
        self._ep_return = 0.0
        self._ep_len = 0

        state_dim = int(self._env.observation_space["state"].shape[-1])
        observation_space = spaces.Dict(
            {
                "depth": spaces.Box(
                    0.0, 1.0, (self._depth_size, self._depth_size, 1), np.float32
                ),
                "state": spaces.Box(-np.inf, np.inf, (state_dim,), np.float32),
            }
        )
        contract = get_policy_interface_contract(
            family_id, resolve_policy_interface_version(family_id, None)
        )
        action_space = spaces.Box(
            low=np.asarray(contract["action_space"]["lower_bound"], dtype=np.float32),
            high=np.asarray(contract["action_space"]["upper_bound"], dtype=np.float32),
            dtype=np.float32,
        )
        super().__init__(self._env.NUM_DRONES, observation_space, action_space)
        self._last_obs = first_obs
        self._pending_actions: np.ndarray | None = None

    def _next_task(self):
        template = self._family.benchmark_template()
        if self._map_seeds:
            idx = self._seed_cursor % len(self._map_seeds)
            seed = int(self._map_seeds[idx])
            offset = idx
            total = len(self._map_seeds)
        else:
            seed = self._rng.randint(0, _MAX_MAP_SEED)
            offset = self._seed_cursor % max(1, len(template) or 1)
            total = max(1, len(template) or 1)
        self._seed_cursor += 1
        self._last_map_seed = seed

        if self._family_id == "cf_interceptor":
            last_stage = len(_INTERCEPTOR_CURRICULUM_GAPS) - 1
            if self._curriculum_stage < last_stage:
                lo, hi = interceptor_curriculum_gap(self._curriculum_stage)
                return screening_task(
                    sim_dt=SIM_DT,
                    seed=seed,
                    challenge_type=2,
                    distance_range=(lo, hi),
                    family_id=self._family_id,
                )

        if template:
            return self._family.build_benchmark_tasks(
                sim_dt=SIM_DT,
                seeds=[seed],
                offset=offset,
                total_seed_count=total,
            )[0]
        return random_task(sim_dt=SIM_DT, seed=seed, family_id=self._family_id)

    def _build_episode(self):
        for _ in range(_MAX_TASK_RESAMPLES):
            task = self._next_task()
            if self._forced_drones is None:
                break
            if int(getattr(task, "num_drones", 1) or 1) == self._forced_drones:
                break
        else:
            raise RuntimeError(
                f"Could not sample a {self._forced_drones}-drone task for {self._family_id}"
            )
        env, first_obs = make_env_with_initial_obs(task)
        first_obs = self._tighten_first_contact_spawn(env, first_obs)
        return env, first_obs

    def _tighten_first_contact_spawn(self, env, first_obs):
        """Stage 0 only: pin target to min altitude so 3–5 m XY is ~3–5 m 3D."""
        if self._family_id != "cf_interceptor":
            return first_obs
        if not self._curriculum_enabled or int(self._curriculum_stage) != 0:
            return first_obs
        tpos = getattr(env, "_target_pos", None)
        uid = getattr(env, "_target_uid", None)
        if tpos is None or uid is None:
            return first_obs
        import pybullet as p

        floor = float(getattr(env, "_target_floor_z", 0.0))
        new = np.asarray(tpos, dtype=np.float64).copy()
        new[2] = floor + INTERCEPTOR_ALT_MIN_M
        env._target_pos = new
        env._target_vel = np.zeros(3, dtype=float)
        env.task.goal = (float(new[0]), float(new[1]), float(new[2]))
        env.GOAL_POS = new.copy()
        cli = getattr(env, "CLIENT", 0)
        _pos, orn = p.getBasePositionAndOrientation(int(uid), physicsClientId=cli)
        p.resetBasePositionAndOrientation(
            int(uid), new.tolist(), orn, physicsClientId=cli
        )
        p.resetBaseVelocity(int(uid), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cli)
        refresh = getattr(getattr(env, "family_runtime", None), "_refresh_search_clue", None)
        if callable(refresh):
            refresh(env)
        compute_obs = getattr(env, "_computeObs", None)
        if callable(compute_obs):
            return compute_obs()
        return first_obs

    def _slot_obs(self, obs: dict[str, np.ndarray], slot: int) -> dict[str, np.ndarray]:
        if self.num_envs == 1:
            return _policy_view(obs["depth"], obs["state"], self._depth_size)
        return _policy_view(obs["depth"][slot], obs["state"][slot], self._depth_size)

    def _stack_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        slots = [self._slot_obs(obs, i) for i in range(self.num_envs)]
        return {
            key: np.stack([slot[key] for slot in slots])
            for key in ("depth", "state")
        }

    def _chase_geometry(
        self,
    ) -> tuple[np.ndarray | None, float | None, np.ndarray | None]:
        tpos = getattr(self._env, "_target_pos", None)
        if tpos is None:
            return None, None, None
        chaser = self._env._getDroneStateVector(0)
        cpos = np.asarray(chaser[0:3], dtype=np.float64)
        delta = np.asarray(tpos, dtype=np.float64) - cpos
        dist = float(np.linalg.norm(delta))
        if not np.isfinite(dist):
            return None, None, None
        tvel = np.asarray(
            getattr(self._env, "_target_vel", np.zeros(3)), dtype=np.float64
        ).reshape(-1)
        return delta, dist, tvel

    def _shape_interceptor_reward(
        self,
        base_reward: float,
        info: dict,
        done: bool,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        chase_delta: np.ndarray | None,
        chase_dist: float | None,
        next_dist: float | None,
        target_vel: np.ndarray | None = None,
    ) -> float:
        """Far: clue. 5–15 m: true target. <5 m: lead aim + last-metre commit.

        Until a catch, evaluate_rollout stays at participation (~0.01). A
        train-only catch bonus keeps intercepting better than timing out;
        crash/tilt is penalised so dying at 5 m is not a local optimum.
        """
        extra = 0.0
        ram = chase_dist is not None and chase_dist <= _INTERCEPTOR_RAM_M
        near = chase_dist is not None and chase_dist <= _INTERCEPTOR_NEAR_M
        if ram and chase_delta is not None:
            extra = action_offset_align_reward(
                action, lead_aim_offset(chase_delta, target_vel)
            )
            extra += close_range_reward(chase_dist, next_dist)
            extra += last_metre_commit_reward(chase_dist, next_dist)
        elif near and chase_delta is not None:
            extra = action_offset_align_reward(action, chase_delta)
            extra += close_range_reward(chase_dist, next_dist)
        else:
            state = obs.get("state") if obs is not None else None
            if state is not None:
                extra = clue_action_align_reward(action, state)
        extra += interceptor_terminal_bonus(
            done=done,
            caught=bool(info.get("intercept_caught")),
            failure_reason=info.get("failure_reason"),
        )
        return float(base_reward) + extra

    def _maybe_promote_curriculum(self, caught: bool) -> None:
        if not self._curriculum_enabled:
            return
        last = len(_INTERCEPTOR_CURRICULUM_GAPS) - 1
        if self._curriculum_stage >= last:
            return
        self._caught_window.append(bool(caught))
        if not should_promote_curriculum(self._caught_window):
            return
        rate = sum(self._caught_window) / len(self._caught_window)
        old = self._curriculum_stage
        self._curriculum_stage += 1
        self._caught_window.clear()
        lo, hi = interceptor_curriculum_gap(self._curriculum_stage)
        print(
            f"[train] curriculum {old}→{self._curriculum_stage} "
            f"gap={lo:.0f}-{hi:.0f}m window_catch_rate={rate:.3f}"
        )

    def reset(self):
        self._ep_return = 0.0
        self._ep_len = 0
        return self._stack_obs(self._last_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        env_action = self._pending_actions
        if self.num_envs == 1:
            env_action = env_action.reshape(1, -1)
        if self._family_id == "cf_interceptor":
            pre_delta, pre_dist, pre_vel = self._chase_geometry()
        else:
            pre_delta, pre_dist, pre_vel = None, None, None
        obs, reward, terminated, truncated, info = self._env.step(env_action)
        done = bool(terminated or truncated)
        info = info or {}

        shaped = float(reward)
        if self._family_id == "cf_interceptor":
            _, post_dist, _ = self._chase_geometry()
            shaped = self._shape_interceptor_reward(
                shaped,
                info,
                done,
                self._last_obs,
                env_action[0],
                pre_delta,
                pre_dist,
                post_dist,
                pre_vel,
            )

        self._ep_return += shaped
        self._ep_len += 1

        rewards = np.full(self.num_envs, shaped, dtype=np.float32)
        dones = np.full(self.num_envs, done, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        if done:
            caught = bool(info.get("intercept_caught", terminated and not truncated))
            stage = int(self._curriculum_stage)
            gap_lo, gap_hi = interceptor_curriculum_gap(stage)
            self._maybe_promote_curriculum(caught)
            terminal = self._stack_obs(obs)
            ep_stats = {
                "r": float(self._ep_return),
                "l": int(self._ep_len),
                "caught": caught,
                "min_dist": float(info.get("intercept_min_dist", np.nan)),
                "failure_reason": str(info.get("failure_reason", "") or ""),
                "map_seed": self._last_map_seed,
                "curriculum_stage": stage,
                "gap_lo": float(gap_lo),
                "gap_hi": float(gap_hi),
            }
            for slot in range(self.num_envs):
                infos[slot]["terminal_observation"] = {
                    key: terminal[key][slot] for key in terminal
                }
                infos[slot]["TimeLimit.truncated"] = bool(truncated and not terminated)
                infos[slot]["episode"] = ep_stats
            self._env.close()
            self._env, self._last_obs = self._build_episode()
            self._ep_return = 0.0
            self._ep_len = 0
        else:
            self._last_obs = obs

        return self._stack_obs(self._last_obs), rewards, dones, infos

    def close(self) -> None:
        self._env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self._env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None) -> None:
        setattr(self._env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self._env, method_name)(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        return [seed] * self.num_envs


class EpisodeStatCallback(BaseCallback):
    """Log catch rate / min distance from FamilyVecEnv episode infos."""

    def __init__(self, log_every: int = 10):
        super().__init__()
        self._log_every = max(1, log_every)
        self._episodes = 0
        self._caught = 0
        self._returns: list[float] = []
        self._min_dists: list[float] = []
        self._reasons: list[str] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos") or []:
            ep = info.get("episode")
            if not ep:
                continue
            self._episodes += 1
            self._caught += int(bool(ep.get("caught")))
            self._returns.append(float(ep.get("r", 0.0)))
            md = ep.get("min_dist")
            if md is not None and np.isfinite(md):
                self._min_dists.append(float(md))
            reason = str(ep.get("failure_reason") or "")
            if reason:
                self._reasons.append(reason)
            if self._episodes % self._log_every == 0:
                rate = self._caught / self._episodes
                mean_r = float(np.mean(self._returns[-self._log_every :]))
                mean_d = float(np.mean(self._min_dists[-self._log_every :])) if self._min_dists else float("nan")
                map_seed = ep.get("map_seed")
                seed_bit = f" map_seed={map_seed}" if map_seed is not None else ""
                stage = ep.get("curriculum_stage")
                gap_lo, gap_hi = ep.get("gap_lo"), ep.get("gap_hi")
                gap_bit = ""
                if stage is not None and gap_lo is not None and gap_hi is not None:
                    gap_bit = f" stage={int(stage)} gap={float(gap_lo):.0f}-{float(gap_hi):.0f}m"
                reason_bit = ""
                if self._reasons:
                    counts = Counter(self._reasons[-self._log_every :])
                    reason_bit = " " + ",".join(
                        f"{name}={n}" for name, n in counts.most_common(4)
                    )
                print(
                    f"[train] episodes={self._episodes} catch_rate={rate:.3f} "
                    f"mean_return={mean_r:.3f} mean_min_dist={mean_d:.2f}m"
                    f"{gap_bit}{reason_bit}{seed_bit}"
                )
                if self.logger is not None:
                    self.logger.record("rollout/catch_rate", rate)
                    self.logger.record("rollout/mean_min_dist", mean_d)
                    if stage is not None:
                        self.logger.record("rollout/curriculum_stage", float(stage))
        return True


def _package_submission(policy_path: Path, family_id: str, out_dir: Path) -> Path:
    pkg_dir = out_dir / "package"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parent / "agent_template.py", pkg_dir / "drone_agent.py")
    shutil.copy2(policy_path, pkg_dir / "ppo_policy.zip")

    submission_zip = out_dir / "submission.zip"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "swarm.cli",
            "model",
            "package",
            "--source",
            str(pkg_dir),
            "--family-id",
            family_id,
            "--output",
            str(submission_zip),
            "--overwrite",
        ],
        check=True,
    )
    return submission_zip


def _tensorboard_log_dir(out_dir: Path) -> str | None:
    """SB3 raises if tensorboard_log is set but the tensorboard package is missing."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("tensorboard not installed; continuing without TB logs "
              "(pip install tensorboard to enable).")
        return None
    return str(out_dir / "tb")


def _ppo_kwargs(family_id: str) -> dict:
    cfg = dict(_DEFAULT_PPO)
    if family_id == "cf_interceptor":
        cfg.update(_INTERCEPTOR_PPO)
    return cfg


def train_family(family_id: str, *, supports_drone_count: bool = False) -> None:
    default_steps = (
        DEFAULT_INTERCEPTOR_TIMESTEPS if family_id == "cf_interceptor" else DEFAULT_TIMESTEPS
    )
    parser = argparse.ArgumentParser(
        description=f"Train a baseline PPO model for {family_id} and package it."
    )
    parser.add_argument("--timesteps", type=int, default=default_steps)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="PPO RNG seed and map-seed-pool generator. Not a single map.",
    )
    interceptor_map_default = (
        BENCHMARK_TOTAL_SEED_COUNT if family_id == "cf_interceptor" else 0
    )
    parser.add_argument(
        "--map-seeds",
        type=int,
        default=interceptor_map_default,
        help=(
            "Cycle this many unique map seeds (validator uses "
            f"{BENCHMARK_TOTAL_SEED_COUNT}). 0 = a new random map every episode."
        ),
    )
    parser.add_argument(
        "--map-seed-file",
        type=Path,
        default=None,
        help="JSON list of map seeds (overrides --map-seeds). Use published epoch seeds if you have them.",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--lr", type=float, default=None, help="Override PPO learning rate.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override PPO rollout length.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a PPO zip.")
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Skip submission.zip packaging (faster train-only loops).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50_000,
        help="Checkpoint every N timesteps (0 disables).",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Interceptor: skip the 3–5 m warmup and always use 60–100 m gaps.",
    )
    if supports_drone_count:
        parser.add_argument(
            "--drones",
            type=int,
            default=DEFAULT_SWARM_DRONES,
            help="Fixed drone count during training (evaluation still varies 2-8).",
        )
    args = parser.parse_args()

    drones = getattr(args, "drones", None)
    if drones is not None and not SWARM_MIN_DRONES <= drones <= SWARM_MAX_DRONES:
        parser.error(f"--drones must be between {SWARM_MIN_DRONES} and {SWARM_MAX_DRONES}")
    if args.map_seeds < 0:
        parser.error("--map-seeds must be >= 0")

    out_dir = Path(__file__).resolve().parent / family_id / "out"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.map_seed_file is not None:
        map_seeds = _load_map_seed_file(args.map_seed_file)
        print(f"Loaded {len(map_seeds)} map seeds from {args.map_seed_file}")
    elif args.map_seeds > 0:
        map_seeds = _unique_map_seeds(args.map_seeds, random.Random(args.seed))
        print(
            f"Cycling {len(map_seeds)} unique map seeds through the "
            f"{family_id} benchmark template (validator uses "
            f"{BENCHMARK_TOTAL_SEED_COUNT}/epoch). --seed {args.seed} only "
            "seeds PPO and this pool."
        )
    else:
        map_seeds = None
        print(
            f"Unbounded random map seeds every episode for {family_id} "
            "(still uses the family benchmark template when one exists)."
        )

    if map_seeds:
        seed_path = out_dir / "map_seeds.json"
        seed_path.write_text(json.dumps({"seeds": map_seeds}, indent=2))
        print(f"Wrote map-seed pool to {seed_path}")

    env = FamilyVecEnv(
        family_id,
        seed=args.seed,
        n_drones=drones,
        map_seeds=map_seeds,
        curriculum=not args.no_curriculum,
    )
    ppo_kw = _ppo_kwargs(family_id)
    if args.lr is not None:
        ppo_kw["learning_rate"] = args.lr
    if args.n_steps is not None:
        ppo_kw["n_steps"] = args.n_steps
        # Keep batch_size a divisor of n_steps * n_envs
        n_envs = env.num_envs
        roll = ppo_kw["n_steps"] * n_envs
        ppo_kw["batch_size"] = min(int(ppo_kw.get("batch_size", 64)), roll)

    tb_log = _tensorboard_log_dir(out_dir)
    if args.resume is not None:
        model = PPO.load(str(args.resume), env=env, device=args.device, seed=args.seed)
        model.tensorboard_log = tb_log
        print(f"Resumed PPO from {args.resume}")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            seed=args.seed,
            verbose=1,
            device=args.device,
            tensorboard_log=tb_log,
            **ppo_kw,
        )

    if family_id == "cf_interceptor":
        gap_lo, gap_hi = interceptor_curriculum_gap(env._curriculum_stage)
        print(
            "Interceptor PPO: target_kl="
            f"{ppo_kw.get('target_kl')} lr={ppo_kw.get('learning_rate')} "
            f"n_epochs={ppo_kw.get('n_epochs')}; "
            f"curriculum_stage={env._curriculum_stage} gap={gap_lo:.0f}-{gap_hi:.0f}m "
            "(lead-aim + last-metre commit + crash penalty). "
            "Do not resume a pre-fix checkpoint."
        )

    callbacks: list[BaseCallback] = [EpisodeStatCallback(log_every=5)]
    if args.save_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.save_every // max(1, env.num_envs)),
                save_path=str(ckpt_dir),
                name_prefix="ppo",
            )
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume is None,
        tb_log_name=family_id,
    )

    policy_path = out_dir / "ppo_policy.zip"
    model.save(str(policy_path))
    env.close()

    if args.no_package:
        print(f"\nPolicy saved: {policy_path}")
        return

    submission_zip = _package_submission(policy_path, family_id, out_dir)
    smoke_ok, smoke_reason = smoke_test_policy_package(submission_zip)
    if not smoke_ok:
        raise RuntimeError(f"Packaged submission failed the contract smoke test: {smoke_reason}")

    print(f"\nSubmission ready: {submission_zip}")
    print("Test it like a validator:")
    print(f"  python3 RL/test_RL.py --model {submission_zip} --family_id {family_id}")
