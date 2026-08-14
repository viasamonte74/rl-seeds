"""Shared training harness for the per-family baseline starters.

Each RL/<family_id>/train.py calls train_family(): PPO is trained on real
generator tasks, saved, packaged into a validator-ready submission.zip, and
smoke-tested against the family's policy contract.

The policy sees a min-pooled depth frame plus the full state vector; rgb is
never requested. Interceptor uses 256×256 nearest-surface pool so a few-pixel
target at 60–100 m is not deleted by stride downsampling. Multi-drone families
train one shared policy by exposing each drone as one slot of a vectorized
environment over a single shared simulation.

Interceptor (cf_interceptor) uses longer rollouts, a larger MLP, and dense
closing-distance shaping. The validator score is sparse until a catch
(~3,000 steps at 50 Hz), so vanilla score-delta PPO barely learns.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from swarm.challenge_families import get_challenge_family
from swarm.constants import (
    BENCHMARK_TOTAL_SEED_COUNT,
    SIM_DT,
    SWARM_MAX_DRONES,
    SWARM_MIN_DRONES,
)
from swarm.domain_model import get_policy_interface_contract
from swarm.policy_interface import resolve_policy_interface_version, smoke_test_policy_package
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.task_gen import random_task

_MAX_MAP_SEED = 2**32 - 1

POLICY_DEPTH_SIZE = 64
# Interceptor HD depth is 1024² so a 36 cm target is a few pixels at 60–100 m.
# Stride-16 into 64² drops those pixels. 256² + nearest-surface pool keeps them.
INTERCEPTOR_DEPTH_SIZE = 256
DEFAULT_TIMESTEPS = 50_000
DEFAULT_INTERCEPTOR_TIMESTEPS = 500_000
DEFAULT_SWARM_DRONES = 4
_MAX_TASK_RESAMPLES = 200

# Closing 1 m of gap → this much extra reward. A 80 m close-in is ~+1.6,
# comparable to a catch score (0.5–1.0) so PPO is not dominated by shaping.
_INTERCEPTOR_CLOSE_REWARD_PER_M = 0.02
_INTERCEPTOR_SHAPE_CLIP = 0.15

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

class InterceptorFeaturesExtractor(BaseFeaturesExtractor):
    """Keep the search clue visible: CNN on NCHW depth, MLP on the 140-d state.

    SB3 CombinedExtractor treats float (256,256,1) as channels-first, so it
    builds Conv2d(256, ...) on a 256×1 strip and then concatenates a flattened
    image with the state. The clue (state[138:140]) is drowned and the camera
    is unused. This extractor permutes HWC→CHW and gives state its own MLP.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        depth_shape = observation_space["depth"].shape
        state_dim = int(observation_space["state"].shape[0])
        height, width = int(depth_shape[0]), int(depth_shape[1])
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_cnn = int(self.cnn(torch.zeros(1, 1, height, width)).shape[1])
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(n_cnn + 128, features_dim), nn.ReLU())

    def forward(self, observations):
        depth = observations["depth"]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2).contiguous()
        return self.merge(torch.cat([self.cnn(depth), self.state_mlp(observations["state"])], dim=1))


# Chase episodes are 60 s × 50 Hz = 3,000 steps. n_steps=512 only sees ~10 s
# of a pursuit; 2048 covers ~40 s so GAE can credit a catch. Slight entropy
# keeps the 5-D velocity command from collapsing to hover.
_INTERCEPTOR_PPO = dict(
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    ent_coef=0.01,
    learning_rate=3e-4,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        features_extractor_class=InterceptorFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    ),
)


def _policy_depth_size(family_id: str) -> int:
    if family_id == "cf_interceptor":
        return INTERCEPTOR_DEPTH_SIZE
    return POLICY_DEPTH_SIZE


def downsample_depth(depth: np.ndarray, size: int) -> np.ndarray:
    """Block min-pool depth to size×size×1.

    Env depth is 0=near, 1=far. Min-pool keeps the nearest surface in each
    block, so a 2–4 px interceptor target still marks the downsampled cell.
    Stride sampling (``depth[::k, ::k]``) skips those pixels and blinds the
    policy until the target is huge.
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
        out = cropped.reshape(size, bh, size, bw).min(axis=(1, 3))
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

    Interceptor episodes cycle a validator-sized map-seed pool through the
    family's 100-slot benchmark template (open maps, three chase-gap bands).
    ``--seed`` only seeds PPO and this pool; it is not a single map.
    """

    def __init__(
        self,
        family_id: str,
        *,
        seed: int,
        n_drones: int | None = None,
        map_seeds: list[int] | None = None,
    ):
        self._family_id = family_id
        self._family = get_challenge_family(family_id)
        self._rng = random.Random(seed)
        self._forced_drones = n_drones
        self._map_seeds = list(map_seeds) if map_seeds else None
        self._seed_cursor = 0
        self._last_map_seed: int | None = None
        self._env, first_obs = self._build_episode()
        self._depth_size = _policy_depth_size(family_id)
        self._prev_chase_dist: float | None = None
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
        return make_env_with_initial_obs(task)

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

    def _current_chase_dist(self) -> float | None:
        """Live chaser→target range. Do not use intercept_min_dist for this.

        intercept_min_dist is a running episode minimum: it only falls, never
        rises, so shaping on it is 0 on most steps and never punishes flying
        away. The target's own cruise can also lower the running min while the
        chaser hovers, which is the ~73 m plateau in the logs.
        """
        tpos = getattr(self._env, "_target_pos", None)
        if tpos is None:
            return None
        chaser = self._env._getDroneStateVector(0)
        cpos = np.asarray(chaser[0:3], dtype=np.float64)
        return float(np.linalg.norm(cpos - np.asarray(tpos, dtype=np.float64)))

    def _shape_interceptor_reward(self, base_reward: float, info: dict, done: bool) -> float:
        """Potential-based shaping on current range, plus validator score-delta.

        Until a catch, evaluate_rollout stays at participation (~0.01), so the
        score delta is ~0 for thousands of steps. Closing the live gap is the
        actual chase signal.
        """
        dist = self._current_chase_dist()
        if dist is None:
            raw = info.get("intercept_min_dist")
            dist = float(raw) if raw is not None and np.isfinite(raw) else None
        shaped = float(base_reward)
        if dist is not None and np.isfinite(dist):
            if self._prev_chase_dist is not None:
                delta = self._prev_chase_dist - dist
                shaped += float(
                    np.clip(
                        _INTERCEPTOR_CLOSE_REWARD_PER_M * delta,
                        -_INTERCEPTOR_SHAPE_CLIP,
                        _INTERCEPTOR_SHAPE_CLIP,
                    )
                )
            self._prev_chase_dist = dist
        if done:
            self._prev_chase_dist = None
        return shaped

    def reset(self):
        self._prev_chase_dist = None
        self._ep_return = 0.0
        self._ep_len = 0
        return self._stack_obs(self._last_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        env_action = self._pending_actions
        if self.num_envs == 1:
            env_action = env_action.reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(env_action)
        done = bool(terminated or truncated)
        info = info or {}

        shaped = float(reward)
        if self._family_id == "cf_interceptor":
            shaped = self._shape_interceptor_reward(shaped, info, done)

        self._ep_return += shaped
        self._ep_len += 1

        rewards = np.full(self.num_envs, shaped, dtype=np.float32)
        dones = np.full(self.num_envs, done, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        if done:
            terminal = self._stack_obs(obs)
            ep_stats = {
                "r": float(self._ep_return),
                "l": int(self._ep_len),
                "caught": bool(info.get("intercept_caught", terminated and not truncated)),
                "min_dist": float(info.get("intercept_min_dist", np.nan)),
                "map_seed": self._last_map_seed,
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
            self._prev_chase_dist = None
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
            if self._episodes % self._log_every == 0:
                rate = self._caught / self._episodes
                mean_r = float(np.mean(self._returns[-self._log_every :]))
                mean_d = float(np.mean(self._min_dists[-self._log_every :])) if self._min_dists else float("nan")
                map_seed = ep.get("map_seed")
                seed_bit = f" map_seed={map_seed}" if map_seed is not None else ""
                print(
                    f"[train] episodes={self._episodes} catch_rate={rate:.3f} "
                    f"mean_return={mean_r:.3f} mean_min_dist={mean_d:.2f}m{seed_bit}"
                )
                if self.logger is not None:
                    self.logger.record("rollout/catch_rate", rate)
                    self.logger.record("rollout/mean_min_dist", mean_d)
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

    env = FamilyVecEnv(family_id, seed=args.seed, n_drones=drones, map_seeds=map_seeds)
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
