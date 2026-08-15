"""SAC training harness for cf_interceptor.

PPO never saw a catch, so it polished "point at the clue and timeout." SAC
reuses rare catches in a replay buffer. A privileged lead-pursuit teacher is
mixed into training rollouts only, then annealed to 0. The packaged agent
(RL/agent_sac.py) is the neural net — the teacher is never submitted.

Do not resume a PPO zip. Start at 1 map seed until catch_rate > 0, then expand.
Replay is capped: 256² depth makes a large buffer unusable.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from agent_sac import InterceptorFeaturesExtractor
from common import (
    DEFAULT_INTERCEPTOR_TIMESTEPS,
    DEFAULT_SWARM_DRONES,
    EpisodeStatCallback,
    FamilyVecEnv,
    _INTERCEPTOR_RAM_M,
    _load_map_seed_file,
    _tensorboard_log_dir,
    _unique_map_seeds,
    interceptor_curriculum_gap,
    lead_aim_offset,
)
from swarm.constants import (
    BENCHMARK_TOTAL_SEED_COUNT,
    SWARM_MAX_DRONES,
    SWARM_MIN_DRONES,
)
from swarm.policy_interface import smoke_test_policy_package

# 256² float depth ≈ 0.25 MB/frame; SB3 stores obs + next_obs. 12k ≈ 6 GB.
_INTERCEPTOR_SAC = dict(
    learning_rate=3e-4,
    buffer_size=12_000,
    learning_starts=3_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=64,
    gradient_steps=64,
    ent_coef="auto",
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        features_extractor_class=InterceptorFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    ),
)


def expert_vel_action(
    offset: np.ndarray,
    target_vel: np.ndarray | None = None,
    *,
    lead: bool = False,
) -> np.ndarray:
    """Privileged VEL command: fly at ``offset`` (optionally lead) at full speed."""
    off = np.asarray(offset, dtype=np.float64).reshape(-1)
    if off.size < 2:
        return np.zeros(5, dtype=np.float32)
    if lead:
        off = lead_aim_offset(off, target_vel)
    xy = off[:2]
    nxy = float(np.linalg.norm(xy))
    if off.size >= 3:
        direction = off[:3]
    else:
        direction = np.array([xy[0], xy[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        dir3 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        dir3 = direction / norm
    yaw = 0.0 if nxy < 1e-6 else float(np.arctan2(xy[1], xy[0]) / np.pi)
    return np.array(
        [dir3[0], dir3[1], dir3[2], 1.0, float(np.clip(yaw, -1.0, 1.0))],
        dtype=np.float32,
    )


def expert_mix_prob(stage: int, dist: float | None) -> float:
    """Intercept-first mix. Anneals to 0 so the submitted policy is the net."""
    ram = dist is not None and float(dist) <= _INTERCEPTOR_RAM_M
    stage = int(stage)
    if stage <= 0:
        return 0.7 if ram else 0.2
    if stage <= 2:
        return 0.3 if ram else 0.1
    if stage <= 4:
        return 0.1 if ram else 0.0
    return 0.0


class SacInterceptorVecEnv(FamilyVecEnv):
    """FamilyVecEnv plus a train-only privileged teacher mixed into actions."""

    def __init__(self, *args, expert_mix: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._expert_mix = bool(expert_mix) and self._family_id == "cf_interceptor"
        self._mix_rng = random.Random(int(kwargs.get("seed", 0)) + 17)
        self._ep_expert_steps = 0
        self._ep_total_steps = 0

    def _maybe_mix_expert(self) -> None:
        if not self._expert_mix or self._pending_actions is None:
            return
        delta, dist, vel = self._chase_geometry()
        p = expert_mix_prob(self._curriculum_stage, dist)
        self._ep_total_steps += 1
        if self._mix_rng.random() >= p:
            return
        if dist is not None and dist <= _INTERCEPTOR_RAM_M and delta is not None:
            expert = expert_vel_action(delta, vel, lead=True)
        else:
            state = None
            if isinstance(self._last_obs, dict):
                state = self._last_obs.get("state")
            if state is None:
                return
            vec = np.asarray(state, dtype=np.float64).reshape(-1)
            if vec.size < 2:
                return
            expert = expert_vel_action(vec[-2:], lead=False)
        act = np.asarray(self._pending_actions, dtype=np.float32)
        if act.ndim == 1:
            self._pending_actions = expert
        else:
            act = act.copy()
            act[0] = expert
            self._pending_actions = act
        self._ep_expert_steps += 1

    def step_wait(self):
        self._maybe_mix_expert()
        expert_steps = self._ep_expert_steps
        total_steps = self._ep_total_steps
        obs, rewards, dones, infos = super().step_wait()
        if bool(np.any(dones)):
            frac = (expert_steps / total_steps) if total_steps else 0.0
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    ep["expert_frac"] = float(frac)
            self._ep_expert_steps = 0
            self._ep_total_steps = 0
        return obs, rewards, dones, infos


class SacEpisodeStatCallback(EpisodeStatCallback):
    """Same catch/min-dist log, plus teacher mix fraction."""

    def __init__(self, log_every: int = 5):
        super().__init__(log_every=log_every)
        self._expert_fracs: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos") or []:
            ep = info.get("episode")
            if not ep:
                continue
            frac = ep.get("expert_frac")
            if frac is not None:
                self._expert_fracs.append(float(frac))
        ok = super()._on_step()
        if self._episodes > 0 and self._episodes % self._log_every == 0 and self._expert_fracs:
            mean_f = float(np.mean(self._expert_fracs[-self._log_every :]))
            print(f"[train] expert_frac={mean_f:.2f}")
            if self.logger is not None:
                self.logger.record("rollout/expert_frac", mean_f)
        return ok


def _package_sac_submission(policy_path: Path, family_id: str, out_dir: Path) -> Path:
    pkg_dir = out_dir / "package"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parent / "agent_sac.py", pkg_dir / "drone_agent.py")
    shutil.copy2(policy_path, pkg_dir / "sac_policy.zip")

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


def _looks_like_ppo_zip(path: Path) -> bool:
    name = path.name.lower()
    return "ppo" in name and "sac" not in name


def train_family_sac(family_id: str, *, supports_drone_count: bool = False) -> None:
    if family_id != "cf_interceptor":
        raise SystemExit("common_sac.py trains cf_interceptor only.")

    parser = argparse.ArgumentParser(
        description="Train a SAC interceptor (teacher mix in training only)."
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_INTERCEPTOR_TIMESTEPS)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="SAC RNG seed and map-seed-pool generator. Not a single map.",
    )
    parser.add_argument(
        "--map-seeds",
        type=int,
        default=1,
        help=(
            "Start at 1 seed until catch_rate>0, then resume with more. "
            f"Validator uses {BENCHMARK_TOTAL_SEED_COUNT}. 0 = new map every episode."
        ),
    )
    parser.add_argument("--map-seed-file", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume a SAC zip only.")
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Skip the 3–5 m intercept warmup (60–100 m from the start).",
    )
    parser.add_argument(
        "--no-expert",
        action="store_true",
        help="Disable the privileged teacher mix (not recommended until catch_rate>0).",
    )
    if supports_drone_count:
        parser.add_argument("--drones", type=int, default=DEFAULT_SWARM_DRONES)
    args = parser.parse_args()

    drones = getattr(args, "drones", None)
    if drones is not None and not SWARM_MIN_DRONES <= drones <= SWARM_MAX_DRONES:
        parser.error(f"--drones must be between {SWARM_MIN_DRONES} and {SWARM_MAX_DRONES}")
    if args.map_seeds < 0:
        parser.error("--map-seeds must be >= 0")
    if args.map_seeds > 20:
        print(
            "Warning: --map-seeds>20 before catch_rate>0 repeats the PPO deadlock. "
            "Use 1 seed until the first catches, then resume with more."
        )
    if args.resume is not None and _looks_like_ppo_zip(args.resume):
        parser.error(
            f"{args.resume} looks like a PPO checkpoint. SAC cannot load it. "
            "Train SAC from scratch."
        )

    out_dir = Path(__file__).resolve().parent / family_id / "out_sac"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.map_seed_file is not None:
        map_seeds = _load_map_seed_file(args.map_seed_file)
        print(f"Loaded {len(map_seeds)} map seeds from {args.map_seed_file}")
    elif args.map_seeds > 0:
        map_seeds = _unique_map_seeds(args.map_seeds, random.Random(args.seed))
        print(
            f"Cycling {len(map_seeds)} unique map seeds. --seed {args.seed} only "
            "seeds SAC and this pool. Expand seeds only after catch_rate>0."
        )
    else:
        map_seeds = None
        print(f"Unbounded random map seeds every episode for {family_id}.")

    if map_seeds:
        seed_path = out_dir / "map_seeds.json"
        seed_path.write_text(json.dumps({"seeds": map_seeds}, indent=2))
        print(f"Wrote map-seed pool to {seed_path}")

    env = SacInterceptorVecEnv(
        family_id,
        seed=args.seed,
        n_drones=drones,
        map_seeds=map_seeds,
        curriculum=not args.no_curriculum,
        expert_mix=not args.no_expert,
    )
    sac_kw = dict(_INTERCEPTOR_SAC)
    if args.lr is not None:
        sac_kw["learning_rate"] = args.lr

    tb_log = _tensorboard_log_dir(out_dir)
    if args.resume is not None:
        model = SAC.load(str(args.resume), env=env, device=args.device)
        model.tensorboard_log = tb_log
        print(f"Resumed SAC from {args.resume}")
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            seed=args.seed,
            verbose=1,
            device=args.device,
            tensorboard_log=tb_log,
            **sac_kw,
        )

    gap_lo, gap_hi = interceptor_curriculum_gap(env._curriculum_stage)
    print(
        "Interceptor SAC: "
        f"buffer={sac_kw.get('buffer_size')} gamma={sac_kw.get('gamma')} "
        f"lr={sac_kw.get('learning_rate')} mix={'off' if args.no_expert else 'on'}; "
        f"curriculum_stage={env._curriculum_stage} gap={gap_lo:.0f}-{gap_hi:.0f}m "
        "(teacher lead-aim in training only). Do not resume a PPO checkpoint."
    )

    callbacks: list[BaseCallback] = [SacEpisodeStatCallback(log_every=5)]
    if args.save_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.save_every // max(1, env.num_envs)),
                save_path=str(ckpt_dir),
                name_prefix="sac",
            )
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume is None,
        tb_log_name=f"{family_id}_sac",
    )

    policy_path = out_dir / "sac_policy.zip"
    model.save(str(policy_path))
    env.close()

    if args.no_package:
        print(f"\nPolicy saved: {policy_path}")
        return

    submission_zip = _package_sac_submission(policy_path, family_id, out_dir)
    smoke_ok, smoke_reason = smoke_test_policy_package(submission_zip)
    if not smoke_ok:
        raise RuntimeError(f"Packaged submission failed the contract smoke test: {smoke_reason}")

    print(f"\nSubmission ready: {submission_zip}")
    print("Test it like a validator:")
    print(f"  python3 RL/test_RL.py --model {submission_zip} --family_id {family_id}")
