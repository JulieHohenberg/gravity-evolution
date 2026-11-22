"""Train a walker controller with Stable-Baselines3 PPO."""

import argparse
import json
import tempfile
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from simulation import FINISH_BONUS, GRAVITY, MOTOR_PENALTY_SCALE, RACE_STEPS, SEED, SPEED_BONUS, make_environment

TOTAL_STEPS = 1_000_000  # TOTAL number of simulation steps used for training (amount of practice steps robot gets while PPO repeatedly updates its controller).
ROLLOUT_STEPS = 2048  # Steps collected before each PPO update.
SAVE_EVERY = 20_480  # Steps between checkpoints and evaluations (once every 10 PPO updates, since each update uses 2048 steps).


class WalkerObservations(gym.ObservationWrapper):
    """override built-in class to give PPO the robot's sensor readings instead of CARL's full context""" 
    def __init__(self, env):
        """Set up the sensor readings as the observation space."""
        super().__init__(env)
        self.observation_space = env.observation_space["obs"]

    def observation(self, observation):
        """Return the 24 readings used by the genetic controller."""
        return observation["obs"]

    def reset(self, **kwargs):
        """Use the same starting course for each run."""
        kwargs["seed"] = SEED
        return super().reset(**kwargs)


def train(planet, steps):
    """Train PPO, saving checkpoints and evaluation scores along the way."""
    Path("runs").mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"ppo-{planet.lower()}-", dir="runs"))
    (run_dir / "settings.json").write_text(json.dumps({
        "algorithm": "ppo", "reward_version": "finish-and-speed-v1", "planet": planet, "gravity": GRAVITY[planet],
        "seed": SEED, "requested_steps": steps, "rollout_steps": ROLLOUT_STEPS,
        "checkpoint_interval": SAVE_EVERY,
        "motor_penalty_scale": MOTOR_PENALTY_SCALE, "finish_bonus": FINISH_BONUS,
        "speed_bonus_per_step": SPEED_BONUS, "bonus_step_limit": RACE_STEPS,
    }, indent=2) + "\n")
    print(f"saving PPO results to {run_dir}", flush=True)
    environment = Monitor(WalkerObservations(make_environment(planet)), str(run_dir / "training"))
    evaluation = Monitor(WalkerObservations(make_environment(planet)))
    try:
        model = PPO("MlpPolicy", environment, seed=SEED, n_steps=ROLLOUT_STEPS, device="cpu", verbose=1)
        callbacks = [
            CheckpointCallback(save_freq=SAVE_EVERY, save_path=str(run_dir), name_prefix="checkpoint"),
            EvalCallback(evaluation, best_model_save_path=str(run_dir), log_path=str(run_dir),
                         eval_freq=SAVE_EVERY, n_eval_episodes=1, deterministic=True),
        ]

        # run the FULL PPO ALGO for the specified number of steps, saving checkpoints and evaluation scores along the way
        model.learn(total_timesteps=steps, callback=callbacks)
        model.save(run_dir / "final-model")
    finally:
        environment.close()
        evaluation.close()
    print(f"saved {run_dir / 'final-model.zip'}")


def main():
    """Read training options and start PPO."""
    parser = argparse.ArgumentParser(description="Train a walking controller with PPO.")
    parser.add_argument("--planet", choices=GRAVITY, default="Earth")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be at least 1")
    train(args.planet, args.steps)


if __name__ == "__main__":
    main()
