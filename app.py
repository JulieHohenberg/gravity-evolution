"""API for racing trained controllers under Earth or Moon gravity."""

import os
from pathlib import Path
from threading import Lock
from typing import Literal

# The API runs the simulator without opening a desktop window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from stable_baselines3 import PPO

from controller import WEIGHT_COUNT, motor_commands
from simulation import GRAVITY, SEED, make_environment

ROOT = Path(__file__).resolve().parent
MAX_STEPS = 1600  # Match the maximum length of a genetic training run.
FRAME_INTERVAL = 10  # Record positions every 10 simulation steps.
race_lock = Lock()  # CARL changes shared physics settings while creating worlds.

app = FastAPI(title="Gravity Evolution API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ModelId = Literal["genetic-earth", "genetic-moon", "ppo-earth", "ppo-moon"]


class RaceInput(BaseModel):
    """Choose the race environment and one to four different controllers."""

    planet: Literal["Earth", "Moon"]
    models: list[ModelId] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_models(self):
        """Avoid racing the same controller twice by mistake."""
        if len(set(self.models)) != len(self.models):
            raise ValueError("Choose each model only once.")
        return self


def model_path(model_id):
    """Find the genetic weights or the newest completed PPO model."""
    algorithm, planet = model_id.split("-")
    if algorithm == "genetic":
        path = ROOT / f"controller-{planet}.npy"
        return path if path.is_file() else None
    paths = list((ROOT / "runs").glob(f"ppo-{planet}-*/final-model.zip"))
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path))) if paths else None


def load_controller(model_id, path):
    """Load a trusted local controller for the race."""
    if model_id.startswith("ppo-"):
        return PPO.load(path, device="cpu")
    weights = np.load(path, allow_pickle=False)
    if weights.shape != (WEIGHT_COUNT,) or not np.isfinite(weights).all():
        raise ValueError("Invalid genetic controller weights")
    return weights


def snapshot(walker, step, start_x):
    """Save the body outlines and distance for replaying this step."""
    bodies = [walker.hull, *walker.legs]
    return {
        "step": step,
        "distance": round(float(walker.hull.position.x - start_x), 4),
        "body": [
            [[round(float(point[0]), 4), round(float(point[1]), 4)]
             for point in (body.transform * vertex for vertex in body.fixtures[0].shape.vertices)]
            for body in bodies
        ],
    }


def run_controller(planet, model_id, path):
    """Run one controller on the same seeded course and record its movement."""
    controller = load_controller(model_id, path)
    environment = make_environment(planet)
    try:
        observation, _ = environment.reset(seed=SEED)
        walker = environment.env.unwrapped
        start_x = float(walker.hull.position.x)
        terrain = [[round(float(point[0]), 4), round(float(point[1]), 4)]
                   for polygon, _ in walker.terrain_poly for point in polygon[:1]]
        terrain.append([round(float(walker.terrain_poly[-1][0][1][0]), 4),
                        round(float(walker.terrain_poly[-1][0][1][1]), 4)])
        frames = [snapshot(walker, 0, start_x)]
        reward_total = 0.0
        status = "time_limit"
        for step in range(1, MAX_STEPS + 1):
            if model_id.startswith("ppo-"):
                action, _ = controller.predict(observation["obs"], deterministic=True)
            else:
                action = motor_commands(observation, controller)
            observation, reward, terminated, truncated, _ = environment.step(action)
            reward_total += float(reward)
            if step % FRAME_INTERVAL == 0 or terminated or truncated or step == MAX_STEPS:
                frames.append(snapshot(walker, step, start_x))
            if terminated or truncated:
                if terminated:
                    status = "fell" if walker.game_over or walker.hull.position.x < 0 else "finished"
                break
        return {
            "model": model_id,
            "model_file": str(path.relative_to(ROOT)),
            "status": status,
            "steps": step,
            "time_seconds": round(step / 50, 2),
            "distance": frames[-1]["distance"],
            "reward": round(reward_total, 4),
            "terrain": terrain,
            "frames": frames,
        }
    finally:
        environment.close()


@app.get("/models")
def list_models():
    """List the four controllers and whether their saved files are available."""
    models = []
    for model_id in ("genetic-earth", "genetic-moon", "ppo-earth", "ppo-moon"):
        path = model_path(model_id)
        models.append({"id": model_id, "available": path is not None,
                       "file": str(path.relative_to(ROOT)) if path else None})
    return {"models": models}


@app.post("/race")
def race(input: RaceInput):
    """Race the selected controllers and return results and replay frames."""
    paths = {model_id: model_path(model_id) for model_id in input.models}
    missing = [model_id for model_id, path in paths.items() if path is None]
    if missing:
        raise HTTPException(status_code=404, detail=f"Saved model not found: {', '.join(missing)}")
    if not race_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A race is already running. Try again shortly.")
    try:
        racers = [run_controller(input.planet, model_id, paths[model_id]) for model_id in input.models]
    finally:
        race_lock.release()
    furthest = max(racer["distance"] for racer in racers)
    return {"planet": input.planet, "gravity": GRAVITY[input.planet], "seed": SEED,
            "steps_per_second": 50, "frame_interval": FRAME_INTERVAL,
            "winners": [racer["model"] for racer in racers if racer["distance"] == furthest],
            "racers": racers}
