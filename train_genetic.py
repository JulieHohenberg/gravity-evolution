"""Train a walker controller using a genetic algorithm."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from controller import WEIGHT_COUNT, motor_commands
from simulation import FINISH_BONUS, GRAVITY, MOTOR_PENALTY_SCALE, RACE_STEPS, SEED, SPEED_BONUS, make_environment

POPULATION = 30  # Number of controllers tested per generation.
GENERATIONS = 30  # Number of generations (rounds of testing and breeding).
SURVIVORS = 6  # Top controllers chosen as parents for the next generation.
MAX_STEPS = 1600  # Maximum simulation steps for one controller's run.


def evaluate(environment, weights):
    """Run one controller and add up its rewards."""
    # Observation contains the robot's sensor readings.
    observation, _ = environment.reset(seed=SEED)
    score = 0.0
    for _ in range(MAX_STEPS):

        # run the simulation for one step with the given motor commands (calculated from weights in the controller)
        observation, reward, terminated, truncated, _ = environment.step( # Reward scores this step; terminated means the run ended in the simulation. Truncated means it stopped due to a time limit or another outside limit.
            motor_commands(observation, weights) # run motor commands with the given weights
        )

        score += reward
        if terminated or truncated:
            break
    return score


def train(planet, generations):
    """Evolve controllers and save the one with the best score."""
    Path("runs").mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"genetic-{planet.lower()}-", dir="runs"))
    (run_dir / "settings.json").write_text(json.dumps({
        "algorithm": "genetic", "reward_version": "finish-and-speed-v1", "planet": planet, "gravity": GRAVITY[planet],
        "seed": SEED, "generations": generations, "population": POPULATION,
        "survivors": SURVIVORS, "max_steps": MAX_STEPS,
        "mutation_probability": 0.1, "mutation_std": 0.1,
        "motor_penalty_scale": MOTOR_PENALTY_SCALE, "finish_bonus": FINISH_BONUS,
        "speed_bonus_per_step": SPEED_BONUS, "bonus_step_limit": RACE_STEPS,
    }, indent=2) + "\n")
    print(f"saving generation results to {run_dir}", flush=True)
    rng = np.random.default_rng(SEED)
    population = rng.normal(0, 0.3, (POPULATION, WEIGHT_COUNT))
    environment = make_environment(planet)
    best_score = float("-inf")
    best = None
    try:
        # run the genetic algorithm for the specified number of generations 
        # -> for each generation, evaluate the current population of controllers, select the best ones, and create a new population by breeding them together
        for generation in range(generations):
            # get all of the scores for the current population of controllers (in current generation)
            scores = np.array([evaluate(environment, weights) for weights in population])
            # rank the scores
            ranked = np.argsort(scores)[::-1]
            # if the best score in this generation is better than the best score overall, update the best score and best weights
            if scores[ranked[0]] > best_score:
                best_score = scores[ranked[0]]
                best = population[ranked[0]].copy()
            # Save each generation immediately so its results survive an interruption.
            np.save(run_dir / f"generation-{generation + 1:03d}.npy", population[ranked[0]])
            with (run_dir / "scores.csv").open("a", newline="") as file:
                writer = csv.writer(file)
                if generation == 0:
                    writer.writerow(["generation", "best", "mean", "worst", "best_overall"])
                writer.writerow([generation + 1, scores[ranked[0]], scores.mean(), scores[ranked[-1]], best_score])
            with (run_dir / "population-scores.csv").open("a", newline="") as file:
                writer = csv.writer(file)
                if generation == 0:
                    writer.writerow(["generation", "controller", "score"])
                writer.writerows((generation + 1, index + 1, score) for index, score in enumerate(scores))
            print(f"generation {generation + 1}: best score {best_score:.1f}", flush=True)
            parents = population[ranked[:SURVIVORS]]
            children = [best.copy()] # elitism: carry the best controller to the next generation
            while len(children) < POPULATION:
                # randomly pick two of the six best controllers as parents
                first, second = parents[rng.integers(SURVIVORS, size=2)]
                # BREEDING: create a child controller by randomly choosing weights from the two parents
                child = np.where(rng.random(WEIGHT_COUNT) < 0.5, first, second)
                # MUTATION: randomly change some of the weights in the child controller to introduce new variations
                child += rng.normal(0, 0.1, WEIGHT_COUNT) * (rng.random(WEIGHT_COUNT) < 0.1) # 10% chance of mutation for each weight (small random adjustment)
                children.append(child)
            # reefine population to be the new generation of controllers
            population = np.array(children)
    finally:
        environment.close()

    path = Path(f"controller-{planet.lower()}.npy")
    np.save(path, best)
    print(f"saved {path}")


def main():
    """Read training options and start the genetic algorithm."""
    parser = argparse.ArgumentParser(description="Evolve a walking controller.")
    parser.add_argument("--planet", choices=GRAVITY, default="Earth")
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--overwrite", action="store_true", help="replace saved weights")
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("generations must be at least 1")
    if Path(f"controller-{args.planet.lower()}.npy").exists() and not args.overwrite:
        parser.error("saved controller already exists; use --overwrite to replace it")
    train(args.planet, args.generations)


if __name__ == "__main__":
    main()
