"""Explore the same CARL walking robot under Earth and Moon gravity."""

import argparse

import gymnasium as gym
import numpy as np
import pygame
from carl.envs.gymnasium.box2d.carl_bipedal_walker import CARLBipedalWalker
from controller import WEIGHT_COUNT, motor_commands

# Gravity is measured in meters per second squared. Negative means downward.
GRAVITY = {"Earth": -9.81, "Moon": -1.62}
SEED = 42  # Reusing a seed gives both environments the same generated terrain.
MOTOR_PENALTY_SCALE = 0.25  # Keep 25% of the simulator's motor penalty.
FINISH_BONUS = 100  # Points for reaching the finish line.
SPEED_BONUS = 0.1  # Extra points for each unused step out of 1,600.
RACE_STEPS = 1600


class WalkingRewards(gym.Wrapper):
    """Adjust rewards equally for genetic and PPO training."""

    def reset(self, **kwargs):
        """Start counting steps for a new run."""
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Reduce the motor penalty and reward reaching the finish quickly."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        walker = self.env.unwrapped
        failed = walker.game_over or walker.hull.position.x < 0
        if not failed:
            # The simulator replaces the whole reward with -100 on a fall.
            original_cost = 0.00035 * self.env.context["MOTORS_TORQUE"] * np.clip(np.abs(action), 0, 1).sum()
            reward += (1 - MOTOR_PENALTY_SCALE) * original_cost
            finish_x = (self.env.context["TERRAIN_LENGTH"] - self.env.context["TERRAIN_GRASS"]) * self.env.context["TERRAIN_STEP"]
            if terminated and walker.hull.position.x > finish_x:
                reward += FINISH_BONUS + SPEED_BONUS * max(0, RACE_STEPS - self.steps)
        return observation, float(reward), terminated, truncated, info


def make_environment(planet):
    """Create a walker with the chosen gravity and shared reward rules."""
    # Keep all settings available when CARL selects the context again on reset.
    context = CARLBipedalWalker.get_default_context()
    context["GRAVITY_Y"] = GRAVITY[planet]
    return WalkingRewards(CARLBipedalWalker(contexts={0: context}))


def main():
    """Display the robot with an optional trained controller."""
    parser = argparse.ArgumentParser(description="Watch the walking robot.")
    controllers = parser.add_mutually_exclusive_group()
    controllers.add_argument("--controller", help="path to genetic controller weights")
    controllers.add_argument("--ppo", help="path to a trained PPO model")
    args = parser.parse_args()
    weights = np.load(args.controller, allow_pickle=False) if args.controller else None
    if weights is not None and weights.shape != (WEIGHT_COUNT,):
        parser.error(f"controller must contain {WEIGHT_COUNT} weights")
    model = None
    if args.ppo:
        from stable_baselines3 import PPO
        model = PPO.load(args.ppo, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((600, 540))
    pygame.display.set_caption("Gravity Evolution")
    font = pygame.font.SysFont(None, 25)
    clock = pygame.time.Clock()

    environments = {planet: make_environment(planet) for planet in GRAVITY}
    planet = "Earth"
    environment = environments[planet]
    observation, _ = environment.reset(seed=SEED)
    paused = False
    finished = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_1, pygame.K_2):
                    planet = "Earth" if event.key == pygame.K_1 else "Moon"
                    environment = environments[planet]
                    observation, _ = environment.reset(seed=SEED)
                    finished = False
                    paused = False
                elif event.key == pygame.K_r:
                    observation, _ = environment.reset(seed=SEED)
                    finished = False
                    paused = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # One step advances the Box2D physics by 1/50 of a simulated second.
        if not paused and not finished:
            if model is not None:
                action, _ = model.predict(observation["obs"], deterministic=True)
            else:
                action = motor_commands(observation, weights) if weights is not None else np.zeros(4, dtype=np.float32)
            observation, _, terminated, truncated, _ = environment.step(action)
            finished = terminated or truncated

        # CARL returns an image as an array. Pygame displays it in our window.
        image = environment.render()
        if planet == "Moon":
            # Replace the sky and ground colors to give the Moon a darker look.
            for original, replacement in (
                ((215, 215, 255), (25, 25, 35)),
                ((255, 255, 255), (25, 25, 35)),
                ((102, 153, 76), (115, 115, 115)),
                ((76, 255, 76), (160, 160, 160)),
                ((76, 204, 76), (140, 140, 140)),
            ):
                image[np.all(image == original, axis=2)] = replacement
        picture = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        screen.fill((24, 31, 44))
        screen.blit(picture, (0, 0))
        lines = [
            f"{planet}   |   Gravity: {abs(GRAVITY[planet])} m/s² downward",
            "1: Earth    2: Moon    R: reset    Space: pause",
            ("Controller active" if weights is not None or model is not None else "Motors off") + ("   |   Paused" if paused else ""),
            "Run ended — press R to restart" if finished else "Esc: quit",
        ]
        for index, line in enumerate(lines):
            screen.blit(font.render(line, True, (245, 245, 245)), (15, 410 + index * 29))
        pygame.display.flip()
        clock.tick(50)

    for environment in environments.values():
        environment.close()
    pygame.quit()


if __name__ == "__main__":
    main()
