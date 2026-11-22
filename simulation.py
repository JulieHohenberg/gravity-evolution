"""Explore the same CARL walking robot under Earth and Moon gravity."""

import numpy as np
import pygame
from carl.envs.gymnasium.box2d.carl_bipedal_walker import CARLBipedalWalker

# Gravity is measured in meters per second squared. Negative means downward.
GRAVITY = {"Earth": -9.81, "Moon": -1.62}
SEED = 42  # Reusing a seed gives both environments the same generated terrain.


def make_environment(planet):
    # A context tells CARL which properties to change from its defaults.
    # Each environment has its own context, so gravity stays fixed during a run.
    return CARLBipedalWalker(contexts={0: {"GRAVITY_Y": GRAVITY[planet]}})


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 540))
    pygame.display.set_caption("Gravity Evolution")
    font = pygame.font.SysFont(None, 25)
    clock = pygame.time.Clock()

    environments = {planet: make_environment(planet) for planet in GRAVITY}
    planet = "Earth"
    environment = environments[planet]
    environment.reset(seed=SEED)
    paused = False
    finished = False
    running = True

    # No brain exists yet: zero means none of the four motors is powered.
    action = np.zeros(4, dtype=np.float32)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_1, pygame.K_2):
                    planet = "Earth" if event.key == pygame.K_1 else "Moon"
                    environment = environments[planet]
                    environment.reset(seed=SEED)
                    finished = False
                    paused = False
                elif event.key == pygame.K_r:
                    environment.reset(seed=SEED)
                    finished = False
                    paused = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # One step advances the Box2D physics by 1/50 of a simulated second.
        if not paused and not finished:
            _, _, terminated, truncated, _ = environment.step(action)
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
            "Motors off — no training yet" + ("   |   Paused" if paused else ""),
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
