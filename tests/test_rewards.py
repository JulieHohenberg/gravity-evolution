import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from simulation import WalkingRewards, make_environment


class FakeWalker(gym.Env):
    """Provide controlled outcomes for testing the reward changes."""

    def __init__(self):
        """Start with a walker that is upright and away from the finish."""
        self.context = {"MOTORS_TORQUE": 80, "TERRAIN_LENGTH": 200,
                        "TERRAIN_GRASS": 10, "TERRAIN_STEP": 14 / 30}
        self.hull = SimpleNamespace(position=SimpleNamespace(x=5.0))
        self.game_over = False
        self.terminated = False
        self.reward = 0.0

    def reset(self, **kwargs):
        """Return a simple starting observation."""
        return {}, {}

    def step(self, action):
        """Return the configured simulator outcome."""
        return {}, self.reward, self.terminated, False, {}


class RewardTest(unittest.TestCase):
    def test_rewards_work_after_repeated_resets(self):
        """Keep reward settings available for every controller's run."""
        for planet in ("Earth", "Moon"):
            env = make_environment(planet)
            try:
                for _ in range(3):
                    env.reset(seed=42)
                    _, reward, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
                    self.assertTrue(np.isfinite(reward))
            finally:
                env.close()

    def test_motor_penalty_is_reduced(self):
        """Keep one quarter of the simulator's motor cost."""
        base = FakeWalker()
        base.reward = -0.112
        env = WalkingRewards(base)
        env.reset()
        self.assertAlmostEqual(env.step(np.ones(4))[1], -0.028)

    def test_fall_remains_minus_100(self):
        """Don't refund motor costs or award a finish bonus after a fall."""
        base = FakeWalker()
        base.reward = -100
        base.game_over = True
        base.terminated = True
        env = WalkingRewards(base)
        env.reset()
        self.assertEqual(env.step(np.ones(4))[1], -100)

    def test_finish_bonus_depends_on_time_and_resets(self):
        """Award a larger bonus for finishing in fewer steps."""
        base = FakeWalker()
        base.hull.position.x = 90
        base.terminated = True
        env = WalkingRewards(base)
        env.reset()
        env.steps = 999
        self.assertAlmostEqual(env.step(np.zeros(4))[1], 160)
        env.reset()
        env.steps = 1499
        self.assertAlmostEqual(env.step(np.zeros(4))[1], 110)

    def test_early_termination_is_not_a_finish(self):
        """Only crossing the finish line earns the bonus."""
        base = FakeWalker()
        base.terminated = True
        env = WalkingRewards(base)
        env.reset()
        self.assertEqual(env.step(np.zeros(4))[1], 0)


if __name__ == "__main__":
    unittest.main()
