import unittest

import numpy as np

from controller import WEIGHT_COUNT, motor_commands
from train_genetic import evaluate


class ControllerTest(unittest.TestCase):
    def test_motor_commands_are_bounded(self):
        observation = {"obs": np.ones(24)}
        commands = motor_commands(observation, np.ones(WEIGHT_COUNT))
        self.assertEqual(commands.shape, (4,))
        self.assertTrue(np.all(np.abs(commands) <= 1))

    def test_fitness_adds_rewards_until_run_ends(self):
        class Environment:
            def reset(self, seed):
                self.steps = 0
                return {"obs": np.zeros(24)}, {}

            def step(self, action):
                self.steps += 1
                return {"obs": np.zeros(24)}, 2.0, self.steps == 3, False, {}

        self.assertEqual(evaluate(Environment(), np.zeros(WEIGHT_COUNT)), 6.0)


if __name__ == "__main__":
    unittest.main()
