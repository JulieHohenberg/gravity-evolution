import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stable_baselines3 import PPO

import train_rl
from simulation import make_environment


class RLTest(unittest.TestCase):
    def test_training_saves_checkpoints_and_uses_sensor_readings(self):
        """Check saved models, scores, and motor commands from a short run."""
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train_rl, "ROLLOUT_STEPS", 64), patch.object(train_rl, "SAVE_EVERY", 64), patch.object(train_rl.tempfile, "mkdtemp", return_value=directory):
                train_rl.train("Moon", 64)
            run = Path(directory)
            self.assertTrue((run / "checkpoint_64_steps.zip").exists())
            self.assertTrue((run / "evaluations.npz").exists())
            self.assertTrue((run / "training.monitor.csv").exists())
            model = PPO.load(run / "final-model.zip", device="cpu")
            environment = make_environment("Moon")
            try:
                observation, _ = environment.reset(seed=42)
                action, _ = model.predict(observation["obs"], deterministic=True)
                self.assertEqual(action.shape, (4,))
                environment.step(action)
            finally:
                environment.close()


if __name__ == "__main__":
    unittest.main()
