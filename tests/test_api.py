import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import app
from controller import WEIGHT_COUNT


class RaceAPITest(unittest.TestCase):
    def test_bundled_models_work_without_local_training_files(self):
        """Run all four bundled models when local training files aren't present."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(Path(app.__file__).parent / "models", root / "models")
            with patch.object(app, "ROOT", root), patch.object(app, "MAX_STEPS", 2):
                client = TestClient(app.app)
                models = client.get("/models").json()["models"]
                self.assertTrue(all(model["available"] for model in models))
                response = client.post("/race", json={"planet": "Moon", "models": [model["id"] for model in models]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()["racers"]), 4)

    def test_seed_is_shared_by_all_racers(self):
        """Send the chosen seed to each racer and return it in the result."""
        results = [
            {"model": "genetic-earth", "status": "fell", "steps": 50, "distance": 4.0},
            {"model": "ppo-earth", "status": "time_limit", "steps": 1600, "distance": 8.0},
        ]
        with patch.object(app, "model_path", return_value=Path("model")), patch.object(app, "run_controller", side_effect=results) as run:
            response = TestClient(app.app).post("/race", json={"planet": "Moon", "models": ["genetic-earth", "ppo-earth"], "seed": 123})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["seed"], 123)
        self.assertEqual([call.args[-1] for call in run.call_args_list], [123, 123])

    def test_first_finisher_wins_even_if_another_goes_further(self):
        """Choose the first finisher, not the greatest final distance."""
        results = [
            {"model": "genetic-earth", "status": "finished", "steps": 900, "distance": 84.1},
            {"model": "ppo-earth", "status": "finished", "steps": 1100, "distance": 84.3},
        ]
        with patch.object(app, "model_path", return_value=Path("model")), patch.object(app, "run_controller", side_effect=results):
            response = TestClient(app.app).post("/race", json={"planet": "Earth", "models": ["genetic-earth", "ppo-earth"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["winners"], ["genetic-earth"])

    def test_furthest_wins_when_no_one_finishes(self):
        """Compare distances if all racers fall or run out of time."""
        results = [
            {"model": "genetic-earth", "status": "fell", "steps": 50, "distance": 4.0},
            {"model": "ppo-earth", "status": "time_limit", "steps": 1600, "distance": 8.0},
        ]
        with patch.object(app, "model_path", return_value=Path("model")), patch.object(app, "run_controller", side_effect=results):
            response = TestClient(app.app).post("/race", json={"planet": "Earth", "models": ["genetic-earth", "ppo-earth"]})
        self.assertEqual(response.json()["winners"], ["ppo-earth"])

    def test_request_validation_and_missing_model(self):
        """Reject invalid choices and report models that aren't installed."""
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "ROOT", Path(directory)):
            client = TestClient(app.app)
            self.assertEqual(client.post("/race", json={"planet": "Mars", "models": ["genetic-earth"]}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": []}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": ["ppo-earth", "ppo-earth"]}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": ["ppo-earth"], "seed": -1}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": ["ppo-earth"], "seed": 1.5}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": ["ppo-earth"]}).status_code, 404)

    def test_seed_changes_course_but_keeps_it_equal_for_racers(self):
        """Generate the same terrain for both racers and a new one for a new seed."""
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "ROOT", Path(directory)), patch.object(app, "MAX_STEPS", 2):
            for planet in ("earth", "moon"):
                np.save(Path(directory) / f"controller-{planet}.npy", np.zeros(WEIGHT_COUNT))
            client = TestClient(app.app)
            request = {"planet": "Earth", "models": ["genetic-earth", "genetic-moon"], "seed": 42}
            first = client.post("/race", json=request).json()
            second = client.post("/race", json={**request, "seed": 123}).json()
            self.assertEqual(first["racers"][0]["terrain"], first["racers"][1]["terrain"])
            self.assertNotEqual(first["racers"][0]["terrain"], second["racers"][0]["terrain"])

    def test_race_records_actual_simulation(self):
        """Check a controller produces timed frames on both planets."""
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "ROOT", Path(directory)), patch.object(app, "MAX_STEPS", 20):
            np.save(Path(directory) / "controller-earth.npy", np.zeros(WEIGHT_COUNT))
            client = TestClient(app.app)
            for planet in ("Earth", "Moon"):
                response = client.post("/race", json={"planet": planet, "models": ["genetic-earth"]})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                racer = result["racers"][0]
                self.assertEqual(result["winners"], ["genetic-earth"])
                self.assertEqual(result["gravity"], app.GRAVITY[planet])
                self.assertEqual(result["seed"], 42)
                self.assertEqual(racer["frames"][0]["step"], 0)
                self.assertEqual(racer["frames"][-1]["step"], racer["steps"])
                self.assertEqual(len(racer["frames"][0]["body"]), 5)
                self.assertEqual(len(racer["frames"][0]["joint_angles"]), 4)
                self.assertGreaterEqual(racer["frames"][0]["speed_mps"], 0)
                self.assertIsNone(racer["frames"][0]["motor_commands"])
                self.assertEqual(len(racer["frames"][-1]["motor_commands"]), 4)
                self.assertTrue(racer["terrain"])


if __name__ == "__main__":
    unittest.main()
