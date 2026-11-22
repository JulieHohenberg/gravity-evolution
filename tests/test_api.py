import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import app
from controller import WEIGHT_COUNT


class RaceAPITest(unittest.TestCase):
    def test_request_validation_and_missing_model(self):
        """Reject invalid choices and report models that aren't installed."""
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "ROOT", Path(directory)):
            client = TestClient(app.app)
            self.assertEqual(client.post("/race", json={"planet": "Mars", "models": ["genetic-earth"]}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": []}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": ["ppo-earth", "ppo-earth"]}).status_code, 422)
            self.assertEqual(client.post("/race", json={"planet": "Earth", "models": ["ppo-earth"]}).status_code, 404)

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
                self.assertEqual(racer["frames"][0]["step"], 0)
                self.assertEqual(racer["frames"][-1]["step"], racer["steps"])
                self.assertEqual(len(racer["frames"][0]["body"]), 5)
                self.assertTrue(racer["terrain"])


if __name__ == "__main__":
    unittest.main()
