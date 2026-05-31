from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import autonomous


class AutonomousRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old_data_dir = autonomous.DATA_DIR
        self.old_memory = autonomous.AUTO_MEMORY_FILE
        self.old_profile = autonomous.AUTO_PROFILE_FILE
        tmp_path = Path(self.tmp.name)
        autonomous.DATA_DIR = tmp_path
        autonomous.AUTO_MEMORY_FILE = tmp_path / "autonomous_learning.jsonl"
        autonomous.AUTO_PROFILE_FILE = tmp_path / "autonomous_profile.json"

    def tearDown(self) -> None:
        autonomous.DATA_DIR = self.old_data_dir
        autonomous.AUTO_MEMORY_FILE = self.old_memory
        autonomous.AUTO_PROFILE_FILE = self.old_profile
        self.tmp.cleanup()

    def test_decision_matrix_forces_decision_theory_route(self):
        payload = {
            "question": "Should we launch?",
            "objective": "Maximize expected utility.",
            "context": "Startup launch decision.",
            "riskTolerance": 0.5,
            "timeHorizon": "30 days",
            "options": [
                {"name": "Launch", "scenarios": [{"name": "growth", "probability": 0.5, "utility": 80}]},
                {"name": "Wait", "scenarios": [{"name": "better data", "probability": 0.5, "utility": 65}]},
            ],
        }
        route = autonomous.select_decision_route(payload)
        self.assertEqual(route["route"], "decision_analysis")
        self.assertEqual(route["selected_solver"], "decision_engine")
        self.assertEqual(route["case_type"], "decision_theory")
        self.assertIn("compute_scores", route["agent_steps"])

    def test_text_route_records_learning_and_recalls_similar_case(self):
        event = {
            "route": "edss_solver",
            "case_type": "transportation_assignment",
            "selected_solver": "edss_text_solver",
            "confidence": 0.91,
            "agent_steps": autonomous.ROUTE_STEPS["edss_solver"],
            "similar_cases": [],
        }
        profile = autonomous.record_autonomous_learning(event)
        self.assertEqual(profile["events"], 1)
        self.assertEqual(profile["case_types"]["transportation_assignment"], 1)

        route = autonomous.select_text_solver_route("transportation supply demand cost matrix")
        self.assertEqual(route["route"], "edss_solver")
        self.assertIn("run_solver", route["agent_steps"])
        self.assertGreaterEqual(len(route["similar_cases"]), 1)


if __name__ == "__main__":
    unittest.main()
