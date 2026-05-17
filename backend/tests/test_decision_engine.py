import unittest

from decision_engine import compute_results


class DecisionEngineTests(unittest.TestCase):
    def test_expected_best_option_for_sample_business_decision(self):
        payload = {
            "question": "Should we launch?",
            "domain": "business",
            "objective": "Maximize validated learning with controlled downside.",
            "context": "Small team, reversible release.",
            "riskTolerance": 0.55,
            "options": [
                {
                    "id": "mvp",
                    "name": "Launch MVP",
                    "cost": 18,
                    "reversibility": 0.75,
                    "scenarios": [
                        {"name": "Strong signal", "probability": 0.42, "utility": 86},
                        {"name": "Slow adoption", "probability": 0.38, "utility": 48},
                        {"name": "Quality miss", "probability": 0.2, "utility": 18},
                    ],
                },
                {
                    "id": "wait",
                    "name": "Research more",
                    "cost": 10,
                    "reversibility": 0.9,
                    "scenarios": [
                        {"name": "Better model", "probability": 0.5, "utility": 68},
                        {"name": "Lose momentum", "probability": 0.3, "utility": 35},
                        {"name": "Find major risk", "probability": 0.2, "utility": 58},
                    ],
                },
            ],
        }
        result = compute_results(payload)
        self.assertEqual(result["recommendation"], "Research more")
        self.assertGreater(result["option_results"][0]["risk_adjusted_score"], result["option_results"][1]["risk_adjusted_score"])
        self.assertGreaterEqual(result["value_of_information"]["score"], 0)

    def test_probability_normalization_warning(self):
        payload = {
            "question": "Pick a route",
            "objective": "Arrive reliably.",
            "riskTolerance": 0.4,
            "options": [
                {
                    "id": "a",
                    "name": "Route A",
                    "cost": 1,
                    "reversibility": 0.8,
                    "scenarios": [
                        {"name": "Fast", "probability": 60, "utility": 70},
                        {"name": "Slow", "probability": 40, "utility": 20},
                    ],
                },
                {
                    "id": "b",
                    "name": "Route B",
                    "cost": 3,
                    "reversibility": 0.9,
                    "scenarios": [
                        {"name": "Fast", "probability": 0.5, "utility": 65},
                        {"name": "Slow", "probability": 0.5, "utility": 45},
                    ],
                },
            ],
        }
        result = compute_results(payload)
        self.assertTrue(any("chuẩn hóa" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
