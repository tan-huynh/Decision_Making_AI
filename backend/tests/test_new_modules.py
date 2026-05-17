"""Tests for new EDIS modules: probability engine, model validator, VOI, sensitivity, data estimation."""

from __future__ import annotations

import unittest


class ProbabilityEngineTests(unittest.TestCase):
    def test_bayes_update_positive(self):
        from edss.probability_engine import bayes_update
        r = bayes_update(prior=0.3, sensitivity=0.9, false_positive_rate=0.2, observed_positive=True)
        self.assertAlmostEqual(r["prior"], 0.3)
        self.assertGreater(r["posterior"], 0.3)  # evidence supports hypothesis
        self.assertLessEqual(r["posterior"], 1.0)

    def test_bayes_update_negative(self):
        from edss.probability_engine import bayes_update
        r = bayes_update(prior=0.3, sensitivity=0.9, false_positive_rate=0.2, observed_positive=False)
        self.assertLess(r["posterior"], 0.3)  # negative evidence weakens hypothesis

    def test_bayes_multi_step(self):
        from edss.probability_engine import bayes_multi_step
        r = bayes_multi_step(
            prior=0.5,
            tests=[
                {"sensitivity": 0.9, "false_positive_rate": 0.1, "observed_positive": True},
                {"sensitivity": 0.8, "false_positive_rate": 0.2, "observed_positive": True},
            ],
        )
        self.assertEqual(len(r["steps"]), 2)
        self.assertGreater(r["final_posterior"], 0.5)

    def test_independent_events(self):
        from edss.probability_engine import independent_events
        r = independent_events([0.5, 0.5])
        self.assertAlmostEqual(r["P_all_occur"], 0.25)
        self.assertAlmostEqual(r["P_none_occur"], 0.25)
        self.assertAlmostEqual(r["P_at_least_one"], 0.75)

    def test_distribution_normal(self):
        from edss.probability_engine import distribution_pmf_pdf
        r = distribution_pmf_pdf("normal", {"mean": 0, "std": 1}, [0])
        self.assertIn("results", r)
        self.assertAlmostEqual(r["results"][0]["cdf"], 0.5, places=3)

    def test_fit_distribution(self):
        from edss.probability_engine import fit_distribution
        import random
        rng = random.Random(42)
        data = [rng.gauss(10, 2) for _ in range(100)]
        r = fit_distribution(data)
        self.assertIsNotNone(r.get("best_fit"))
        self.assertEqual(r["n_samples"], 100)


class ModelValidatorTests(unittest.TestCase):
    def test_valid_lp(self):
        from edss.model_validator import validate_model
        problem = {
            "problem_type": "linear_programming",
            "context": {"unit": "USD"},
            "variables": [{"name": "x", "lower_bound": 0}],
            "objective": {"sense": "maximize", "coefficients": {"x": 5}},
            "constraints": [{"name": "c1", "coefficients": {"x": 1}, "operator": "<=", "rhs": 10}],
            "assumptions": ["Test"],
        }
        r = validate_model(problem)
        self.assertTrue(r["is_valid"])
        self.assertEqual(len(r["errors"]), 0)

    def test_invalid_probability(self):
        from edss.model_validator import validate_model
        problem = {
            "problem_type": "decision_tree",
            "states": [{"name": "A", "probability": -0.5}, {"name": "B", "probability": 0.3}],
        }
        r = validate_model(problem)
        self.assertFalse(r["is_valid"])
        self.assertTrue(any("âm" in e for e in r["errors"]))

    def test_duplicate_variable(self):
        from edss.model_validator import validate_model
        problem = {
            "problem_type": "linear_programming",
            "variables": [{"name": "x"}, {"name": "x"}],
            "objective": {"sense": "maximize", "coefficients": {"x": 1}},
            "constraints": [{"name": "c1", "coefficients": {"x": 1}, "operator": "<=", "rhs": 5}],
        }
        r = validate_model(problem)
        self.assertFalse(r["is_valid"])


class VOIEngineTests(unittest.TestCase):
    def test_evpi(self):
        from edss.voi_engine import compute_evpi
        alts = ["Drill", "NoDrill"]
        states = [{"name": "Oil", "probability": 0.4}, {"name": "NoOil", "probability": 0.6}]
        lookup = {
            ("Drill", "Oil"): 600, ("Drill", "NoOil"): -100,
            ("NoDrill", "Oil"): 0, ("NoDrill", "NoOil"): 0,
        }
        r = compute_evpi(alts, states, lookup)
        self.assertGreater(r["EVPI"], 0)
        self.assertAlmostEqual(r["EVwoPI"], 0.4 * 600 + 0.6 * (-100), places=2)

    def test_evi(self):
        from edss.voi_engine import compute_evi
        alts = ["Drill", "NoDrill"]
        states = [{"name": "Oil", "probability": 0.4}, {"name": "NoOil", "probability": 0.6}]
        lookup = {
            ("Drill", "Oil"): 600, ("Drill", "NoOil"): -100,
            ("NoDrill", "Oil"): 0, ("NoDrill", "NoOil"): 0,
        }
        test_sens = {
            "Oil": {"Positive": 0.9, "Negative": 0.1},
            "NoOil": {"Positive": 0.3, "Negative": 0.7},
        }
        r = compute_evi(alts, states, lookup, test_sens, test_cost=50)
        self.assertGreater(r["EVI"], 0)
        self.assertLess(r["EVI"], r["EVI"] + 1)  # sanity
        self.assertIn("should_buy_info", r)


class DataEstimationTests(unittest.TestCase):
    def test_descriptive_stats(self):
        from edss.data_estimation import descriptive_stats
        r = descriptive_stats([10, 20, 30, 40, 50])
        self.assertAlmostEqual(r["mean"], 30.0)
        self.assertAlmostEqual(r["median"], 30.0)
        self.assertEqual(r["n"], 5)

    def test_histogram(self):
        from edss.data_estimation import build_histogram
        r = build_histogram([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_bins=5)
        self.assertEqual(r["n_bins"], 5)
        self.assertEqual(sum(b["count"] for b in r["bins"]), 10)

    def test_compare_groups(self):
        from edss.data_estimation import compare_groups
        r = compare_groups({"A": [10, 20, 30], "B": [15, 25, 35]})
        self.assertEqual(r["n_groups"], 2)
        self.assertEqual(r["best_by_mean"], "B")


class SensitivityEngineTests(unittest.TestCase):
    def test_probability_sensitivity(self):
        from edss.sensitivity_engine import sensitivity_analysis
        problem = {
            "problem_type": "decision_tree",
            "alternatives": [{"name": "A"}, {"name": "B"}],
            "states": [{"name": "Good", "probability": 0.6}, {"name": "Bad", "probability": 0.4}],
            "payoff_matrix": [
                {"alternative": "A", "state": "Good", "payoff": 100},
                {"alternative": "A", "state": "Bad", "payoff": -50},
                {"alternative": "B", "state": "Good", "payoff": 30},
                {"alternative": "B", "state": "Bad", "payoff": 20},
            ],
        }
        r = sensitivity_analysis(problem, {"status": "computed"})
        self.assertIn("probability_sensitivity", r)
        self.assertEqual(len(r["probability_sensitivity"]), 2)


class AuditTests(unittest.TestCase):
    def test_audit_trail(self):
        from edss.audit import create_audit_trail, log_step, decision_quality_assessment
        trail = create_audit_trail("test-problem")
        log_step(trail, "classification", "linear_programming")
        log_step(trail, "model_validation", {"is_valid": True})
        log_step(trail, "solver_execution", "optimal")
        self.assertEqual(len(trail["steps"]), 3)

        quality = decision_quality_assessment(trail)
        self.assertGreater(quality["decision_quality_score"], 0)


if __name__ == "__main__":
    unittest.main()
