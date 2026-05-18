import unittest

from edss.behavioral import behavioral_bias_audit
from edss.inventory import simulate_stochastic_inventory
from edss.influence_diagram import influence_to_decision_tree
from edss.lp_advanced import sensitivity_ranges
from edss.multiobjective import additive_utility, ahp_weights
from edss.queueing import solve_mg1, solve_open_queue_network
from edss.utility import fit_exponential_risk_tolerance
from edss.voi_engine import compute_evi


class CompletionModuleTests(unittest.TestCase):
    def test_mg1_and_queue_network(self):
        mg1 = solve_mg1(arrival_rate=0.5, mean_service_time=1.0, service_time_variance=1.0)
        self.assertEqual(mg1["status"], "stable")
        self.assertAlmostEqual(mg1["rho"], 0.5)

        network = solve_open_queue_network(
            nodes=[{"name": "A", "service_rate": 3}, {"name": "B", "service_rate": 4}],
            routing_matrix=[[0, 0.2], [0, 0]],
            external_arrivals=[1, 0.5],
        )
        self.assertEqual(network["status"], "computed")
        self.assertEqual(len(network["nodes"]), 2)

    def test_stochastic_inventory_simulation(self):
        result = simulate_stochastic_inventory(
            periods=5,
            initial_inventory=3,
            reorder_point=2,
            order_up_to=6,
            demand_mean=2,
            demand_std=0,
            holding_cost=1,
            shortage_cost=10,
            seed=1,
        )
        self.assertEqual(result["status"], "computed")
        self.assertEqual(len(result["rows"]), 5)

    def test_mada_ahp_and_additive_utility(self):
        ahp = ahp_weights([[1, 3], [1 / 3, 1]], ["cost", "quality"])
        self.assertEqual(ahp["status"], "computed")
        self.assertTrue(ahp["consistent"])

        maut = additive_utility(
            {
                "objectives": [
                    {"name": "cost", "weight": 0.4, "direction": "minimize", "min": 0, "max": 100},
                    {"name": "quality", "weight": 0.6, "direction": "maximize", "min": 0, "max": 10},
                ],
                "alternatives": [
                    {"name": "A", "attributes": {"cost": 30, "quality": 7}},
                    {"name": "B", "attributes": {"cost": 60, "quality": 10}},
                ],
            }
        )
        self.assertEqual(maut["recommendation"], "B")

    def test_evi_and_utility_fit_and_behavioral(self):
        evi = compute_evi(
            alternatives=["Drill", "NoDrill"],
            states=[{"name": "Oil", "probability": 0.4}, {"name": "Dry", "probability": 0.6}],
            payoff_lookup={("Drill", "Oil"): 100, ("Drill", "Dry"): -30, ("NoDrill", "Oil"): 0, ("NoDrill", "Dry"): 0},
            test_sensitivity={"Oil": {"Positive": 0.8, "Negative": 0.2}, "Dry": {"Positive": 0.2, "Negative": 0.8}},
            test_cost=1,
        )
        self.assertIn("EVI", evi)

        fit = fit_exponential_risk_tolerance([{"low": 0, "high": 100, "probability_high": 0.5, "certainty_equivalent": 40}])
        self.assertEqual(fit["status"], "computed")
        self.assertGreater(fit["risk_tolerance"], 0)

        audit = behavioral_bias_audit("Tôi chắc chắn 100% vì giá đầu tiên là neo tốt.")
        self.assertTrue(audit["bias_hits"])

    def test_sensitivity_has_objective_ranges_and_influence_tree(self):
        sens = sensitivity_ranges(
            {
                "sense": "maximize",
                "c": [3, 2],
                "A_ub": [[1, 1], [1, 0], [0, 1]],
                "b_ub": [4, 2, 3],
                "bounds": [(0, None), (0, None)],
                "variable_names": ["x", "y"],
                "constraint_names_ub": ["resource", "x_cap", "y_cap"],
            }
        )
        self.assertTrue(sens["objective_coefficient_ranges"])

        tree = influence_to_decision_tree(
            {
                "nodes": [{"id": "D", "type": "decision"}, {"id": "C", "type": "chance"}, {"id": "V", "type": "value"}],
                "arcs": [{"from": "D", "to": "V"}, {"from": "C", "to": "V"}],
            }
        )
        self.assertEqual(tree["status"], "computed")
        self.assertEqual(tree["decision_tree_skeleton"]["root"], "D")


if __name__ == "__main__":
    unittest.main()
