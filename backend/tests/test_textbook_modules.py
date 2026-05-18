import math
import unittest

from edss.classifier import classify_problem
from edss.inventory import solve_eoq, solve_newsvendor
from edss.markov import absorbing_chain, steady_state
from edss.module_registry import module_coverage_report
from edss.queueing import solve_mm1, solve_mmc
from edss.text_solver import solve_text_problem
from edss.utility import exponential_certainty_equivalent


class TextbookModuleTests(unittest.TestCase):
    def test_module_registry_tracks_twenty_modules(self):
        report = module_coverage_report()
        self.assertEqual(len(report["modules"]), 20)
        self.assertGreaterEqual(report["counts"].get("implemented", 0), 7)

    def test_inventory_eoq(self):
        result = solve_eoq(demand=1000, ordering_cost=50, holding_cost=2)
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["order_quantity"], math.sqrt(50000), places=5)
        self.assertIn("Báo cáo EOQ", result["markdown_report"])

    def test_inventory_newsvendor_critical_ratio(self):
        result = solve_newsvendor(unit_cost=5, selling_price=10, salvage_value=2)
        self.assertAlmostEqual(result["critical_ratio"], 0.625)

    def test_queueing_mm1_metrics(self):
        result = solve_mm1(arrival_rate=2, service_rate=3)
        self.assertEqual(result["status"], "stable")
        self.assertAlmostEqual(result["rho"], 2 / 3)
        self.assertAlmostEqual(result["L"], 2)
        self.assertAlmostEqual(result["W"], 1)

    def test_queueing_mmc_metrics(self):
        result = solve_mmc(arrival_rate=4, service_rate=3, servers=2)
        self.assertEqual(result["status"], "stable")
        self.assertLess(result["rho"], 1)
        self.assertGreater(result["L"], result["Lq"])

    def test_markov_steady_state(self):
        result = steady_state([[0.8, 0.2], [0.1, 0.9]])
        self.assertAlmostEqual(result["steady_state"][0], 1 / 3, places=5)
        self.assertAlmostEqual(result["steady_state"][1], 2 / 3, places=5)

    def test_markov_absorbing_chain(self):
        result = absorbing_chain([[0.5, 0.5, 0], [0.2, 0.3, 0.5], [0, 0, 1]], [2])
        self.assertEqual(result["status"], "computed")
        self.assertEqual(len(result["expected_time_to_absorption"]), 2)

    def test_utility_certainty_equivalent_is_below_ev_for_risk_averse_lottery(self):
        result = exponential_certainty_equivalent(
            [{"value": 0, "probability": 0.5}, {"value": 100, "probability": 0.5}],
            risk_tolerance=50,
        )
        self.assertLess(result["certainty_equivalent"], result["expected_value"])
        self.assertGreater(result["risk_premium"], 0)

    def test_classifier_recognizes_new_textbook_modules(self):
        self.assertEqual(classify_problem("EOQ with reorder point and safety stock")["problem_type"], "inventory_theory")
        self.assertEqual(classify_problem("M/M/1 queue with arrival rate and service rate")["problem_type"], "queueing_theory")
        self.assertEqual(classify_problem("Markov transition matrix steady-state")["problem_type"], "markov_processes")

    def test_text_solver_direct_eoq_and_mm1(self):
        eoq = solve_text_problem("EOQ problem: D=1000, S=50, H=2.")
        self.assertEqual(eoq["status"], "solved")
        self.assertEqual(eoq["solved"]["problem_type"], "inventory")
        self.assertAlmostEqual(eoq["solved"]["result"]["order_quantity"], math.sqrt(50000), places=5)

        mm1 = solve_text_problem("M/M/1 queue: lambda=2, mu=3.")
        self.assertEqual(mm1["status"], "solved")
        self.assertEqual(mm1["solved"]["problem_type"], "queueing")
        self.assertAlmostEqual(mm1["solved"]["result"]["L"], 2)

    def test_paper_inventory_quantity_discount_text_solver(self):
        text = """
        ### Paper Inventory Management Problem
        The Planetas Publishing House has studied 70-gram book paper demand in the last 12 months:
        10, 11, 10, 9, 10, 11, 9, 10.5, 10, 9, 9, 11.5 tonnes each month.
        The purchasing price is to remain at $2,300,000 per tonne.
        The order cost is $500,000.
        The annual inventory management charge is 15% of the unit cost.
        There is an additional storage cost of $55,000 per tonne.
        If the supplier offers a 10% discount for purchases over 30 tonnes, and an 11% discount for purchases of 60 tonnes or more,
        how would the inventory policy change? If storage per tonne is lower, how would policy change?
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        solved = result["solved"]["result"]
        self.assertEqual(solved["model"], "paper_inventory_quantity_discount")
        self.assertAlmostEqual(solved["annual_demand"], 120)
        self.assertAlmostEqual(solved["holding_cost"], 400000)
        self.assertAlmostEqual(solved["basic_eoq"], 17.320508, places=5)
        self.assertAlmostEqual(solved["basic_total_cost"], 282928203.230276, places=3)
        self.assertAlmostEqual(solved["current_monthly_total_cost"], 284000000, places=3)
        self.assertAlmostEqual(solved["best_discount"]["quantity"], 30)
        self.assertAlmostEqual(solved["best_discount"]["total_cost"], 256400000, places=3)
        self.assertAlmostEqual(solved["reduced_storage_eoq"], 46.709937, places=5)
        self.assertAlmostEqual(solved["reduced_storage_total_cost"], 250969046.515733, places=3)
        self.assertIn("Paper Inventory Management Problem", solved["markdown_report"])


if __name__ == "__main__":
    unittest.main()
