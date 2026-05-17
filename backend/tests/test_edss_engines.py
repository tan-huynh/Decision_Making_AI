import unittest

from edss.assignment import solve_assignment
from edss.classifier import classify_problem
from edss.linear_programming import solve_lp
from edss.network import solve_shortest_path
from edss.router import solve_problem
from edss.text_solver import solve_text_problem
from edss.uncertainty import expected_payoff, value_of_information


def production_mix_problem():
    return {
        "context": {"description": "production mix resource capacity optimization", "unit": "USD/week"},
        "problem_type": "linear_programming",
        "variables": [
            {"name": "x_A", "lower_bound": 0},
            {"name": "x_B", "lower_bound": 0},
        ],
        "objective": {"sense": "maximize", "coefficients": {"x_A": 40, "x_B": 30}},
        "constraints": [
            {"name": "labor", "coefficients": {"x_A": 2, "x_B": 1}, "operator": "<=", "rhs": 100},
            {"name": "machine", "coefficients": {"x_A": 1, "x_B": 2}, "operator": "<=", "rhs": 80},
            {"name": "demand_A", "coefficients": {"x_A": 1}, "operator": "<=", "rhs": 40},
            {"name": "demand_B", "coefficients": {"x_B": 1}, "operator": "<=", "rhs": 50},
        ],
    }


class EDSSEngineTests(unittest.TestCase):
    def test_lp_solver_production_mix(self):
        result = solve_lp(production_mix_problem())
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["solution"]["x_A"], 40)
        self.assertAlmostEqual(result["solution"]["x_B"], 20)
        self.assertAlmostEqual(result["objective_value"], 2200)

    def test_assignment_solver(self):
        result = solve_assignment([[9, 2, 7], [6, 4, 3], [5, 8, 1]])
        self.assertEqual(result["status"], "optimal")
        self.assertEqual(result["objective_value"], 9)

    def test_shortest_path(self):
        result = solve_shortest_path(
            {
                "source": "A",
                "target": "D",
                "edges": [
                    {"from": "A", "to": "B", "cost": 1},
                    {"from": "B", "to": "D", "cost": 2},
                    {"from": "A", "to": "D", "cost": 10},
                ],
            }
        )
        self.assertEqual(result["path"], ["A", "B", "D"])
        self.assertEqual(result["objective_value"], 3)

    def test_uncertainty_expected_payoff_and_voi(self):
        problem = {
            "alternatives": [{"name": "A"}, {"name": "B"}],
            "states": [{"name": "high", "probability": 0.4}, {"name": "low", "probability": 0.6}],
            "payoff_matrix": [
                {"alternative": "A", "state": "high", "payoff": 100},
                {"alternative": "A", "state": "low", "payoff": 0},
                {"alternative": "B", "state": "high", "payoff": 50},
                {"alternative": "B", "state": "low", "payoff": 40},
            ],
        }
        expected = expected_payoff(problem)
        self.assertEqual(expected["recommendation"], "B")
        self.assertGreater(value_of_information(problem)["EVPI"], 0)

    def test_router_and_classifier(self):
        classification = classify_problem("production mix with labor capacity and profit")
        self.assertEqual(classification["problem_type"], "linear_programming")
        solved = solve_problem(production_mix_problem())
        self.assertEqual(solved["result"]["status"], "optimal")

    def test_text_solver_power_transportation(self):
        text = """
        Nhà máy | Công suất cung cấp
        Nhà máy 1 | 20 MW
        Nhà máy 2 | 30 MW
        Nhà máy 3 | 45 MW
        Thành phố | Nhu cầu
        Thành phố 1 | 20 MW
        Thành phố 2 | 25 MW
        Thành phố 3 | 15 MW
        Thành phố 4 | 35 MW
        C11 = 12 C12 = 5 C13 = 16 C14 = 18
        C21 = 10 C22 = 9 C23 = 7 C24 = 14
        C31 = 8 C32 = 11 C33 = 12 C34 = 15
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["result"]["status"], "optimal")
        self.assertEqual(result["solved"]["result"]["objective_value"], 925)

    def test_text_solver_dynamic_programming_steel(self):
        text = """
        Một nhà máy sản xuất thép, trong 3 ngày sản xuất được 7 tấn thép.
        | Ngày đầu tiên | Lợi nhuận (10^5 USD) | 0 | 15 | 40 | 65 |
        | Ngày thứ 2 | Lợi nhuận (10^5 USD) | 0 | 30 | 45 | 55 |
        | Ngày thứ 3 | Lợi nhuận (10^5 USD) | 0 | 20 | 35 | 60 |
        Giải bằng phương pháp quy hoạch động.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "dynamic_programming")
        self.assertEqual(result["solved"]["result"]["objective_value"], 155)
        allocation = result["solved"]["result"]["allocation"]
        self.assertEqual([item["tons"] for item in allocation], [3, 1, 3])

    def test_text_solver_dynamic_programming_steel_full_markdown_table(self):
        text = """
        Một nhà máy sản xuất thép, trong **3 ngày** sản xuất được **7 tấn thép**.
        | Ngày | Sản xuất / Lợi nhuận | 0 | 1 | 2 | 3 |
        |---|---|---:|---:|---:|---:|
        | Ngày đầu tiên | Sản xuất (tấn) | 0 | 1 | 2 | 3 |
        |  | Lợi nhuận \\((10^5 \\text{ USD})\\) | 0 | 15 | 40 | 65 |
        | Ngày thứ 2 | Sản xuất (tấn) | 0 | 1 | 2 | 3 |
        |  | Lợi nhuận \\((10^5 \\text{ USD})\\) | 0 | 30 | 45 | 55 |
        | Ngày thứ 3 | Sản xuất (tấn) | 0 | 1 | 2 | 3 |
        |  | Lợi nhuận \\((10^5 \\text{ USD})\\) | 0 | 20 | 35 | 60 |
        Giải bằng phương pháp quy hoạch động.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        returns = result["problem"]["resource_allocation"]["stage_returns"]
        self.assertEqual(returns, [[0.0, 15.0, 40.0, 65.0], [0.0, 30.0, 45.0, 55.0], [0.0, 20.0, 35.0, 60.0]])
        self.assertEqual(result["solved"]["result"]["objective_value"], 155)
        self.assertEqual([item["tons"] for item in result["solved"]["result"]["allocation"]], [3, 1, 3])

    def test_text_solver_oil_drilling_probability_tree(self):
        text = """
        Một công ty thăm dò dầu mỏ khoan trúng dầu trong 10% những giếng khoan của mình.
        Nếu công ty này khoan hai giếng, thì bốn biến cố đơn có thể xảy ra.
        a. Hãy vẽ sơ đồ hình cây minh họa bốn biến cố trên.
        b. Hãy tìm xác suất để công ty này sẽ khoan trúng dầu trong giếng khoan thứ nhất
        và không trúng dầu trong giếng khoan thứ hai.
        c. Hãy tìm xác suất để công ty này sẽ khoan trúng dầu trong ít nhất một trong hai giếng khoan này.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "decision_tree")
        solved = result["solved"]["result"]
        self.assertEqual(len(solved["outcomes"]), 4)
        self.assertAlmostEqual(solved["queries"]["first_success_second_failure"], 0.09)
        self.assertAlmostEqual(solved["queries"]["at_least_one_success"], 0.19)


if __name__ == "__main__":
    unittest.main()
