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

    def test_text_solver_bayes_problem(self):
        text = """
        Giải bằng Bayes. P(H)=1%, P(E|H)=99%, P(E|not H)=5%.
        Quan sát kết quả dương tính. Hãy tính posterior.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "decision_tree")
        solved = result["solved"]["result"]
        self.assertEqual(solved["solver"], "bayes_rule")
        self.assertAlmostEqual(solved["posterior"], 0.166667, places=5)
        self.assertTrue(result["solved"]["validation"]["is_valid"])

    def test_text_solver_expected_value_problem(self):
        text = """
        Expected value decision.
        States: high, low
        Probabilities: 0.4, 0.6
        A: 100, 0
        B: 50, 40
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "decision_tree")
        self.assertEqual(result["solved"]["result"]["recommendation"], "B")
        self.assertAlmostEqual(result["solved"]["result"]["results"][0]["expected_value"], 44)

    def test_text_solver_markdown_linear_programming_batch(self):
        text = r"""
        ### Bài 3
        Solve the following **linear programming problems**.

        #### a)
        Maximize:
        \[
        Z = 2x + 5y
        \]
        Subject to:
        \[
        3x + 4y \leq 8
        \]
        \[
        2x + 7y \leq 12
        \]
        \[
        x \geq 0,\quad y \geq 0
        \]

        #### b)
        Maximize:
        \[
        Z = 3x_1 - x_2 + 2x_3 + 4x_4
        \]
        Subject to:
        \[
        x_2 + 7x_3 + 2x_4 = 9
        \]
        \[
        2x_1 + 3x_2 + x_3 = 12
        \]
        \[
        x_i \geq 0,\quad i = 1,2,3,4
        \]
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "linear_programming_batch")
        items = result["solved"]["result"]["items"]
        self.assertEqual(len(items), 2)
        self.assertAlmostEqual(items[0]["result"]["objective_value"], 116 / 13, places=5)
        self.assertAlmostEqual(items[0]["result"]["all_values"]["x"], 8 / 13, places=5)
        self.assertAlmostEqual(items[0]["result"]["all_values"]["y"], 20 / 13, places=5)
        self.assertAlmostEqual(items[1]["result"]["objective_value"], 36, places=5)
        self.assertAlmostEqual(items[1]["result"]["all_values"]["x1"], 6, places=5)
        self.assertAlmostEqual(items[1]["result"]["all_values"]["x4"], 4.5, places=5)

    def test_text_solver_lp_minimize_with_ge_constraint(self):
        text = """
        Minimize:
        Z = x + y
        Subject to:
        x + y ≥ 4
        x ≥ 0, y ≥ 0
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        item = result["solved"]["result"]["items"][0]["result"]
        self.assertEqual(item["status"], "optimal")
        self.assertAlmostEqual(item["objective_value"], 4)

    def test_text_solver_lp_infeasible(self):
        text = """
        Maximize:
        Z = x
        Subject to:
        x <= 1
        x >= 2
        x >= 0
        """
        result = solve_text_problem(text)
        item = result["solved"]["result"]["items"][0]["result"]
        self.assertEqual(item["status"], "infeasible")
        self.assertIn("không có nghiệm khả thi", item["markdown_report"].lower())

    def test_text_solver_lp_unbounded(self):
        text = """
        Maximize:
        Z = x + y
        Subject to:
        x - y >= 0
        x >= 0, y >= 0
        """
        result = solve_text_problem(text)
        item = result["solved"]["result"]["items"][0]["result"]
        self.assertEqual(item["status"], "unbounded")
        self.assertIn("không bị chặn", item["markdown_report"].lower())

    def test_text_solver_lp_unicode_leq(self):
        text = """
        Maximize:
        Z = 4x1 + 3x2
        Subject to:
        2x1 + x2 ≤ 8
        x1 + 2x2 ≤ 8
        x1 >= 0, x2 >= 0
        """
        result = solve_text_problem(text)
        item = result["solved"]["result"]["items"][0]["result"]
        self.assertEqual(item["status"], "optimal")
        self.assertAlmostEqual(item["objective_value"], 56 / 3, places=5)

    def test_text_solver_max_flow(self):
        text = """
        Max flow problem
        Source: S
        Sink: T
        S -> A | 3
        S -> B | 2
        A -> T | 2
        B -> T | 4
        A -> B | 1
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "max_flow")
        self.assertEqual(result["solved"]["result"]["status"], "optimal")
        self.assertAlmostEqual(result["solved"]["result"]["objective_value"], 5)
        self.assertIn("Max Flow", result["solved"]["result"]["markdown_report"])

    def test_text_solver_min_cost_flow(self):
        text = """
        Min cost flow problem
        S: supply=5
        T: demand=5
        S -> A | 1 | 3
        S -> T | 5 | 5
        A -> T | 1 | 3
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "min_cost_flow")
        self.assertEqual(result["solved"]["result"]["status"], "optimal")
        self.assertAlmostEqual(result["solved"]["result"]["objective_value"], 16)

    def test_text_solver_general_dp_minimize(self):
        text = """
        Dynamic programming resource allocation
        Minimize cost
        Total resource: 4
        Stage 1: 0, 4, 7, 9, 12
        Stage 2: 0, 3, 6, 10, 13
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["solved"]["problem_type"], "dynamic_programming")
        solved = result["solved"]["result"]
        self.assertEqual(solved["sense"], "minimize")
        self.assertAlmostEqual(solved["objective_value"], 12)
        self.assertEqual([item["resource"] for item in solved["allocation"]], [4, 0])


if __name__ == "__main__":
    unittest.main()
