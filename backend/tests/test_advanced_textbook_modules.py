import unittest

from edss.dynamic_programming import solve_inventory_dp
from edss.goal_programming import solve_goal_programming
from edss.influence_diagram import analyze_influence_diagram
from edss.lp_advanced import duality_analysis, sensitivity_ranges, simplex_step_by_step
from edss.mip import solve_mip
from edss.network import solve_bellman_ford, solve_minimum_spanning_tree
from edss.nonlinear import solve_nonlinear
from edss.text_solver import solve_text_problem


LP = {
    "sense": "maximize",
    "c": [3, 2],
    "A_ub": [[1, 1], [1, 0], [0, 1]],
    "b_ub": [4, 2, 3],
    "bounds": [(0, None), (0, None)],
    "variable_names": ["x", "y"],
    "constraint_names_ub": ["resource", "x_cap", "y_cap"],
}


class AdvancedTextbookModuleTests(unittest.TestCase):
    def test_simplex_step_by_step(self):
        result = simplex_step_by_step(LP)
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["objective_value"], 10)
        self.assertIn("Simplex tableau", result["markdown_report"])

    def test_duality_and_sensitivity(self):
        dual = duality_analysis(LP)
        self.assertEqual(dual["status"], "computed")
        self.assertAlmostEqual(dual["primal"]["objective_value"], dual["dual"]["objective_value"], places=5)
        self.assertIn("Complementary slackness", dual["markdown_report"])

        sensitivity = sensitivity_ranges(LP)
        self.assertEqual(sensitivity["status"], "computed")
        self.assertTrue(sensitivity["rhs_ranges"])

    def test_mip_binary(self):
        result = solve_mip(
            {
                "sense": "maximize",
                "c": [10, 7],
                "A_ub": [[6, 4]],
                "b_ub": [8],
                "bounds": [(0, 1), (0, 1)],
                "integrality": [1, 1],
                "variable_names": ["project_a", "project_b"],
            }
        )
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["objective_value"], 10)

    def test_goal_programming_weighted(self):
        result = solve_goal_programming(
            {
                "variable_names": ["x"],
                "goals": [{"coefficients": [1], "target": 10, "weight_under": 1, "weight_over": 2}],
                "A_ub": [[1]],
                "b_ub": [8],
            }
        )
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["objective_value"], 2)
        self.assertIn("Deviation variables", result["markdown_report"])

    def test_bellman_ford_and_mst(self):
        bf = solve_bellman_ford(
            {
                "source": "A",
                "target": "D",
                "edges": [
                    {"from": "A", "to": "B", "cost": 1},
                    {"from": "B", "to": "C", "cost": -2},
                    {"from": "C", "to": "D", "cost": 2},
                    {"from": "A", "to": "D", "cost": 5},
                ],
            }
        )
        self.assertEqual(bf["status"], "optimal")
        self.assertAlmostEqual(bf["objective_value"], 1)

        mst = solve_minimum_spanning_tree(
            {
                "edges": [
                    {"from": "A", "to": "B", "cost": 1},
                    {"from": "B", "to": "C", "cost": 2},
                    {"from": "A", "to": "C", "cost": 5},
                ]
            }
        )
        self.assertEqual(mst["status"], "optimal")
        self.assertAlmostEqual(mst["objective_value"], 3)

    def test_inventory_dp(self):
        result = solve_inventory_dp(
            {
                "demands": [2, 3],
                "initial_inventory": 0,
                "max_inventory": 5,
                "order_cost": 1,
                "holding_cost": 0.5,
                "shortage_cost": 100,
            }
        )
        self.assertEqual(result["status"], "optimal")
        self.assertEqual(sum(row["order_q"] for row in result["rollout"]), 5)

    def test_nonlinear_quadratic(self):
        result = solve_nonlinear(
            {
                "sense": "minimize",
                "variable_names": ["x", "y"],
                "initial": [0, 0],
                "objective": {"type": "quadratic", "Q": [[2, 0], [0, 2]], "c": [-2, -4]},
                "constraints": [{"type": "ineq", "coefficients": [1, 1], "rhs": 10}],
                "bounds": [(0, None), (0, None)],
            }
        )
        self.assertEqual(result["status"], "optimal_local")
        self.assertAlmostEqual(result["solution"]["x"], 1, places=4)
        self.assertAlmostEqual(result["solution"]["y"], 2, places=4)

    def test_influence_diagram(self):
        result = analyze_influence_diagram(
            {
                "nodes": [
                    {"id": "D", "type": "decision", "label": "Drill?"},
                    {"id": "O", "type": "chance", "label": "Oil"},
                    {"id": "V", "type": "value", "label": "Profit"},
                ],
                "arcs": [{"from": "D", "to": "V"}, {"from": "O", "to": "V"}],
            }
        )
        self.assertEqual(result["status"], "computed")
        self.assertIn("mermaid", result["markdown_report"].lower())

    def test_designing_boxes_circle_packing_text_solver(self):
        text = """
        ### Designing Boxes Problem
        A firm must design a box of minimum dimensions to pack three circular objects with radii:
        R_1 = 6, R_2 = 12, R_3 = 16 cm.
        Consider a non-linear programming model that minimises the perimeter of the box.
        The circles placed inside the box cannot overlap.
        (X1-X2)^2 + (Y1-Y2)^2 >= (6+12)^2
        (X1-X3)^2 + (Y1-Y3)^2 >= (6+16)^2
        (X2-X3)^2 + (Y2-Y3)^2 >= (12+16)^2
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["problem"]["problem_type"], "circle_packing_box")
        solved = result["solved"]["result"]
        self.assertEqual(solved["status"], "optimal_local")
        self.assertGreater(solved["box"]["A_height"], 0)
        self.assertGreater(solved["box"]["B_width"], 0)
        self.assertTrue(all(check["slack"] >= -1e-5 for check in solved["constraint_checks"]))
        self.assertIn("Nonlinear circle packing", solved["markdown_report"])

    def test_library_shelving_shortest_path_text_solver(self):
        text = """
        ### Library Shelving Problem
        | Book height | Number of books |
        |---:|---:|
        | 19 cm | 200 |
        | 24 cm | 150 |
        | 31 cm | 100 |
        | 35 cm | 80 |
        A mean thickness of 3 cm is considered for all the books.
        The construction of each shelving unit costs $2,500.
        There is an additional cost of $5/cm² for the available area used to store books.
        Consider and solve the shortest path problem at a minimum cost.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["problem"]["problem_type"], "library_shelving_shortest_path")
        solved = result["solved"]["result"]
        self.assertEqual(solved["status"], "optimal")
        self.assertEqual(solved["path"][0]["from"], 0)
        self.assertEqual(solved["path"][-1]["to"], 4)
        self.assertAlmostEqual(solved["objective_value"], 209500)
        self.assertIn("Shortest path", solved["markdown_report"])

    def test_bac_inventory_planned_shortage_text_solver(self):
        text = """
        ### BAC Inventory Problem
        An RFG cold water dispenser costs BAC $320, and BAC estimates that it would obtain a gross profit of $80 for each dispenser sold.
        The cost of making an order of RFG dispensers is $100, and BAC employs a 20% annual inventory maintenance cost rate in relation to the purchasing price.
        If a stockout should occur in BAC's inventory, BAC estimates a cost of $50 per unmet dispenser throughout the planning horizon.
        The weekly mean demand is 75 units, and the supply time is considered null.
        By considering a 52-week year, work out the optimum inventory policy, weeks between orders, percentage delayed, maximum wait, and annual profit.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["problem"]["problem_type"], "eoq_planned_shortages")
        solved = result["solved"]["result"]
        self.assertEqual(solved["status"], "optimal")
        self.assertEqual(solved["model"], "EOQ_with_planned_shortages")
        self.assertAlmostEqual(solved["order_quantity"], 166.695831, places=5)
        self.assertAlmostEqual(solved["maximum_inventory"], 73.112206, places=5)
        self.assertAlmostEqual(solved["maximum_shortage"], 93.583624, places=5)
        self.assertAlmostEqual(solved["cycle_weeks"], 2.222611, places=5)
        self.assertAlmostEqual(solved["delayed_customer_percent"], 56.140351, places=5)
        self.assertAlmostEqual(solved["shortage_weeks"], 1.247782, places=5)
        self.assertAlmostEqual(solved["annual_total_cost_with_purchase"], 1252679.181215, places=4)
        self.assertAlmostEqual(solved["annual_profit_after_inventory_cost"], 307320.818785, places=4)
        self.assertIn("EOQ with Planned Shortages", solved["markdown_report"])
        self.assertIn("data:image/svg+xml", solved["markdown_report"])
        gate = result["recognition_gate"]
        self.assertEqual(gate["recognized_problem_type"], "inventory_theory")
        self.assertEqual(gate["decision_to_solve"], "solve")
        self.assertGreaterEqual(gate["confidence"], 0.85)
        self.assertGreaterEqual(gate["required_slot_completeness"], 0.9)


if __name__ == "__main__":
    unittest.main()
