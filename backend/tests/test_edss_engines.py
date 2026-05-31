import unittest

from edss.assignment import solve_assignment
from edss.classifier import classify_problem
from edss.linear_programming import solve_lp
from edss.network import solve_shortest_path
from edss.router import solve_problem
from edss.text_solver import solve_text_problem
from edss.uncertainty import expected_payoff, value_of_information
from edss.linear_solver import recognize_linear_programming, solve_linear_programming_problem
from edss.markov_solver import recognize_markov_processes, solve_markov_processes_problem
from edss.mermaid_visualization import mermaid_visualization_gate
from edss.game_theory import solve_game_theory
from edss.dp_solver import recognize_dynamic_programming, solve_dynamic_programming_problem
from edss.decision_theory import recognize_decision_theory, solve_decision_theory_problem
from edss.queueing_solver import recognize_queueing_theory, solve_queueing_problem
from edss.inventory_solver import recognize_inventory_theory, solve_inventory_theory_problem
from edss.integer_solver import recognize_integer_programming, solve_integer_programming_problem
from edss.network_solver import recognize_network_modelling, solve_network_modelling_problem
from edss.nonlinear_solver import recognize_nonlinear_programming, solve_nonlinear_programming_problem


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

    def test_linear_programming_recognition_gate_product_mix(self):
        problem = production_mix_problem()
        recognition = recognize_linear_programming(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "LP_GRAPHICAL_2D")
        solved = solve_linear_programming_problem(problem)
        self.assertEqual(solved["status"], "optimal")
        self.assertAlmostEqual(solved["objective_value"], 2200)
        self.assertTrue(solved["verification"]["passed"])

    def test_linear_programming_gate_redirects_integer_variables(self):
        problem = production_mix_problem()
        problem["variables"][0]["variable_type"] = "integer"
        result = solve_linear_programming_problem(problem)
        self.assertEqual(result["status"], "redirect_required")
        self.assertEqual(result["target_agent"], "integer_programming")

    def test_linear_programming_gate_redirects_nonlinear_objective(self):
        problem = production_mix_problem()
        problem["objective"] = {"sense": "maximize", "expression": "x_A * x_B"}
        result = solve_linear_programming_problem(problem)
        self.assertEqual(result["status"], "redirect_required")
        self.assertEqual(result["target_agent"], "nonlinear_programming")

    def test_linear_programming_gate_blocks_missing_constraints(self):
        problem = {
            "context": {"description": "Linear programming product mix."},
            "problem_type": "linear_programming",
            "variables": [{"name": "x", "lower_bound": 0}],
            "objective": {"sense": "maximize", "coefficients": {"x": 1}},
        }
        result = solve_linear_programming_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("constraints", result["missing_data"])

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

    def test_decision_theory_recognition_gate_basic_emv(self):
        problem = {
            "context": {"description": "A firm must choose one investment under risk.", "objective_direction": "maximize"},
            "problem_type": "decision_tree",
            "alternatives": [{"name": "A"}, {"name": "B"}],
            "states": [{"name": "High", "probability": 0.4}, {"name": "Low", "probability": 0.6}],
            "payoff_matrix": [
                {"alternative": "A", "state": "High", "payoff": 100},
                {"alternative": "A", "state": "Low", "payoff": 10},
                {"alternative": "B", "state": "High", "payoff": 70},
                {"alternative": "B", "state": "Low", "payoff": 50},
            ],
        }
        recognition = recognize_decision_theory(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "DT_EMV_BASIC")
        solved = solve_decision_theory_problem(problem)
        self.assertEqual(solved["solver"], "decision_theory_basic_emv")
        self.assertEqual(solved["recommendation"], "B")
        self.assertTrue(solved["verification"]["passed"])

    def test_decision_theory_gate_blocks_missing_payoff_pair(self):
        problem = {
            "context": {"description": "A company must choose an alternative under risk."},
            "problem_type": "decision_tree",
            "alternatives": [{"name": "A"}, {"name": "B"}],
            "states": [{"name": "S1", "probability": 0.5}, {"name": "S2", "probability": 0.5}],
            "payoff_matrix": [
                {"alternative": "A", "state": "S1", "payoff": 1},
                {"alternative": "A", "state": "S2", "payoff": 2},
                {"alternative": "B", "state": "S1", "payoff": 3},
            ],
        }
        result = solve_decision_theory_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("payoff_or_cost(B,S2)", result["missing_data"])

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

    def test_dynamic_programming_recognition_gate_resource_allocation(self):
        problem = {
            "context": {"description": "Allocate 4 hours to two projects by dynamic programming."},
            "problem_type": "dynamic_programming",
            "resource_allocation": {
                "total_resource": 4,
                "resource_name": "hours",
                "sense": "maximize",
                "stage_returns": [[0, 2, 5, 6, 7], [0, 3, 4, 7, 9]],
            },
        }
        recognition = recognize_dynamic_programming(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertGreaterEqual(recognition["confidence"], 0.85)
        self.assertEqual(recognition["transition_function"], "s_{n+1} = s_n - x_n")
        solved = solve_dynamic_programming_problem(problem)
        self.assertEqual(solved["status"], "optimal")
        self.assertTrue(solved["verification"]["passed"])

    def test_dynamic_programming_gate_blocks_incomplete_stage_model(self):
        problem = {
            "context": {"description": "Dynamic programming with stages and states but no transition."},
            "problem_type": "dynamic_programming",
            "stages": [{"states": ["S0"], "actions": ["A"]}],
        }
        result = solve_dynamic_programming_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("stage_1.transitions", result["missing_data"])

    def test_queueing_recognition_gate_mm1(self):
        problem = {
            "context": {"description": "M/M/1 queue with Poisson arrivals and exponential service.", "unit": "hour"},
            "problem_type": "queueing_theory",
            "arrival_rate": 2,
            "service_rate": 3,
            "servers": 1,
        }
        recognition = recognize_queueing_theory(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "QT_MM1_INFINITE")
        solved = solve_queueing_problem(problem)
        self.assertEqual(solved["status"], "stable")
        self.assertAlmostEqual(solved["L"], 2)
        self.assertTrue(solved["verification"]["passed"])

    def test_mermaid_visualization_gate_and_router_queue_diagram(self):
        problem = {
            "context": {"description": "Vẽ hàng đợi M/M/1 with Poisson arrivals and exponential service.", "unit": "hour"},
            "problem_type": "queueing_theory",
            "arrival_rate": 2,
            "service_rate": 3,
            "servers": 1,
        }
        gate = mermaid_visualization_gate(problem, "queueing_theory")
        self.assertTrue(gate["needs_mermaid"])
        self.assertEqual(gate["diagram_type"], "queue_structure")
        solved = solve_problem(problem)
        self.assertIn("```mermaid", solved["result"]["markdown_report"])
        self.assertIn("Single server", solved["result"]["markdown_report"])

    def test_queueing_cost_optimization(self):
        problem = {
            "context": {"description": "Choose number of cashiers for an M/M/s queue.", "unit": "hour"},
            "problem_type": "queueing_theory",
            "arrival_rate": 4,
            "service_rate": 3,
            "optimize_servers": True,
            "waiting_cost": 10,
            "service_cost": 20,
            "max_servers": 5,
        }
        solved = solve_queueing_problem(problem)
        self.assertEqual(solved["status"], "optimal")
        self.assertGreaterEqual(solved["optimal_servers"], 2)
        self.assertIn("options", solved)

    def test_inventory_recognition_gate_basic_eoq(self):
        problem = {
            "context": {"description": "EOQ inventory problem", "unit": "USD/year"},
            "problem_type": "inventory_theory",
            "annual_demand": 1000,
            "order_cost": 50,
            "holding_cost": 2,
        }
        recognition = recognize_inventory_theory(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "INV_BASIC_EOQ")
        solved = solve_inventory_theory_problem(problem)
        self.assertEqual(solved["status"], "optimal")
        self.assertAlmostEqual(solved["order_quantity"], 223.606797, places=5)
        self.assertTrue(solved["verification"]["passed"])

    def test_inventory_gate_blocks_missing_holding_cost(self):
        problem = {
            "context": {"description": "EOQ inventory problem"},
            "problem_type": "inventory_theory",
            "annual_demand": 1000,
            "order_cost": 50,
        }
        result = solve_inventory_theory_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("holding_cost_h", result["missing_data"])

    def test_network_recognition_gate_shortest_path(self):
        problem = {
            "context": {"description": "Find the shortest path from A to D."},
            "problem_type": "shortest_path",
            "graph": {
                "source": "A",
                "target": "D",
                "edges": [
                    {"from": "A", "to": "B", "cost": 1},
                    {"from": "B", "to": "D", "cost": 2},
                    {"from": "A", "to": "D", "cost": 10},
                ],
            },
        }
        recognition = recognize_network_modelling(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "NET_SHORTEST_PATH")
        solved = solve_network_modelling_problem(problem)
        self.assertEqual(solved["status"], "optimal")
        self.assertEqual(solved["path"], ["A", "B", "D"])
        self.assertTrue(solved["verification"]["passed"])

    def test_network_gate_blocks_missing_target(self):
        problem = {
            "context": {"description": "Find shortest path in a route network."},
            "problem_type": "shortest_path",
            "graph": {
                "source": "A",
                "edges": [{"from": "A", "to": "B", "cost": 1}],
            },
        }
        result = solve_network_modelling_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("target_node", result["missing_data"])

    def test_nonlinear_recognition_gate_quadratic(self):
        problem = {
            "context": {"description": "Quadratic nonlinear programming with KKT conditions."},
            "problem_type": "nonlinear_programming",
            "nlp": {
                "sense": "minimize",
                "variable_names": ["x", "y"],
                "initial": [0, 0],
                "objective": {"type": "quadratic", "Q": [[2, 0], [0, 2]], "c": [-2, -4]},
                "constraints": [{"type": "ineq", "coefficients": [1, 1], "rhs": 10}],
                "bounds": [(0, None), (0, None)],
            },
        }
        recognition = recognize_nonlinear_programming(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "NLP_INEQUALITY_CONSTRAINED_KKT")
        solved = solve_nonlinear_programming_problem(problem)
        self.assertEqual(solved["status"], "optimal_local")
        self.assertAlmostEqual(solved["solution"]["x"], 1, places=4)
        self.assertTrue(solved["verification"]["passed"])

    def test_nonlinear_gate_blocks_missing_objective(self):
        problem = {
            "context": {"description": "Nonlinear programming problem with variables only."},
            "problem_type": "nonlinear_programming",
            "nlp": {"variable_names": ["x"], "initial": [0], "bounds": [(0, None)]},
        }
        result = solve_nonlinear_programming_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("objective_function", result["missing_data"])

    def test_integer_programming_recognition_gate_binary_mip(self):
        problem = {
            "context": {"description": "Binary project selection integer programming problem."},
            "problem_type": "integer_programming",
            "ip": {
                "sense": "maximize",
                "c": [10, 7],
                "A_ub": [[6, 4]],
                "b_ub": [8],
                "bounds": [(0, 1), (0, 1)],
                "integrality": [1, 1],
                "variable_names": ["project_a", "project_b"],
            },
        }
        recognition = recognize_integer_programming(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "IP_BINARY_SELECTION")
        solved = solve_integer_programming_problem(problem)
        self.assertEqual(solved["status"], "optimal")
        self.assertAlmostEqual(solved["objective_value"], 10)
        self.assertTrue(solved["verification"]["passed"])

    def test_integer_programming_gate_blocks_missing_integrality(self):
        problem = {
            "context": {"description": "Optimization model with whole-number requirements not yet specified."},
            "problem_type": "integer_programming",
            "c": [1, 1],
            "A_ub": [[1, 1]],
            "b_ub": [3],
            "bounds": [(0, None), (0, None)],
            "variable_names": ["x", "y"],
        }
        result = solve_integer_programming_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("integrality_requirements", result["missing_data"])

    def test_router_dispatches_integer_programming(self):
        problem = {
            "context": {"description": "Binary project selection integer programming problem."},
            "problem_type": "integer_programming",
            "ip": {
                "sense": "maximize",
                "c": [10, 7],
                "A_ub": [[6, 4]],
                "b_ub": [8],
                "bounds": [(0, 1), (0, 1)],
                "integrality": [1, 1],
                "variable_names": ["project_a", "project_b"],
            },
        }
        result = solve_problem(problem)
        self.assertEqual(result["problem_type"], "integer_programming")
        self.assertEqual(result["result"]["status"], "optimal")
        self.assertTrue(result["result"]["verification"]["passed"])

    def test_markov_recognition_gate_stationary_distribution(self):
        problem = {
            "context": {"description": "Markov transition matrix steady-state by month."},
            "problem_type": "markov_processes",
            "markov": {
                "states": ["A", "B"],
                "time_step": "month",
                "transition_matrix": [[0.8, 0.2], [0.1, 0.9]],
                "requested_outputs": ["stationary_distribution"],
            },
        }
        recognition = recognize_markov_processes(problem)
        self.assertTrue(recognition["can_solve"])
        self.assertEqual(recognition["subtype"], "MC_STATIONARY_DISTRIBUTION")
        solved = solve_markov_processes_problem(problem)
        self.assertEqual(solved["status"], "computed")
        self.assertAlmostEqual(solved["stationary_distribution"][0], 1 / 3, places=5)
        self.assertTrue(solved["verification"]["passed"])

    def test_router_adds_markov_state_mermaid(self):
        problem = {
            "context": {"description": "Vẽ Markov chain transition matrix steady-state by month."},
            "problem_type": "markov_processes",
            "markov": {
                "states": ["A", "B"],
                "time_step": "month",
                "transition_matrix": [[0.8, 0.2], [0.1, 0.9]],
                "requested_outputs": ["stationary_distribution"],
            },
        }
        solved = solve_problem(problem)
        self.assertEqual(solved["result"]["status"], "computed")
        self.assertIn("stateDiagram-v2", solved["result"]["markdown_report"])
        self.assertIn("A --> B: 0.2", solved["result"]["markdown_report"])

    def test_markov_n_step_distribution(self):
        problem = {
            "context": {"description": "Markov brand switching after 2 months."},
            "problem_type": "markov_processes",
            "markov": {
                "states": ["A", "B"],
                "time_step": "month",
                "transition_matrix": [[0.8, 0.2], [0.1, 0.9]],
                "initial_distribution": [1, 0],
                "n_steps": 2,
                "requested_outputs": ["n_step_probability"],
            },
        }
        solved = solve_markov_processes_problem(problem)
        self.assertEqual(solved["status"], "computed")
        self.assertEqual(solved["state_distribution_after_n"], [0.66, 0.34])
        self.assertTrue(solved["verification"]["passed"])

    def test_markov_absorbing_chain_and_first_passage(self):
        absorbing_problem = {
            "context": {"description": "Absorbing Markov chain by period."},
            "problem_type": "markov_processes",
            "markov": {
                "states": ["Start", "Middle", "Done"],
                "time_step": "period",
                "transition_matrix": [[0.5, 0.5, 0], [0.2, 0.3, 0.5], [0, 0, 1]],
                "requested_outputs": ["absorbing_chain"],
            },
        }
        absorbing = solve_markov_processes_problem(absorbing_problem)
        self.assertEqual(absorbing["status"], "computed")
        self.assertEqual(absorbing["absorbing_states"], [2])
        self.assertTrue(absorbing["verification"]["passed"])

        first_passage_problem = {
            "context": {"description": "Mean first passage time to sunny by day."},
            "problem_type": "markov_processes",
            "markov": {
                "states": ["Rainy", "Sunny"],
                "time_step": "day",
                "transition_matrix": [[0.6, 0.4], [0.2, 0.8]],
                "target_state_for_first_passage": "Sunny",
                "requested_outputs": ["first_passage_time"],
            },
        }
        first = solve_markov_processes_problem(first_passage_problem)
        self.assertEqual(first["status"], "computed")
        self.assertAlmostEqual(first["mean_first_passage_steps"]["Rainy"], 2.5)

    def test_markov_gate_blocks_missing_time_step(self):
        problem = {
            "context": {"description": "Markov transition matrix steady-state."},
            "problem_type": "markov_processes",
            "markov": {
                "states": ["A", "B"],
                "transition_matrix": [[0.8, 0.2], [0.1, 0.9]],
                "requested_outputs": ["stationary_distribution"],
            },
        }
        result = solve_markov_processes_problem(problem)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("time_step", result["missing_data"])

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

    def test_text_solver_spying_diagnostic_decision_tree(self):
        text = """
        GATACA is a firm which undertakes genetic research. Its competitor arrived before it in
        patenting important discoveries, which led to a loss of $3 million. The Board hypothesizes
        that Paul Piller is a spy and estimates that the probability of being right is 70%.
        If they dismiss Paul and he was not a spy, they will lose their best researcher, which can
        entail a loss of $12 million over the next year. The Board could make Paul do a lie detector
        test. The cost of this process is estimated at $50,000. The results would not be definitive
        because it detects only 85% of liars. Should the lie detector results favour Paul, a private
        investigator could be contracted, which would guarantee a 95% probability of success. The fee
        would come to $100,000. Solve the decision tree.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        solved = result["solved"]["result"]
        self.assertEqual(solved["solver"], "diagnostic_decision_tree")
        self.assertAlmostEqual(solved["root_value"], -1.141, places=3)
        self.assertAlmostEqual(solved["test_value"], -1.141, places=3)
        self.assertAlmostEqual(solved["followup"]["value_after_cost"], -2.694, places=3)
        self.assertAlmostEqual(solved["followup"]["max_fee"], 0.406, places=3)
        self.assertIn("```mermaid", solved["markdown_report"])

    def test_text_solver_forklift_decision_tree(self):
        text = """
        The Director of Logistics must choose a new electric forklift truck costing USD 25,000
        or a second-hand forklift truck costing USD 12,500. Maintenance costs of a new forklift
        for the next 10 years are USD 1,000, whereas second-hand maintenance costs are double.
        The proportion of faulty second-hand equipment is 20%. TEST A costs USD 1,000 and its
        diagnosis offers a percentage of failures of 5% if it is faulty, and of 20% if it operates
        properly. TEST B has a first phase costing USD 800 with probability of error 15%, and a
        second phase costing USD 700 to be absolutely sure.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        solved = result["solved"]["result"]
        self.assertEqual(solved["solver"], "forklift_decision_tree")
        self.assertEqual(solved["recommendation"]["action"], "use_test_b")
        self.assertAlmostEqual(solved["costs"]["second_hand_without_test_expected"], 19300)
        self.assertAlmostEqual(solved["test_a"]["textbook_rounded_information_value"], 530)
        self.assertAlmostEqual(solved["test_b"]["textbook_rounded_expected_cost"], 18180)
        self.assertAlmostEqual(solved["test_b"]["textbook_rounded_information_value"], 1921.2)
        self.assertAlmostEqual(solved["perfect_information"]["value"], 2500)
        self.assertIn("```mermaid", solved["markdown_report"])

    def test_text_solver_electric_car_pilot_scheme(self):
        text = """
        A firm is contemplating replacing its fleet of petrol cars with a fleet of electric cars.
        If the manufacturer is right, the firm will save USD 1,000,000.
        If the new technology fails, the change will cost the firm USD 450,000.
        A third possibility is no difference.

        | Event | Probability |
        |---|---:|
        | Money is saved | 0.25 |
        | Losses occur | 0.45 |
        | No difference | 0.30 |

        The pilot scheme will cost the firm USD 50,000.

        | Actual situation with change | Pilot scheme indicates: Savings | Pilot scheme indicates: No change | Pilot scheme indicates: Losses |
        |---|---:|---:|---:|
        | Money is saved | 0.6 | 0.3 | 0.1 |
        | No difference | 0.4 | 0.4 | 0.2 |
        | Losses | 0.1 | 0.5 | 0.4 |
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        solved = result["solved"]["result"]
        self.assertEqual(solved["solver"], "imperfect_information_decision_tree")
        self.assertAlmostEqual(solved["without_information"]["expected_value"], 47500)
        self.assertAlmostEqual(solved["with_sample_information"]["expected_value_before_cost"], 129750)
        self.assertAlmostEqual(solved["with_sample_information"]["expected_value_after_cost"], 79750)
        self.assertAlmostEqual(solved["maximum_information_cost"], 82250)
        self.assertEqual(solved["recommendation"]["action"], "with_pilot")
        self.assertIn("```mermaid", solved["markdown_report"])

    def test_text_solver_binary_market_research_cost_decision(self):
        text = """
        A firm is deciding whether to build an assembly plant in Brazil or in Mississippi, USA.
        | Location | Construction cost |
        |---|---:|
        | Brazil | USD 10 million |
        | Mississippi | USD 20 million |

        If the firm builds this plant in Brazil and local demand drops over the following 5 years,
        the project will be stopped and the firm will lose USD 10 million. It will still have to
        build a plant in Mississippi. The probability of demand dropping is 20%.
        For USD 1 million, it can contract a market research firm.
        The firm will predict the occurrence of a drop in demand 95% of the time when a drop actually occurs.
        It will predict that a drop will not occur 90% of the time when a drop actually does not occur.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        solved = result["solved"]["result"]
        self.assertEqual(solved["solver"], "imperfect_information_decision_tree")
        self.assertAlmostEqual(solved["without_information"]["expected_value"], 14.0)
        self.assertAlmostEqual(solved["with_sample_information"]["expected_value_before_cost"], 12.9)
        self.assertAlmostEqual(solved["with_sample_information"]["expected_value_after_cost"], 13.9)
        self.assertAlmostEqual(solved["sample_information_value"], 1.1)
        self.assertAlmostEqual(solved["perfect_information"]["value"], 2.0)
        self.assertEqual(solved["recommendation"]["action"], "with_pilot")

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

    def test_game_theory_zero_sum_saddle_point(self):
        problem = {
            "context": {"description": "zero-sum payoff matrix game"},
            "game": {
                "players": ["A", "B"],
                "row_player": "A",
                "column_player": "B",
                "row_strategies": ["A1", "A2"],
                "column_strategies": ["B1", "B2"],
                "payoff_orientation": "payoff to row player",
                "is_zero_sum": True,
                "payoff_matrix": [[3, 1], [4, 2]],
            },
        }
        result = solve_game_theory(problem)
        self.assertEqual(result["status"], "computed")
        self.assertTrue(result["pure_strategy_check"]["has_saddle_point"])
        self.assertEqual(result["solution"]["maximin_strategy"], "A2")
        self.assertEqual(result["solution"]["minimax_strategy"], "B2")
        self.assertAlmostEqual(result["solution"]["maximin"], 2)

    def test_game_theory_zero_sum_2x2_mixed(self):
        problem = {
            "context": {"description": "zero-sum mixed strategy game"},
            "game": {
                "players": ["A", "B"],
                "row_player": "A",
                "column_player": "B",
                "row_strategies": ["A1", "A2"],
                "column_strategies": ["B1", "B2"],
                "payoff_orientation": "payoff to row player",
                "is_zero_sum": True,
                "payoff_matrix": [[2, -1], [-2, 1]],
            },
        }
        result = solve_game_theory(problem)
        self.assertEqual(result["status"], "computed")
        self.assertEqual(result["solution"]["type"], "mixed_2x2")
        self.assertAlmostEqual(result["solution"]["row_probabilities"]["A1"], 0.5)
        self.assertAlmostEqual(result["solution"]["column_probabilities"]["B1"], 1 / 3)
        self.assertAlmostEqual(result["solution"]["game_value"], 0)

    def test_text_solver_building_encounter_game(self):
        text = """
        Encounter in a Building - Games Theory Problem.
        Two people are in opposite corners. Each time, both move simultaneously:
        Move to the right, R; Move to the left, L; Remain still, O.
        If both move towards the same side, they do not meet. They meet if
        they move towards opposite sides or one waits while the other moves.
        The LP solution is x1 = x2 = x3 = 1/2, z = 3/2,
        y1 = y2 = y3 = 1/2, w = 3/2.
        """
        result = solve_text_problem(text)
        self.assertEqual(result["status"], "solved")
        solved = result["solved"]["result"]
        self.assertEqual(solved["solver"], "game_theory")
        self.assertAlmostEqual(solved["solution"]["row_probabilities"]["R"], 1 / 3)
        self.assertAlmostEqual(solved["solution"]["column_probabilities"]["R"], 1 / 3)
        self.assertAlmostEqual(solved["solution"]["game_value"], 2 / 3)
        self.assertAlmostEqual(solved["solution"]["expected_rounds_to_meet"], 3 / 2)

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
