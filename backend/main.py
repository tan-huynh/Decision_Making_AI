from __future__ import annotations

from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agentic_research import run_agentic_research
from autonomous import record_autonomous_learning, select_decision_route, select_text_solver_route
from book_rag import book_context_for_decision
from clarification import build_clarifying_questions
from decision_engine import compute_results, learn_from_outcome
from edss.classifier import classify_problem, missing_data_questions
from edss.data_estimation import build_histogram, compare_groups, descriptive_stats, empirical_cdf
from edss.linear_programming import solve_lp
from edss.markov import absorbing_chain, n_step_transition, steady_state
from edss.model_validator import validate_model
from edss.module_registry import module_coverage_report
from edss.network import solve_bellman_ford, solve_minimum_spanning_tree, solve_shortest_path, solve_transportation, solve_max_flow, solve_min_cost_flow
from edss.probability_engine import bayes_multi_step, bayes_update, distribution_pmf_pdf, distribution_quantile, fit_distribution, independent_events
from edss.queueing import solve_mg1, solve_mm1, solve_mm1k, solve_mmc, solve_open_queue_network
from edss.report import render_markdown_report
from edss.risk import value_at_risk
from edss.router import build_mathematical_model, solve_problem
from edss.sensitivity_engine import sensitivity_analysis, what_if_scenario
from edss.text_solver import solve_text_problem, solve_text_problem_async
from edss.utility import expected_utility, exponential_certainty_equivalent, fit_exponential_risk_tolerance
from edss.voi_engine import compute_evpi, compute_evi, compute_voi_from_problem
from knowledge_base import BOOK_PRINCIPLES, decision_system_prompt
from model_generator import generate_decision_model
from monte_carlo import run_monte_carlo
from safety import safety_assessment
from schemas import DecisionRequest, GenerateModelRequest, LearnRequest
from edss.models import EDSSProblem, NaturalProblemRequest


app = FastAPI(title="AI-Powered Decision Making Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def call_ollama(model: str, system: str, prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0.2, "num_predict": 700},
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
    except Exception as exc:
        return f"Ollama chưa phản hồi ({exc}). Kết quả định lượng vẫn được tính bằng engine nội bộ."


@app.get("/ollama-status")
async def ollama_status() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get("http://127.0.0.1:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [item.get("name", "") for item in data.get("models", []) if item.get("name")]
            return {"ok": True, "models": models}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc)}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "decision-making-ai", "version": "0.1.0"}


@app.get("/edss/modules")
async def edss_modules() -> dict[str, Any]:
    return module_coverage_report()


@app.post("/edss/classify")
async def edss_classify(request: NaturalProblemRequest) -> dict[str, Any]:
    return classify_problem(request.description, {"context": {"domain": request.domain, "description": request.description}})


@app.post("/edss/clarify")
async def edss_clarify(problem: EDSSProblem) -> dict[str, Any]:
    payload = problem.model_dump()
    classification = classify_problem(payload.get("context", {}).get("description", ""), payload)
    payload["problem_type"] = payload.get("problem_type") or classification["problem_type"]
    return {"classification": classification, "questions": missing_data_questions(payload)}


@app.post("/edss/model/build")
async def edss_model_build(problem: EDSSProblem) -> dict[str, Any]:
    return build_mathematical_model(problem.model_dump())


@app.post("/edss/solve")
async def edss_solve(problem: EDSSProblem) -> dict[str, Any]:
    try:
        return solve_problem(problem.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/edss/solve-text")
async def edss_solve_text(payload: dict[str, str]) -> dict[str, Any]:
    text = payload.get("text", "")
    if len(text.strip()) < 4:
        raise HTTPException(status_code=400, detail="Missing problem text.")
    autonomy = select_text_solver_route(text)
    result = await solve_text_problem_async(text)
    profile = record_autonomous_learning(
        {
            **autonomy,
            "source": "edss_solve_text",
            "problem_preview": text[:500],
            "status": result.get("status"),
        }
    )
    result["autonomy"] = {**autonomy, "learning_profile": profile}
    return result


@app.post("/edss/solve/lp")
async def edss_solve_lp(problem: EDSSProblem) -> dict[str, Any]:
    try:
        return solve_lp(problem.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/edss/solve/lp/simplex")
async def edss_lp_simplex(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.lp_advanced import simplex_step_by_step

    return simplex_step_by_step(payload)


@app.post("/edss/solve/lp/duality")
async def edss_lp_duality(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.lp_advanced import duality_analysis

    return duality_analysis(payload)


@app.post("/edss/solve/lp/sensitivity-ranges")
async def edss_lp_sensitivity_ranges(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.lp_advanced import sensitivity_ranges

    return sensitivity_ranges(payload)


@app.post("/edss/solve/mip")
async def edss_solve_mip(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.mip import solve_mip

    return solve_mip(payload)


@app.post("/edss/solve/goal-programming")
async def edss_solve_goal_programming(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.goal_programming import solve_goal_programming

    return solve_goal_programming(payload)


@app.post("/edss/solve/shortest-path")
async def edss_shortest_path(problem: EDSSProblem) -> dict[str, Any]:
    try:
        return solve_shortest_path(problem.model_dump().get("graph", {}))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/edss/solve/bellman-ford")
async def edss_bellman_ford(payload: dict[str, Any]) -> dict[str, Any]:
    return solve_bellman_ford(payload.get("graph", payload))


@app.post("/edss/solve/mst")
async def edss_mst(payload: dict[str, Any]) -> dict[str, Any]:
    return solve_minimum_spanning_tree(payload.get("graph", payload), str(payload.get("algorithm", "kruskal")))


@app.post("/edss/report")
async def edss_report(problem: EDSSProblem) -> dict[str, str]:
    payload = problem.model_dump()
    solved = solve_problem(payload)
    return {"format": "markdown", "report": render_markdown_report(payload, solved)}


@app.post("/edss/validate")
async def edss_validate(problem: EDSSProblem) -> dict[str, Any]:
    """Validate a model before solving."""
    return validate_model(problem.model_dump())


@app.post("/edss/probability/bayes")
async def edss_bayes(payload: dict[str, Any]) -> dict[str, Any]:
    """Bayesian update: single or multi-step."""
    tests = payload.get("tests")
    if tests:
        return bayes_multi_step(
            prior=float(payload.get("prior", 0.5)),
            tests=tests,
        )
    return bayes_update(
        prior=float(payload.get("prior", 0.5)),
        sensitivity=float(payload.get("sensitivity", 0.9)),
        false_positive_rate=float(payload.get("false_positive_rate", 0.1)),
        observed_positive=bool(payload.get("observed_positive", True)),
    )


@app.post("/edss/probability/distribution")
async def edss_distribution(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute PMF/PDF/CDF for a distribution."""
    return distribution_pmf_pdf(
        distribution=str(payload.get("distribution", "normal")),
        params=payload.get("params", {}),
        x=payload.get("x", [0]),
    )


@app.post("/edss/probability/independent")
async def edss_independent(payload: dict[str, Any]) -> dict[str, Any]:
    """Joint probability of independent events."""
    return independent_events(payload.get("probabilities", []))


@app.post("/edss/probability/fit")
async def edss_fit(payload: dict[str, Any]) -> dict[str, Any]:
    """Fit distributions to data."""
    return fit_distribution(
        data=payload.get("data", []),
        candidates=payload.get("candidates"),
    )


@app.post("/edss/data/stats")
async def edss_data_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Descriptive statistics for a dataset."""
    return descriptive_stats(
        data=payload.get("data", []),
        label=str(payload.get("label", "data")),
    )


@app.post("/edss/data/histogram")
async def edss_histogram(payload: dict[str, Any]) -> dict[str, Any]:
    """Build histogram from data."""
    return build_histogram(
        data=payload.get("data", []),
        n_bins=payload.get("n_bins"),
        label=str(payload.get("label", "data")),
    )


@app.post("/edss/data/compare")
async def edss_compare(payload: dict[str, Any]) -> dict[str, Any]:
    """Compare multiple groups/suppliers."""
    return compare_groups(payload.get("groups", {}))


@app.post("/edss/sensitivity")
async def edss_sensitivity(problem: EDSSProblem) -> dict[str, Any]:
    """Run sensitivity analysis on a solved problem."""
    p = problem.model_dump()
    solved = solve_problem(p)
    result = solved.get("result", {})
    return sensitivity_analysis(p, result, solve_fn=solve_lp if solved.get("problem_type") == "linear_programming" else None)


@app.post("/edss/risk/var")
async def edss_var(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute VaR and CVaR from samples."""
    return value_at_risk(
        samples=payload.get("samples", []),
        confidence=float(payload.get("confidence", 0.95)),
    )


@app.post("/edss/inventory/eoq")
async def edss_inventory_eoq(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.inventory import solve_eoq

    return solve_eoq(
        demand=float(payload["demand"]),
        ordering_cost=float(payload["ordering_cost"]),
        holding_cost=float(payload["holding_cost"]),
    )


@app.post("/edss/inventory/production-lot-size")
async def edss_inventory_production_lot_size(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.inventory import solve_production_lot_size

    return solve_production_lot_size(
        demand=float(payload["demand"]),
        setup_cost=float(payload["setup_cost"]),
        holding_cost=float(payload["holding_cost"]),
        production_rate=float(payload["production_rate"]),
    )


@app.post("/edss/inventory/reorder-point")
async def edss_inventory_reorder_point(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.inventory import solve_reorder_point

    return solve_reorder_point(
        demand_rate=float(payload["demand_rate"]),
        lead_time=float(payload["lead_time"]),
        safety_stock=float(payload.get("safety_stock", 0)),
    )


@app.post("/edss/inventory/newsvendor")
async def edss_inventory_newsvendor(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.inventory import solve_newsvendor

    return solve_newsvendor(
        unit_cost=float(payload["unit_cost"]),
        selling_price=float(payload["selling_price"]),
        salvage_value=float(payload.get("salvage_value", 0)),
        demand_mean=float(payload["demand_mean"]) if payload.get("demand_mean") is not None else None,
        demand_std=float(payload["demand_std"]) if payload.get("demand_std") is not None else None,
    )


@app.post("/edss/inventory/quantity-discount")
async def edss_inventory_quantity_discount(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.inventory import solve_quantity_discount_eoq

    return solve_quantity_discount_eoq(
        demand=float(payload["demand"]),
        ordering_cost=float(payload["ordering_cost"]),
        holding_rate=float(payload["holding_rate"]),
        price_breaks=payload.get("price_breaks", []),
    )


@app.post("/edss/inventory/dp")
async def edss_inventory_dp(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.dynamic_programming import solve_inventory_dp

    return solve_inventory_dp(payload)


@app.post("/edss/inventory/stochastic-sim")
async def edss_inventory_stochastic_sim(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.inventory import simulate_stochastic_inventory

    return simulate_stochastic_inventory(
        periods=int(payload["periods"]),
        initial_inventory=int(payload.get("initial_inventory", 0)),
        reorder_point=int(payload["reorder_point"]),
        order_up_to=int(payload["order_up_to"]),
        demand_mean=float(payload["demand_mean"]),
        demand_std=float(payload.get("demand_std", 0)),
        holding_cost=float(payload.get("holding_cost", 0)),
        shortage_cost=float(payload.get("shortage_cost", 0)),
        ordering_cost=float(payload.get("ordering_cost", 0)),
        fixed_order_cost=float(payload.get("fixed_order_cost", 0)),
        lead_time=int(payload.get("lead_time", 0)),
        seed=payload.get("seed"),
    )


@app.post("/edss/queueing/mm1")
async def edss_queueing_mm1(payload: dict[str, Any]) -> dict[str, Any]:
    return solve_mm1(float(payload["arrival_rate"]), float(payload["service_rate"]))


@app.post("/edss/queueing/mmc")
async def edss_queueing_mmc(payload: dict[str, Any]) -> dict[str, Any]:
    return solve_mmc(float(payload["arrival_rate"]), float(payload["service_rate"]), int(payload["servers"]))


@app.post("/edss/queueing/mm1k")
async def edss_queueing_mm1k(payload: dict[str, Any]) -> dict[str, Any]:
    return solve_mm1k(float(payload["arrival_rate"]), float(payload["service_rate"]), int(payload["capacity"]))


@app.post("/edss/queueing/mg1")
async def edss_queueing_mg1(payload: dict[str, Any]) -> dict[str, Any]:
    return solve_mg1(float(payload["arrival_rate"]), float(payload["mean_service_time"]), float(payload["service_time_variance"]))


@app.post("/edss/queueing/network")
async def edss_queueing_network(payload: dict[str, Any]) -> dict[str, Any]:
    return solve_open_queue_network(payload.get("nodes", []), payload.get("routing_matrix", []), payload.get("external_arrivals", []))


@app.post("/edss/markov/n-step")
async def edss_markov_n_step(payload: dict[str, Any]) -> dict[str, Any]:
    return n_step_transition(payload["matrix"], int(payload.get("steps", 1)), payload.get("initial"))


@app.post("/edss/markov/steady-state")
async def edss_markov_steady_state(payload: dict[str, Any]) -> dict[str, Any]:
    return steady_state(payload["matrix"])


@app.post("/edss/markov/absorbing")
async def edss_markov_absorbing(payload: dict[str, Any]) -> dict[str, Any]:
    return absorbing_chain(payload["matrix"], payload.get("absorbing_indices", []))


@app.post("/edss/utility/expected")
async def edss_utility_expected(payload: dict[str, Any]) -> dict[str, Any]:
    return expected_utility(payload.get("outcomes", []))


@app.post("/edss/utility/exponential-ce")
async def edss_utility_exponential_ce(payload: dict[str, Any]) -> dict[str, Any]:
    return exponential_certainty_equivalent(payload.get("outcomes", []), float(payload["risk_tolerance"]))


@app.post("/edss/utility/fit-risk-tolerance")
async def edss_utility_fit_risk_tolerance(payload: dict[str, Any]) -> dict[str, Any]:
    return fit_exponential_risk_tolerance(payload.get("certainty_equivalents", []))


@app.post("/edss/mada/additive-utility")
async def edss_mada_additive_utility(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.multiobjective import additive_utility

    return additive_utility(payload)


@app.post("/edss/mada/ahp")
async def edss_mada_ahp(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.multiobjective import ahp_weights

    return ahp_weights(payload.get("pairwise_matrix", []), payload.get("criteria"))


@app.post("/edss/behavioral/audit")
async def edss_behavioral_audit(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.behavioral import behavioral_bias_audit

    return behavioral_bias_audit(str(payload.get("description", "")), payload.get("judgments"))


@app.post("/edss/solve/nonlinear")
async def edss_solve_nonlinear(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.nonlinear import solve_nonlinear

    return solve_nonlinear(payload)


@app.post("/edss/influence-diagram/analyze")
async def edss_influence_diagram(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.influence_diagram import analyze_influence_diagram

    return analyze_influence_diagram(payload)


@app.post("/edss/influence-diagram/to-tree")
async def edss_influence_to_tree(payload: dict[str, Any]) -> dict[str, Any]:
    from edss.influence_diagram import influence_to_decision_tree

    return influence_to_decision_tree(payload)


@app.post("/edss/information/evpi")
async def edss_evpi(problem: EDSSProblem) -> dict[str, Any]:
    """Compute EVPI and optionally EVI."""
    return compute_voi_from_problem(problem.model_dump())


@app.post("/edss/information/evi")
async def edss_evi(payload: dict[str, Any]) -> dict[str, Any]:
    alternatives = [item["name"] if isinstance(item, dict) else str(item) for item in payload.get("alternatives", [])]
    payoff_lookup = {(cell["alternative"], cell["state"]): float(cell.get("payoff", 0)) for cell in payload.get("payoff_matrix", [])}
    return compute_evi(
        alternatives=alternatives,
        states=payload.get("states", []),
        payoff_lookup=payoff_lookup,
        test_sensitivity=payload.get("test_sensitivity", {}),
        test_cost=float(payload.get("test_cost", 0)),
    )


@app.post("/edss/solve/max-flow")
async def edss_max_flow(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve maximum flow problem."""
    return solve_max_flow(payload.get("graph", payload))


@app.post("/edss/solve/min-cost-flow")
async def edss_min_cost_flow(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve minimum cost flow problem."""
    return solve_min_cost_flow(payload.get("graph", payload))


@app.post("/edss/solve/transportation")
async def edss_transportation(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve transportation problem with auto-balancing."""
    return solve_transportation(payload)


@app.post("/edss/solve/assignment")
async def edss_assignment(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve assignment problem (Hungarian method)."""
    from edss.assignment import solve_assignment
    return solve_assignment(
        costs=payload.get("costs", payload.get("assignment_costs", [])),
        maximize=bool(payload.get("maximize", False)),
    )


@app.post("/edss/decision-criteria")
async def edss_decision_criteria(problem: EDSSProblem) -> dict[str, Any]:
    """Apply all decision criteria (maximin, maximax, minimax regret, Hurwicz, EV)."""
    from edss.uncertainty import decision_criteria
    return decision_criteria(
        problem.model_dump(),
        alpha=float(problem.model_dump().get("context", {}).get("hurwicz_alpha", 0.5)),
    )


def safety_warnings(domain: str) -> list[str]:
    if domain == "health":
        return [
            "Quyết định về thuốc có thể nguy hiểm nếu thiếu bệnh nền, liều, dị ứng và thuốc dùng chung. Hãy xác nhận với bác sĩ/dược sĩ, đặc biệt khi có triệu chứng nặng.",
        ]
    if domain in {"legal", "safety", "finance"}:
        return [
            "Đây là lĩnh vực high-stakes. Dùng kết quả như hỗ trợ phân tích, không thay thế chuyên gia có thẩm quyền.",
        ]
    return []


@app.post("/generate-model")
async def generate_model(request: GenerateModelRequest) -> dict[str, Any]:
    return generate_decision_model(request.model_dump())


@app.post("/analyze")
async def analyze(request: DecisionRequest) -> dict[str, Any]:
    payload = request.model_dump()
    autonomy = select_decision_route(payload)
    safety = safety_assessment(payload)
    try:
        quantitative = compute_results(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    book = book_context_for_decision(payload)
    web = await run_agentic_research(payload)
    research_context = "\n\n".join(part for part in [book.get("summary", ""), web.get("summary", "")] if part)
    system = decision_system_prompt(research_context)
    simulation = run_monte_carlo(payload)
    prompt = {
        "task": "Run an agentic review: verify missing info, update beliefs with evidence, critique the quantitative decision, and decide whether to ask more before final action.",
        "question": payload.get("question"),
        "domain": payload.get("domain"),
        "objective": payload.get("objective"),
        "context": payload.get("context"),
        "time_horizon": payload.get("timeHorizon"),
        "book_principles": BOOK_PRINCIPLES,
        "book_rag": book.get("results", []),
        "safety": safety,
        "quantitative_result": quantitative,
        "monte_carlo": simulation,
        "agent_trace": web.get("agent_trace", []),
        "web_results": web.get("results", []),
        "market_data": web.get("market_data", []),
    }
    ai_review = await call_ollama(payload.get("model", "llama3.1"), system, str(prompt))
    clarifying_questions = build_clarifying_questions(payload, quantitative)
    trace = [
        {"step": "autonomous_route", "output": autonomy},
        {"step": "clarify", "output": {"questions": clarifying_questions, "safety": safety}},
        {"step": "retrieve_book_rag", "output": [f"p.{item['page']} {item['section']}" for item in book.get("results", [])]},
        *web.get("agent_trace", []),
        {"step": "monte_carlo", "output": simulation.get("distributions", [])[:3]},
        {"step": "decide", "output": quantitative.get("recommendation")},
    ]
    auto_profile = record_autonomous_learning(
        {
            **autonomy,
            "source": "decision_analyze",
            "problem_preview": payload.get("question", "")[:500],
            "recommendation": quantitative.get("recommendation"),
            "decision_gate": "needs_clarification" if safety.get("requires_clarification") else "ready",
        }
    )
    return {
        **quantitative,
        "warnings": safety.get("warnings", []) + safety_warnings(payload.get("domain", "life")) + quantitative["warnings"],
        "ai_review": ai_review,
        "book_rag": book,
        "web_research": web,
        "agent_trace": trace,
        "market_data": web.get("market_data", []),
        "monte_carlo": simulation,
        "safety": safety,
        "decision_gate": "needs_clarification" if safety.get("requires_clarification") else "ready",
        "clarifying_questions": clarifying_questions,
        "autonomy": {**autonomy, "learning_profile": auto_profile},
    }


@app.post("/learn")
async def learn(request: LearnRequest) -> dict[str, Any]:
    return learn_from_outcome(request.model_dump())


if __name__ == "__main__":
    import os

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("BACKEND_PORT", "8008")), reload=True)
