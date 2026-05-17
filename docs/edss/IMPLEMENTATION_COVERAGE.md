# EDSS Implementation Coverage

This file maps the requested Engineering Decision Support System requirements to the codebase.

1. Canonical decision problem schema: `backend/edss/models.py`.
2. Analysis engines:
   - LP/simplex-style vertex solver: `backend/edss/linear_programming.py`
   - Duality/sensitivity approximation: shadow prices and slacks in LP result
   - Network flow/shortest path: `backend/edss/network.py`
   - Assignment/Hungarian-class exact matching: `backend/edss/assignment.py`
   - Dynamic programming: `backend/edss/dynamic_programming.py`
   - Probability/Bayesian/expected payoff: `backend/edss/uncertainty.py`
   - Decision tree rollback: `backend/edss/uncertainty.py`
   - EVPI baseline: `backend/edss/uncertainty.py`
   - Simulation/risk: `backend/edss/uncertainty.py`, `backend/edss/risk.py`
   - Multi-objective/Pareto: `backend/edss/multiobjective.py`
3. LLM boundary: existing Ollama layer only explains/asks/generates; solver layer performs computation.
4. Architecture:
   - Frontend: Next.js dashboard
   - Backend: FastAPI
   - Database schema: `backend/edss/sql/schema.sql`
   - Knowledge layer: `docs/edss/ONTOLOGY.md`, `backend/book_rag.py`
   - Solver layer: `backend/edss/*`
   - Report layer: `backend/edss/report.py`
5. Workflow:
   - Classify: `/edss/classify`
   - Clarify: `/edss/clarify`
   - Build model: `/edss/model/build`
   - Solve: `/edss/solve`
   - Report: `/edss/report`
6. Detailed proposal docs: `docs/edss/API.md`, `docs/edss/MVP_PLAN.md`, `docs/edss/TECHNICAL_RISKS.md`.
7. Engineering guardrails:
   - Missing data questions in `backend/edss/classifier.py`
   - Recommendation explanations in `backend/edss/router.py`
   - Assumptions and sensitivity in solver outputs and reports
   - Decision quality vs outcome luck noted in `backend/edss/report.py`

MVP limitation: pure-Python solvers are intentionally scoped for small/medium cases. For large industrial cases, route to OR-Tools/Pyomo/NetworkX using the same data model.
