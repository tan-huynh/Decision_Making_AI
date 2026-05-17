# AI-Powered Decision Making

Browser-based local decision analysis tool using Next.js, D3, Python/FastAPI, and local Ollama.

## Run

```bash
chmod +x start.sh
./start.sh
```

The script checks Node.js, npm, Python, frontend packages, backend virtualenv packages, Ollama availability, and common port conflicts before starting the app.

Manual run:

```bash
npm install
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
npm run dev
```

Open `http://localhost:3000`.

Ollama should be running locally at `http://127.0.0.1:11434`. Pull at least one model, for example:

```bash
ollama pull llama3.1
```

## What It Does

- Models choices, scenarios, probabilities, utilities, cost, reversibility, and risk tolerance.
- Computes expected utility, risk-adjusted score, regret, worst/best case, sensitivity, and value of information.
- Uses D3 to visualize scores, regret, and decision trees.
- Uses Ollama for qualitative critique and recommendation review.
- Runs an agentic research loop for many life domains: career, education, earning money, business, finance, relationships, health, legal, safety, life experience, and personal choices.
- Optionally pulls live web evidence through the backend and passes it into the AI review.
- Learns from outcomes by saving cases and updating calibration in `backend/data/expert_profile.json`.

## Engineering Decision Support System

The backend now includes a solver-first EDSS layer for engineering decision problems:

- Canonical EDSS data model: context, alternatives, variables, objective, constraints, states, probabilities, payoff matrix, criteria, sensitivity and recommendation.
- Solver router: classifies LP, transportation, assignment, shortest path, dynamic programming, decision tree, simulation/risk and multi-objective problems.
- Solver engines:
  - LP vertex solver with slacks, binding constraints and finite-difference shadow prices.
  - Exact DP assignment solver.
  - Dijkstra shortest path and transportation heuristic.
  - Finite-horizon dynamic programming.
  - Decision tree rollback, expected payoff and EVPI.
  - Monte Carlo simulation and risk summaries.
  - Weighted multi-objective scoring and Pareto frontier.
- Report layer: Markdown report from structured case and solver result.
- PostgreSQL schema: `backend/edss/sql/schema.sql`.
- Ontology/RAG docs: `docs/edss/ONTOLOGY.md`.
- Example production mix case: `docs/edss/EXAMPLE_PRODUCTION_MIX.json`.
- Rust scaffold for future high-performance kernels: `backend/edss/rust_core`.

Key endpoints:

```text
POST /edss/classify
POST /edss/clarify
POST /edss/model/build
POST /edss/solve
POST /edss/solve/lp
POST /edss/solve/shortest-path
POST /edss/report
```

## PDF Basis

The provided Google Drive PDF was downloaded as `decision_book.pdf` and extracted into `decision_book.txt`. The implementation notes are in `BOOK_NOTES.md`.
