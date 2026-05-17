# Engineering Decision Support System Roadmap

## 4-week MVP

Week 1:
- EDSS canonical data model.
- Classification and clarification endpoints.
- LP model builder.
- PostgreSQL schema.
- RAG hooks to decision-making PDF.

Week 2:
- LP solver with slack, binding constraints and approximate shadow prices.
- Assignment solver.
- Shortest path and transportation heuristic.
- Markdown report endpoint.

Week 3:
- Decision tree rollback.
- EVPI/EVSI baseline.
- Monte Carlo simulation and risk summaries.
- Multi-objective weighted scoring and Pareto frontier.

Week 4:
- Dashboard integration.
- Example templates.
- API integration tests.
- Report export.

## 3-month advanced version

- Replace MVP pure-Python solvers with OR-Tools/Pyomo/NetworkX where appropriate.
- Add MILP and integer production lots.
- Add full min-cost flow and transshipment.
- Add stochastic programming templates.
- Persist cases/results in PostgreSQL.
- Add pgvector-backed RAG.
- Add chart-first solver report UI.
- Add audit trail distinguishing decision quality from realized outcome luck.
