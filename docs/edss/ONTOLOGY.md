# Engineering Decision Making Ontology

Core concept graph:

- Optimization
  - Linear Programming: maximize/minimize `c^T x` subject to `Ax <= b`.
  - Simplex: vertex search over feasible polytope.
  - Duality: marginal value of constraints and resources.
  - Sensitivity: effect of objective/RHS changes on solution.
- Network Models
  - Transportation: supply-demand cost minimization.
  - Transshipment: network flow through intermediate nodes.
  - Shortest Path: minimum path cost over graph.
  - Assignment: one-to-one matching with minimum cost.
- Sequential Decision
  - Dynamic Programming: Bellman recursion over states and stages.
  - Inventory Planning: stage demand, holding, shortage and order cost.
  - Maintenance Planning: replace/repair/operate policy.
- Uncertainty
  - Probability Model: states of nature and probability assumptions.
  - Bayesian Update: posterior = likelihood * prior / evidence.
  - Decision Tree: rollback expected value from outcomes.
  - Value of Information: EVPI = EVwPI - EVwoPI.
- Simulation and Risk
  - Monte Carlo: repeated sampling from uncertain model.
  - VaR: lower percentile of payoff/loss distribution.
  - CVaR: expected value in the downside tail.
  - Regret: opportunity loss relative to best state-contingent action.
- Multi-objective Decision
  - Weighted Score: `score(a)=sum_k w_k f_k(a)`.
  - Pareto Frontier: non-dominated alternatives.
  - Fuzzy Scoring: soft membership for qualitative objectives.

RAG workflow:

1. Classify problem type.
2. Retrieve PDF/book chunks matching the type and variables.
3. Retrieve formulas and examples from this ontology.
4. Build a mathematical model draft.
5. Validate units, constraints, signs and missing data.
6. Run solver, then cite concepts used in explanation.
