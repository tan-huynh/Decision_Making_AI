# EDSS Technical Risks

| Risk | Failure Mode | Mitigation |
|---|---|---|
| LLM creates invalid model | Wrong signs, missing constraints | Model validator, user confirmation, unit checks |
| Solver infeasible | No solution returned | Infeasibility diagnosis and missing-data questions |
| LP MVP scale limit | Vertex enumeration grows combinatorially | Route large LP to OR-Tools/Pyomo |
| Data uncertainty hidden | False confidence | Required states/probability/risk panel |
| Unit mismatch | Invalid objective | Unit normalization and schema metadata |
| Outcome luck confused with decision quality | Wrong learning signal | Store model assumptions and realized outcome separately |
| RAG hallucination | Wrong concept cited | Cite chunk/page and source score |
| Web data unreliable | Poor decision evidence | Domain authority scoring and official-source preference |
