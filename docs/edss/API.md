# EDSS API

## POST `/edss/classify`

Input:

```json
{ "description": "Allocate production between products A and B...", "domain": "manufacturing" }
```

Output: problem type, confidence, keyword scores and reason.

## POST `/edss/clarify`

Input: `EDSSProblem`.

Output: classification and missing-data questions.

## POST `/edss/model/build`

Input: `EDSSProblem`.

Output: mathematical model formulation, assumptions and missing data.

## POST `/edss/solve`

Input: `EDSSProblem`.

Output: solver result, model, recommendation explanation.

## POST `/edss/solve/lp`

Runs LP solver directly.

## POST `/edss/solve/shortest-path`

Runs Dijkstra shortest path directly.

## POST `/edss/report`

Returns Markdown report for a solved problem.
