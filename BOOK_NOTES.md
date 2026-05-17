# Algorithms for Decision Making - implementation notes

Source PDF: `decision_book.pdf`, extracted to `decision_book.txt`.

The app uses the book as an algorithmic foundation rather than copying book text. Main concepts mapped into the product:

- Probabilistic reasoning: every scenario has an explicit probability, and scenario probabilities are normalized before scoring.
- Bayesian mindset: new evidence from web/context and saved outcomes should update beliefs instead of replacing the full decision model.
- Expected utility: each option is scored by probability-weighted utility minus cost.
- Risk attitude: user risk tolerance changes the penalty applied to variance/downside.
- Decision trees: frontend visualizes options and possible outcomes as action-to-scenario branches.
- Regret: the engine estimates expected regret against the best payoff under comparable scenario positions.
- Value of information: when the top options are close, the app flags that more information is valuable.
- Sensitivity analysis: highest-impact uncertainties are surfaced so the user knows what to validate first.
- Sequential learning: the `/learn` endpoint stores outcomes and updates a small calibration profile for future analyses.

Important product boundary: this is a decision-support tool, not an oracle. For medical, legal, safety, or high-stakes financial decisions, it should surface uncertainty and recommend a qualified professional.
