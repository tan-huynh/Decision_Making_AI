"""Probability Engine — Bayes rule, distributions, conditional probability.

Provides exact probability calculations using scipy.stats. LLM is never
used for numerical computation; this module is the single source of truth
for probability values.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Attempt to import scipy.stats; fall back to pure-python basics if missing.
# ---------------------------------------------------------------------------
try:
    from scipy import stats as sp_stats  # type: ignore[import-untyped]

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Bayes Rule ────────────────────────────────────────────────────────────


def bayes_update(
    prior: float,
    sensitivity: float,
    false_positive_rate: float,
    observed_positive: bool = True,
) -> dict[str, Any]:
    """Single-step Bayesian update.

    Parameters
    ----------
    prior : float
        P(H) — prior probability of the hypothesis.
    sensitivity : float
        P(E+|H) — probability of positive evidence given hypothesis true.
    false_positive_rate : float
        P(E+|¬H) — probability of positive evidence given hypothesis false.
    observed_positive : bool
        Whether the evidence observed is positive (True) or negative (False).

    Returns
    -------
    dict with prior, posterior, evidence_probability, likelihood_ratio.
    """
    prior = _clamp(prior)
    sensitivity = _clamp(sensitivity)
    fpr = _clamp(false_positive_rate)

    if observed_positive:
        p_evidence = sensitivity * prior + fpr * (1 - prior)
        posterior = (sensitivity * prior / p_evidence) if p_evidence else prior
        lr = (sensitivity / fpr) if fpr else float("inf")
    else:
        p_evidence = (1 - sensitivity) * prior + (1 - fpr) * (1 - prior)
        posterior = ((1 - sensitivity) * prior / p_evidence) if p_evidence else prior
        lr = ((1 - sensitivity) / (1 - fpr)) if (1 - fpr) else float("inf")

    return {
        "prior": prior,
        "posterior": round(posterior, 6),
        "evidence_probability": round(p_evidence, 6),
        "likelihood_ratio": round(lr, 4) if lr != float("inf") else None,
        "interpretation": _interpret_posterior(prior, posterior),
    }


def bayes_multi_step(
    prior: float,
    tests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sequential Bayesian updates through multiple tests.

    Parameters
    ----------
    prior : float
        Initial P(H).
    tests : list of dict
        Each dict has keys: sensitivity, false_positive_rate, observed_positive.

    Returns
    -------
    dict with steps list and final posterior.
    """
    current = prior
    steps: list[dict[str, Any]] = []
    for i, test in enumerate(tests):
        result = bayes_update(
            prior=current,
            sensitivity=float(test.get("sensitivity", 0.9)),
            false_positive_rate=float(test.get("false_positive_rate", 0.1)),
            observed_positive=bool(test.get("observed_positive", True)),
        )
        result["step"] = i + 1
        steps.append(result)
        current = result["posterior"]
    return {"initial_prior": prior, "final_posterior": current, "steps": steps}


# ── Distribution Calculations ─────────────────────────────────────────────


def distribution_pmf_pdf(
    distribution: str,
    params: dict[str, float],
    x: float | list[float],
) -> dict[str, Any]:
    """Compute PMF/PDF and CDF for a given distribution at point(s) x."""
    if not HAS_SCIPY:
        return {"error": "scipy not installed; cannot compute distribution values."}

    x_vals = x if isinstance(x, list) else [x]
    dist_name = distribution.lower()
    results: list[dict[str, Any]] = []

    try:
        dist = _get_scipy_dist(dist_name, params)
        for val in x_vals:
            entry: dict[str, Any] = {"x": val}
            if dist_name in ("binomial", "poisson"):
                entry["pmf"] = round(float(dist.pmf(val)), 8)
            else:
                entry["pdf"] = round(float(dist.pdf(val)), 8)
            entry["cdf"] = round(float(dist.cdf(val)), 8)
            entry["sf"] = round(float(dist.sf(val)), 8)  # survival = 1 - CDF
            results.append(entry)
    except Exception as exc:
        return {"error": str(exc)}

    mean_val = float(dist.mean())
    var_val = float(dist.var())
    return {
        "distribution": dist_name,
        "params": params,
        "mean": round(mean_val, 6),
        "variance": round(var_val, 6),
        "std": round(math.sqrt(var_val), 6),
        "results": results,
    }


def distribution_quantile(
    distribution: str,
    params: dict[str, float],
    probabilities: list[float],
) -> dict[str, Any]:
    """Compute quantiles (inverse CDF) for given probabilities."""
    if not HAS_SCIPY:
        return {"error": "scipy not installed."}
    try:
        dist = _get_scipy_dist(distribution.lower(), params)
        quantiles = [
            {"probability": p, "quantile": round(float(dist.ppf(p)), 6)}
            for p in probabilities
        ]
        return {"distribution": distribution, "params": params, "quantiles": quantiles}
    except Exception as exc:
        return {"error": str(exc)}


# ── Joint / Conditional Probability ───────────────────────────────────────


def independent_events(probabilities: list[float]) -> dict[str, Any]:
    """Compute joint probability of independent events and at-least-one."""
    probs = [_clamp(p) for p in probabilities]
    joint_all = math.prod(probs)
    joint_none = math.prod(1 - p for p in probs)
    at_least_one = 1 - joint_none
    return {
        "individual_probabilities": probs,
        "P_all_occur": round(joint_all, 8),
        "P_none_occur": round(joint_none, 8),
        "P_at_least_one": round(at_least_one, 8),
        "n_events": len(probs),
    }


def conditional_probability(
    p_a_and_b: float,
    p_b: float,
) -> dict[str, Any]:
    """P(A|B) = P(A∩B) / P(B)."""
    if p_b <= 0:
        return {"error": "P(B) must be > 0.", "P_A_given_B": None}
    result = _clamp(p_a_and_b) / _clamp(p_b, low=1e-12)
    return {
        "P_A_and_B": p_a_and_b,
        "P_B": p_b,
        "P_A_given_B": round(min(1.0, result), 8),
    }


# ── Combinatorial Probability ────────────────────────────────────────────


def combinations_count(n: int, r: int) -> dict[str, Any]:
    """C(n, r) = n! / (r! * (n-r)!)."""
    if r < 0 or r > n or n < 0:
        return {"error": f"Invalid n={n}, r={r}.", "C": 0}
    c = math.comb(n, r)
    return {"n": n, "r": r, "C(n,r)": c}


def permutations_count(n: int, r: int) -> dict[str, Any]:
    """P(n, r) = n! / (n-r)!."""
    if r < 0 or r > n or n < 0:
        return {"error": f"Invalid n={n}, r={r}.", "P": 0}
    p = math.perm(n, r)
    return {"n": n, "r": r, "P(n,r)": p}


# ── Distribution Fitting ─────────────────────────────────────────────────


def fit_distribution(data: list[float], candidates: list[str] | None = None) -> dict[str, Any]:
    """Fit multiple distributions to data and return best fit by AIC.

    Parameters
    ----------
    data : list of float
        Raw data samples.
    candidates : list of str, optional
        Distribution names to try. Defaults to common ones.

    Returns
    -------
    dict with best_fit, all_fits, and descriptive stats.
    """
    if not HAS_SCIPY:
        return {"error": "scipy not installed."}
    if len(data) < 3:
        return {"error": "Need at least 3 data points for fitting."}

    import numpy as np  # type: ignore[import-untyped]

    arr = np.array(data, dtype=float)
    if candidates is None:
        candidates = ["norm", "expon", "lognorm", "gamma", "uniform"]

    fits: list[dict[str, Any]] = []
    for name in candidates:
        try:
            dist = getattr(sp_stats, name)
            params = dist.fit(arr)
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = sp_stats.kstest(arr, name, args=params)
            # Log-likelihood for AIC
            ll = float(np.sum(dist.logpdf(arr, *params)))
            k = len(params)
            aic = 2 * k - 2 * ll
            fits.append({
                "distribution": name,
                "params": {f"param_{i}": round(float(v), 6) for i, v in enumerate(params)},
                "ks_statistic": round(float(ks_stat), 6),
                "ks_p_value": round(float(ks_p), 6),
                "aic": round(aic, 2),
            })
        except Exception:
            continue

    fits.sort(key=lambda f: f["aic"])
    return {
        "n_samples": len(data),
        "descriptive": {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)), 6) if len(data) > 1 else 0.0,
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "median": round(float(np.median(arr)), 6),
        },
        "best_fit": fits[0] if fits else None,
        "all_fits": fits,
        "warning": "Sample size < 30; estimates may be unreliable." if len(data) < 30 else None,
    }


# ── Helpers ───────────────────────────────────────────────────────────────


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _get_scipy_dist(name: str, params: dict[str, float]) -> Any:
    """Map distribution name + params to a frozen scipy.stats distribution."""
    if name in ("normal", "norm"):
        return sp_stats.norm(loc=params.get("mean", 0), scale=params.get("std", 1))
    if name in ("binomial", "binom"):
        return sp_stats.binom(n=int(params.get("n", 10)), p=params.get("p", 0.5))
    if name in ("poisson",):
        return sp_stats.poisson(mu=params.get("mu", params.get("lambda", 5)))
    if name in ("exponential", "expon"):
        return sp_stats.expon(scale=1 / params.get("rate", params.get("lambda", 1)))
    if name in ("uniform",):
        a = params.get("a", params.get("low", 0))
        b = params.get("b", params.get("high", 1))
        return sp_stats.uniform(loc=a, scale=b - a)
    if name in ("gamma",):
        return sp_stats.gamma(a=params.get("shape", params.get("alpha", 2)), scale=params.get("scale", 1))
    if name in ("lognorm", "lognormal"):
        return sp_stats.lognorm(s=params.get("sigma", params.get("s", 1)), scale=math.exp(params.get("mu", 0)))
    raise ValueError(f"Unsupported distribution: {name}")


def _interpret_posterior(prior: float, posterior: float) -> str:
    diff = posterior - prior
    if abs(diff) < 0.01:
        return "Bằng chứng hầu như không thay đổi niềm tin ban đầu."
    if diff > 0.15:
        return "Bằng chứng làm tăng đáng kể xác suất giả thuyết."
    if diff > 0:
        return "Bằng chứng hỗ trợ nhẹ cho giả thuyết."
    if diff < -0.15:
        return "Bằng chứng làm giảm đáng kể xác suất giả thuyết."
    return "Bằng chứng phản bác nhẹ giả thuyết."
