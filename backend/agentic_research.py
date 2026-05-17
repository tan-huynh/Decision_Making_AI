from __future__ import annotations

import math
import re
from typing import Any

import httpx

from web_research import search_web


FINANCE_WORDS = {
    "stock",
    "stocks",
    "equity",
    "invest",
    "investment",
    "portfolio",
    "shares",
    "chứng khoán",
    "cổ phiếu",
    "đầu tư",
}

MEDICATION_WORDS = {
    "drug",
    "medication",
    "medicine",
    "dose",
    "side effect",
    "contraindication",
    "interaction",
    "thuốc",
    "liều",
    "tác dụng phụ",
    "chống chỉ định",
    "tương tác",
    "phản tác dụng",
}

DOMAIN_RESEARCH_LENSES = {
    "career": ["salary outlook", "job market demand", "skill requirements", "long term career risk"],
    "income": ["ways to increase income", "market demand", "skills monetization", "time to revenue"],
    "education": ["learning outcomes", "program reputation", "cost opportunity cost", "career outcomes"],
    "finance": ["personal finance strategy", "risk return", "liquidity", "downside risk"],
    "business": ["market demand", "competitor analysis", "unit economics", "execution risk"],
    "life": ["expert guidance", "common risks", "opportunity cost", "long term wellbeing"],
    "wisdom": ["life experience advice", "common mistakes", "long term consequences", "values tradeoffs"],
    "relationship": ["conflict resolution evidence", "communication risk", "long term compatibility", "wellbeing"],
    "health": ["clinical guidance", "benefits harms", "contraindications interactions", "when to consult professional"],
    "legal": ["legal requirements", "regulatory risk", "professional guidance", "jurisdiction considerations"],
    "safety": ["safety guidance", "failure modes", "mitigation steps", "emergency thresholds"],
}

DOMAIN_AUTHORITIES = {
    "health": ["site:medlineplus.gov", "site:fda.gov", "site:nih.gov", "site:pubmed.ncbi.nlm.nih.gov"],
    "legal": ["site:.gov legal", "official regulation", "government guidance"],
    "finance": ["site:sec.gov", "site:investor.gov", "site:finance.yahoo.com"],
    "career": ["site:bls.gov", "salary survey", "job market report"],
    "education": ["site:bls.gov", "course outcomes", "labor market report"],
    "income": ["market demand report", "freelance rate survey", "small business guidance"],
}


def detect_tickers(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", text)
    blacklist = {"I", "AI", "API", "CEO", "USD", "ETF", "GDP", "EPS", "IPO"}
    unique = []
    for item in candidates:
        if item not in blacklist and item not in unique:
            unique.append(item)
    return unique[:8]


def is_finance_domain(payload: dict[str, Any]) -> bool:
    text = " ".join(
        str(payload.get(key, "")).lower()
        for key in ["question", "domain", "objective", "context"]
    )
    return payload.get("domain") == "finance" or any(word in text for word in FINANCE_WORDS)


def is_medication_question(payload: dict[str, Any]) -> bool:
    text = " ".join(
        str(payload.get(key, "")).lower()
        for key in ["question", "domain", "objective", "context"]
    )
    return payload.get("domain") == "health" and any(word in text for word in MEDICATION_WORDS)


def build_research_queries(payload: dict[str, Any]) -> list[str]:
    question = str(payload.get("question", "")).strip()
    domain = str(payload.get("domain", "life")).strip()
    context = str(payload.get("context", "")).strip()
    base = f"{question} {context}".strip()
    lenses = DOMAIN_RESEARCH_LENSES.get(domain, DOMAIN_RESEARCH_LENSES["life"])
    queries = [f"{base} {lens} latest evidence 2026" for lens in lenses[:4]]
    queries.extend(
        [
            f"{base} risks downside uncertainty",
            f"{base} expert analysis recent data",
        ]
    )
    if is_finance_domain(payload):
        tickers = detect_tickers(base)
        if tickers:
            queries.extend([f"{ticker} stock latest earnings analyst outlook risk" for ticker in tickers])
            queries.extend([f"{ticker} stock news today" for ticker in tickers])
        else:
            queries.append(f"{base} market data valuation earnings momentum")
    for authority in DOMAIN_AUTHORITIES.get(domain, [])[:3]:
        queries.append(f"{base} {authority}")
    if is_medication_question(payload):
        queries.extend(
            [
                f"{base} drug interactions contraindications official label",
                f"{base} common serious side effects warnings",
                f"{base} benefits harms clinical guideline",
                f"{base} avoid with alcohol pregnancy kidney liver disease",
            ]
        )
    return list(dict.fromkeys(query[:450] for query in queries if query.strip()))[:8]


def score_source(url: str, domain: str) -> float:
    lowered = url.lower()
    score = 0.45
    if any(host in lowered for host in [".gov", "nih.gov", "fda.gov", "medlineplus.gov", "sec.gov", "bls.gov"]):
        score += 0.35
    if any(host in lowered for host in ["pubmed", "who.int", "investor.gov"]):
        score += 0.25
    if domain == "finance" and any(host in lowered for host in ["sec.gov", "finance.yahoo.com", "investor.gov"]):
        score += 0.15
    if domain == "health" and any(host in lowered for host in ["drugs.com", "mayoclinic", "nhs.uk"]):
        score += 0.12
    return min(1.0, round(score, 2))


async def fetch_market_data(tickers: list[str]) -> list[dict[str, Any]]:
    output = []
    async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
        for ticker in tickers:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1mo&interval=1d"
                response = await client.get(url, headers={"User-Agent": "AI-Powered Decision Making/0.1"})
                response.raise_for_status()
                result = response.json()["chart"]["result"][0]
                meta = result["meta"]
                closes = [value for value in result["indicators"]["quote"][0].get("close", []) if value is not None]
                if not closes:
                    continue
                last = float(closes[-1])
                previous = float(closes[-2]) if len(closes) > 1 else last
                first = float(closes[0])
                returns = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes)) if closes[i - 1]]
                volatility = math.sqrt(sum((item - (sum(returns) / len(returns))) ** 2 for item in returns) / len(returns)) if returns else 0
                output.append(
                    {
                        "ticker": ticker,
                        "currency": meta.get("currency"),
                        "exchange": meta.get("exchangeName"),
                        "price": last,
                        "day_change_pct": ((last / previous) - 1) * 100 if previous else 0,
                        "month_change_pct": ((last / first) - 1) * 100 if first else 0,
                        "daily_volatility_pct": volatility * 100,
                    }
                )
            except Exception as exc:
                output.append({"ticker": ticker, "error": str(exc)})
    return output


async def run_agentic_research(payload: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(payload.get("webRealtime", True))
    queries = build_research_queries(payload) if enabled else []
    trace = [
        {
            "step": "frame_decision",
            "output": "Xác định mục tiêu, lựa chọn, scenario, utility, rủi ro và thông tin còn thiếu.",
        },
        {
            "step": "plan_research",
            "output": queries,
        },
    ]

    domain = str(payload.get("domain", "life")).lower()
    web_results = []
    web_summary_parts = []
    for query in queries:
        result = await search_web(query, enabled=True)
        web_results.extend(
            {
                **item,
                "source_score": score_source(item.get("url", ""), domain),
                "provider": "duckduckgo",
            }
            for item in result.get("results", [])
        )
        if result.get("summary"):
            web_summary_parts.append(f"Query: {query}\n{result['summary']}")

    tickers = detect_tickers(" ".join(str(payload.get(key, "")) for key in ["question", "context", "objective"]))
    market_data = await fetch_market_data(tickers) if enabled and is_finance_domain(payload) and tickers else []
    if market_data:
        trace.append({"step": "fetch_market_data", "output": market_data})
    trace.append(
        {
            "step": "synthesize_evidence",
            "output": f"{len(web_results)} web snippets, {len(market_data)} market instruments.",
        }
    )
    return {
        "enabled": enabled,
        "queries": queries,
        "results": sorted(web_results, key=lambda item: item.get("source_score", 0), reverse=True)[:12],
        "summary": "\n\n".join(web_summary_parts)[:7000],
        "market_data": market_data,
        "agent_trace": trace,
    }
