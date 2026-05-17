import unittest

from agentic_research import build_research_queries, detect_tickers, is_finance_domain


class AgenticResearchTests(unittest.TestCase):
    def test_detects_stock_tickers(self):
        self.assertEqual(detect_tickers("Should I buy AAPL or MSFT now?"), ["AAPL", "MSFT"])

    def test_builds_finance_research_plan(self):
        payload = {
            "question": "Should I invest in AAPL?",
            "domain": "finance",
            "objective": "Risk-adjusted return",
            "context": "Compare with holding cash",
        }
        queries = build_research_queries(payload)
        self.assertTrue(any("AAPL stock" in query for query in queries))
        self.assertTrue(is_finance_domain(payload))


if __name__ == "__main__":
    unittest.main()
