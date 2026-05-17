export type Scenario = {
  name: string;
  probability: number;
  utility: number;
  evidence?: string;
};

export type DecisionOption = {
  id: string;
  name: string;
  cost: number;
  reversibility: number;
  scenarios: Scenario[];
};

export type DecisionInput = {
  question: string;
  domain: string;
  objective: string;
  context: string;
  riskTolerance: number;
  timeHorizon: string;
  model: string;
  webRealtime: boolean;
  options: DecisionOption[];
};

export type OptionResult = {
  id: string;
  name: string;
  expected_utility: number;
  risk_adjusted_score: number;
  expected_regret: number;
  worst_case: number;
  best_case: number;
  confidence: number;
  normalized_probability_sum: number;
  scenarios: Array<Scenario & { contribution: number; regret: number }>;
};

export type AnalysisResult = {
  recommendation: string;
  summary: string;
  assumptions: string[];
  warnings: string[];
  option_results: OptionResult[];
  value_of_information: {
    score: number;
    most_valuable_unknowns: string[];
  };
  sensitivity: Array<{
    option: string;
    variable: string;
    impact: number;
  }>;
  ai_review: string;
  learning_note: string;
  web_research?: {
    enabled: boolean;
    queries?: string[];
    results: Array<{ title: string; url: string; snippet: string; source_score?: number; provider?: string }>;
    summary: string;
    market_data?: Array<Record<string, string | number>>;
    agent_trace?: Array<{ step: string; output: unknown }>;
  };
  agent_trace?: Array<{ step: string; output: unknown }>;
  market_data?: Array<Record<string, string | number>>;
  clarifying_questions?: string[];
  decision_gate?: "ready" | "needs_clarification";
  safety?: {
    risk_level: string;
    requires_clarification: boolean;
    missing_fields: string[];
    warnings: string[];
  };
  monte_carlo?: {
    iterations: number;
    distributions: Array<{
      option: string;
      mean: number;
      p10: number;
      p50: number;
      p90: number;
      win_rate: number;
    }>;
  };
  book_rag?: {
    results: Array<{
      id: string;
      page: number;
      section: string;
      score: number;
      matched_terms: string[];
      excerpt: string;
    }>;
    summary: string;
  };
};
