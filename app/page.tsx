"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Download,
  ImageDown,
  Maximize2,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RefreshCw,
  Save,
  Sparkles,
  Trash2,
} from "lucide-react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import mermaid from "mermaid";
import DecisionChart from "@/components/DecisionChart";
import { defaultDecision } from "@/lib/defaultDecision";
import type { AnalysisResult, DecisionInput, DecisionOption } from "@/lib/types";

const emptyOption = (index: number): DecisionOption => ({
  id: `option-${Date.now()}-${index}`,
  name: `Lựa chọn ${index + 1}`,
  cost: 10,
  reversibility: 0.5,
  scenarios: [
    { name: "Kết quả tốt", probability: 0.45, utility: 75, evidence: "" },
    { name: "Kết quả trung bình", probability: 0.35, utility: 45, evidence: "" },
    { name: "Kết quả xấu", probability: 0.2, utility: 15, evidence: "" }
  ]
});

type EngineeringResult = {
  status?: string;
  problem?: {
    context?: {
      title?: string;
      domain?: string;
      description?: string;
    };
    problem_type?: string;
    resource_allocation?: {
      resource_name: string;
      total_resource: number;
      stage_returns: number[][];
    };
    probability_tree?: {
      success_probability: number;
      trials: number;
      success_label: string;
      failure_label: string;
    };
    graph?: {
      supplies?: Record<string, number>;
      demands?: Record<string, number>;
      costs?: Record<string, Record<string, number>>;
    };
    alternatives?: Array<{ name: string; description?: string }>;
    states?: Array<{ name: string; probability: number }>;
    payoff_matrix?: Array<{ alternative: string; state: string; payoff: number; cost?: number }>;
    bayes?: Record<string, number | boolean>;
    independent_probabilities?: number[];
    assumptions?: string[];
  };
  solved?: {
    problem_type?: string;
    model?: {
      formulation?: string;
      missing_data?: string[];
      assumptions?: string[];
    };
    validation?: {
      is_valid?: boolean;
      errors?: string[];
      warnings?: string[];
      info?: string[];
    };
    result?: {
      status?: string;
      solver?: string;
      objective_value?: number;
      path?: string[];
      allocations?: Array<{ from: string; to: string; amount: number; unit_cost: number }>;
      allocation?: Array<{ stage: number; tons?: number; resource?: number; profit: number }>;
      dp_tables?: Array<{ stage: number; values: Record<string, number>; choice: Record<string, number> }>;
      success_probability?: number;
      failure_probability?: number;
      trials?: number;
      tree_levels?: Array<{ trial: number; branches: Array<{ label: string; probability: number }> }>;
      outcomes?: Array<{ events: string[]; label: string; probability: number; success_count: number }>;
      queries?: {
        first_success_second_failure?: number | null;
        at_least_one_success?: number;
      };
      optimality_certificate?: {
        is_optimal?: boolean;
        u?: Record<string, number>;
        v?: Record<string, number>;
        reduced_costs?: Record<string, number>;
      };
      recommendation?: string;
      markdown_report?: string;
    };
    recommendation_explanation?: string;
  };
  message?: string;
  questions?: string[];
  autonomy?: {
    mode: string;
    route: string;
    selected_solver: string;
    case_type: string;
    confidence: number;
    reason: string;
    agent_steps?: string[];
    similar_cases?: Array<{ route?: string; case_type?: string; selected_solver?: string; created_at?: string }>;
    learning_profile?: {
      events?: number;
      routes?: Record<string, number>;
      case_types?: Record<string, number>;
      solvers?: Record<string, number>;
    };
  };
  recognition_gate?: {
    recognized_problem_type?: string;
    recognized_subtype?: string;
    confidence?: number;
    evidence?: string[];
    required_slots?: string[];
    filled_slots?: string[];
    missing_slots?: string[];
    decision_to_solve?: string;
  };
};

function formatMetric(value: number | string | undefined, digits = 1) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  }
  return value || "n/a";
}

function buildAllocationMatrix(data: EngineeringResult) {
  const supplies = data.problem?.graph?.supplies || {};
  const demands = data.problem?.graph?.demands || {};
  const allocations = data.solved?.result?.allocations || [];
  const sources = Object.keys(supplies);
  const destinations = Object.keys(demands);
  if (!sources.length || !destinations.length || !allocations.length) return null;
  const lookup = new Map(allocations.map((item) => [`${item.from}::${item.to}`, item.amount]));
  return { sources, destinations, supplies, demands, lookup };
}

function engineeringMarkdown(data: EngineeringResult) {
  let md = recognitionGateMarkdown(data) || "";
  md += "## Mô hình EDSS đã hiểu\n\n";
  md += `**Loại bài toán**: ${data.solved?.problem_type || data.problem?.problem_type || "unknown"}\n\n`;
  if (data.solved?.model?.formulation) {
    md += "### Công thức / Formulation\n\n";
    md += "```text\n" + data.solved.model.formulation + "\n```\n\n";
  }
  if (data.problem) {
    const compactProblem = {
      probability_tree: data.problem.probability_tree,
      bayes: data.problem.bayes,
      independent_probabilities: data.problem.independent_probabilities,
      resource_allocation: data.problem.resource_allocation,
      graph: data.problem.graph,
      alternatives: data.problem.alternatives,
      states: data.problem.states,
      payoff_matrix: data.problem.payoff_matrix,
      assumptions: data.problem.assumptions
    };
    md += "### Dữ liệu mô hình\n\n";
    md += "```json\n" + JSON.stringify(compactProblem, null, 2) + "\n```\n\n";
  }
  if (data.solved?.validation) {
    md += "### Validation trước khi giải\n\n";
    md += `**Hợp lệ**: ${data.solved.validation.is_valid ? "Có" : "Không"}\n\n`;
    if (data.solved.validation.errors?.length) md += `**Errors**:\n${data.solved.validation.errors.map((item) => `- ${item}`).join("\n")}\n\n`;
    if (data.solved.validation.warnings?.length) md += `**Warnings**:\n${data.solved.validation.warnings.map((item) => `- ${item}`).join("\n")}\n\n`;
    if (data.solved.validation.info?.length) md += `**Info**:\n${data.solved.validation.info.map((item) => `- ${item}`).join("\n")}\n\n`;
  }
  md += "## Kết quả solver\n\n";
  if (data.solved?.result?.markdown_report) {
    md += data.solved.result.markdown_report;
  } else if (data.solved?.result) {
    md += `**Status**: ${data.solved.result.status}\n\n`;
    if (data.solved.problem_type) md += `**Loại bài toán**: ${data.solved.problem_type}\n\n`;
    md += jsonToMarkdown(data.solved.result as Record<string, unknown>);
    if (data.solved.recommendation_explanation) md += `\n\n## Khuyến nghị\n\n${data.solved.recommendation_explanation}\n`;
  }
  md += "\n\n## Ghi chú chất lượng quyết định\n\nKết quả số được tính bằng solver deterministic, không dùng LLM để tính toán. Outcome tốt sau thực tế có thể do may mắn; quyết định tốt là quyết định dựa trên mô hình đúng, dữ liệu đủ và giả định đã kiểm tra độ nhạy.\n";
  return md;
}

function copyEngineeringReport(data: EngineeringResult) {
  if (typeof navigator === "undefined") return;
  navigator.clipboard?.writeText(engineeringMarkdown(data)).catch(() => undefined);
}

type EngineeringHistoryItem = {
  id: string;
  title: string;
  text: string;
  result: EngineeringResult;
  createdAt: string;
  problemType?: string;
};

const ENGINEERING_HISTORY_KEY = "edss_solver_history_v1";
const MAX_ENGINEERING_HISTORY = 40;

mermaid.initialize({ startOnLoad: false, theme: "default" });

function jsonToMarkdown(data: Record<string, unknown>, depth = 0): string {
  const lines: string[] = [];
  for (const [key, value] of Object.entries(data)) {
    const label = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    if (value === null || value === undefined) continue;

    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      if (typeof value === "number" && !Number.isInteger(value)) {
        lines.push(`**${label}**: ${Number(value).toFixed(4)}`);
      } else {
        lines.push(`**${label}**: ${value}`);
      }
    } else if (Array.isArray(value)) {
      if (value.length === 0) continue;
      if (typeof value[0] === "object" && value[0] !== null) {
        const keys = Object.keys(value[0] as Record<string, unknown>);
        lines.push("");
        lines.push(`**${label}:**`);
        lines.push("");
        lines.push("| " + keys.map((k) => k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())).join(" | ") + " |");
        lines.push("| " + keys.map(() => "---").join(" | ") + " |");
        for (const row of value.slice(0, 30)) {
          const r = row as Record<string, unknown>;
          lines.push("| " + keys.map((k) => {
            const v = r[k];
            if (typeof v === "number") return Number.isInteger(v) ? String(v) : Number(v).toFixed(4);
            return String(v ?? "");
          }).join(" | ") + " |");
        }
        lines.push("");
      } else if (typeof value[0] === "string") {
        lines.push(`**${label}**:`);
        for (const item of value) lines.push(`- ${item}`);
      } else {
        lines.push(`**${label}**: ${value.join(", ")}`);
      }
    } else if (typeof value === "object") {
      if (depth < 2) {
        lines.push("");
        lines.push(`### ${label}`);
        lines.push("");
        lines.push(jsonToMarkdown(value as Record<string, unknown>, depth + 1));
      } else {
        lines.push(`**${label}**: \`${JSON.stringify(value)}\``);
      }
    }
  }
  return lines.join("\n");
}

function MermaidView({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const id = `mermaid-${Math.random().toString(36).slice(2)}`;
    mermaid
      .render(id, chart)
      .then(({ svg }) => {
        if (!cancelled && ref.current) ref.current.innerHTML = svg;
      })
      .catch((error) => {
        if (ref.current) {
          ref.current.textContent = `Mermaid render error: ${error instanceof Error ? error.message : String(error)}`;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [chart]);

  return <div ref={ref} className="mermaid-wrapper" />;
}

function normalizeMermaidChart(value: string): string {
  return value
    .replace(/^\s*```mermaid\s*/i, "")
    .replace(/\s*```\s*$/i, "")
    .trim();
}

function isMermaidChart(value: string): boolean {
  return /^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|pie|mindmap|timeline)\b/i.test(
    normalizeMermaidChart(value)
  );
}

function markdownUrlTransform(url: string): string {
  if (/^data:image\/(?:svg\+xml|png|jpeg|jpg|webp);/i.test(url)) return url;
  return defaultUrlTransform(url);
}

function recognitionGateMarkdown(data: EngineeringResult): string {
  const gate = data.recognition_gate;
  if (!gate) return "";
  const evidence = gate.evidence || [];
  const required = gate.required_slots || [];
  const filled = gate.filled_slots || [];
  const missing = gate.missing_slots || [];
  return [
    "## 1. Nhận dạng dạng toán",
    "",
    `- **Dạng toán chính:** ${gate.recognized_problem_type || "unknown"}`,
    `- **Dạng toán phụ:** ${gate.recognized_subtype || "unknown"}`,
    `- **Mức tin cậy:** ${gate.confidence ?? "n/a"}`,
    `- **Quyết định gate:** ${gate.decision_to_solve || "unknown"}`,
    "",
    "**Dấu hiệu nhận dạng:**",
    ...(evidence.length ? evidence.map((item) => `- ${item}`) : ["- Chưa đủ tín hiệu rõ ràng."]),
    "",
    "**Dữ liệu cần có:**",
    ...(required.length ? required.map((item) => `- ${item}`) : ["- Chưa xác định."]),
    "",
    "**Dữ liệu đã có:**",
    ...(filled.length ? filled.map((item) => `- ${item}`) : ["- Chưa đủ slot bắt buộc."]),
    "",
    "**Dữ liệu còn thiếu nếu có:**",
    ...(missing.length ? missing.map((item) => `- ${item}`) : ["- Không có slot bắt buộc bị thiếu."]),
    "",
    gate.decision_to_solve === "solve"
      ? "**Kết luận:** Đủ điều kiện để xây mô hình, chọn solver và giải."
      : "**Kết luận:** Chưa đủ điều kiện để đưa lời giải cuối cùng; cần làm rõ trước.",
    "",
  ].join("\n");
}

function EngineeringSolutionView({ data }: { data: EngineeringResult }) {
  if (data.status !== "solved" || !data.solved?.result) {
    const gateMd = recognitionGateMarkdown(data);
    return (
      <div className="analysis" style={{ marginTop: 16 }}>
        <h2>Engineering Solver Result</h2>
        {gateMd ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{gateMd}</ReactMarkdown> : null}
        <p>{data.message || "Không đủ dữ liệu để khuyến nghị chắc chắn."}</p>
        {data.questions?.length ? data.questions.map((question) => <p key={question}>- {question}</p>) : null}
      </div>
    );
  }

  const solved = data.solved.result;
  const matrix = buildAllocationMatrix(data);
  const certificate = solved.optimality_certificate;
  const reducedCosts = certificate?.reduced_costs ? Object.entries(certificate.reduced_costs).filter(([key]) => !key.includes("_dummy")).slice(0, 12) : [];
  const md = engineeringMarkdown(data);

  return (
    <div className="solver-report">
      <div className="solver-summary">
        <div>
          <span className="eyebrow">EDSS result</span>
          <h2>{solved.recommendation || data.solved.recommendation_explanation || "Solver completed"}</h2>
          <p>
            Hệ thống nhận dạng <strong>{data.solved.problem_type || data.problem?.problem_type || "unknown"}</strong>,
            chọn <strong>{solved.solver || data.autonomy?.selected_solver || "solver"}</strong>, và trả nghiệm có thể audit.
          </p>
        </div>
        <div className="solver-kpis">
          <div><span>Status</span><strong>{solved.status || data.status}</strong></div>
          <div><span>Objective</span><strong>{formatMetric(solved.objective_value)}</strong></div>
          <div><span>Optimal proof</span><strong>{certificate?.is_optimal ? "MODI optimal" : solved.status || "n/a"}</strong></div>
          <div><span>Autonomous</span><strong>{data.autonomy ? `${data.autonomy.case_type} ${Math.round(data.autonomy.confidence * 100)}%` : "n/a"}</strong></div>
        </div>
      </div>

      {matrix ? (
        <section className="solver-section">
          <div className="section-head">
            <h3>Ma trận phân bổ tối ưu</h3>
            <button className="button secondary" onClick={() => copyEngineeringReport(data)}>Copy lời giải</button>
          </div>
          <div className="table-wrap">
            <table className="solver-table allocation-matrix">
              <thead>
                <tr>
                  <th>Nguồn / Đích</th>
                  {matrix.destinations.map((destination) => <th key={destination}>{destination}</th>)}
                  <th>Cung</th>
                </tr>
              </thead>
              <tbody>
                {matrix.sources.map((source) => (
                  <tr key={source}>
                    <th>{source}</th>
                    {matrix.destinations.map((destination) => {
                      const amount = matrix.lookup.get(`${source}::${destination}`) || 0;
                      return <td key={destination} className={amount > 0 ? "allocated" : ""}>{formatMetric(amount)}</td>;
                    })}
                    <td>{formatMetric(matrix.supplies[source])}</td>
                  </tr>
                ))}
                <tr>
                  <th>Nhu cầu</th>
                  {matrix.destinations.map((destination) => <td key={destination}>{formatMetric(matrix.demands[destination])}</td>)}
                  <td>{formatMetric(Object.values(matrix.supplies).reduce((sum, value) => sum + Number(value), 0))}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {solved.allocations?.length ? (
        <section className="solver-section">
          <h3>Routing được sử dụng</h3>
          <div className="table-wrap">
            <table className="solver-table">
              <thead>
                <tr><th>Từ</th><th>Đến</th><th>Lượng</th><th>Đơn giá</th><th>Chi phí</th></tr>
              </thead>
              <tbody>
                {solved.allocations.map((item) => (
                  <tr key={`${item.from}-${item.to}`}>
                    <td>{item.from}</td>
                    <td>{item.to}</td>
                    <td>{formatMetric(item.amount)}</td>
                    <td>{formatMetric(item.unit_cost)}</td>
                    <td>{formatMetric(item.amount * item.unit_cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="solver-section">
        <h3>Autonomous agent</h3>
        <div className="agent-flow">
          {(data.autonomy?.agent_steps || ["recognize_case", "validate_inputs", "solve", "audit", "learn"]).map((step, index) => (
            <div key={`${step}-${index}`}><span>{index + 1}</span>{step.replace(/_/g, " ")}</div>
          ))}
        </div>
      </section>

      <details className="solver-details">
        <summary>Chi tiết mô hình, validation và proof</summary>
        {reducedCosts.length ? (
          <div className="table-wrap">
            <table className="solver-table compact">
              <thead><tr><th>Tuyến chưa dùng</th><th>Reduced cost</th></tr></thead>
              <tbody>
                {reducedCosts.map(([route, value]) => (
                  <tr key={route}><td>{route}</td><td>{formatMetric(Number(value))}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        <div className="md-view compact-md">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            urlTransform={markdownUrlTransform}
            components={{
              pre({ children, ...props }) {
                const raw = String((children as any)?.props?.children ?? "");
                const className = String((children as any)?.props?.className ?? "");
                if (className.includes("language-mermaid") || isMermaidChart(raw)) return <MermaidView chart={normalizeMermaidChart(raw)} />;
                return <pre {...props}>{children}</pre>;
              },
              code({ className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || "");
                const raw = String(children).replace(/\n$/, "");
                if (match?.[1] === "mermaid" || isMermaidChart(raw)) return <MermaidView chart={normalizeMermaidChart(raw)} />;
                return <code className={className} {...props}>{children}</code>;
              },
            }}
          >
            {md}
          </ReactMarkdown>
        </div>
      </details>
    </div>
  );
}

export default function Home() {
  const [decision, setDecision] = useState<DecisionInput>(defaultDecision);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<{ ok: boolean; models: string[]; error?: string }>({ ok: false, models: [] });
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [learning, setLearning] = useState(false);
  const [chartMode, setChartMode] = useState<"map" | "logic" | "score" | "regret" | "tree">("map");
  const [error, setError] = useState("");
  const [engineeringText, setEngineeringText] = useState("");
  const [engineeringResult, setEngineeringResult] = useState<EngineeringResult | null>(null);
  const [engineeringLoading, setEngineeringLoading] = useState(false);
  const [engineeringHistory, setEngineeringHistory] = useState<EngineeringHistoryItem[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [mapRefresh, setMapRefresh] = useState(0);
  const [mapFullscreen, setMapFullscreen] = useState(false);
  const [visibleMapKinds, setVisibleMapKinds] = useState(["scenario", "factor", "evidence"]);
  const [selectedMapNode, setSelectedMapNode] = useState<{ label: string; kind: string; detail: string } | null>(null);
  const [collapsedRegions, setCollapsedRegions] = useState({
    sidebar: false,
    workbench: false,
    insight: false,
  });
  const mapShellRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetch("/api/ollama-status")
      .then((r) => r.json())
      .then((data) => {
        setStatus(data);
        if (data.models?.length && !data.models.includes(decision.model)) {
          setDecision((current) => ({ ...current, model: data.models[0] }));
        }
      })
      .catch(() => setStatus({ ok: false, models: [] }));
  }, [decision.model]);

  useEffect(() => {
    const syncFullscreen = () => setMapFullscreen(document.fullscreenElement === mapShellRef.current);
    document.addEventListener("fullscreenchange", syncFullscreen);
    return () => document.removeEventListener("fullscreenchange", syncFullscreen);
  }, []);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(ENGINEERING_HISTORY_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as EngineeringHistoryItem[];
      if (Array.isArray(parsed)) setEngineeringHistory(parsed);
    } catch {
      setEngineeringHistory([]);
    }
  }, []);

  const winner = useMemo(() => result?.option_results[0], [result]);
  const runnerUp = useMemo(() => result?.option_results[1], [result]);
  const resultByOptionId = useMemo(
    () => new Map(result?.option_results.map((item) => [item.id, item]) || []),
    [result]
  );
  const scoreGap = winner && runnerUp ? winner.risk_adjusted_score - runnerUp.risk_adjusted_score : null;
  const topSensitivity = result?.sensitivity?.[0];
  const modelOptions = useMemo(
    () => Array.from(new Set(status.models.length ? status.models : [decision.model, "llama3.1", "qwen2.5", "mistral"])),
    [decision.model, status.models]
  );

  function updateOption(id: string, patch: Partial<DecisionOption>) {
    setDecision((current) => ({
      ...current,
      options: current.options.map((option) => (option.id === id ? { ...option, ...patch } : option))
    }));
  }

  async function analyze(nextDecision: DecisionInput = decision) {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(nextDecision)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  async function learn(outcome: "success" | "mixed" | "failure") {
    if (!result) return;
    setLearning(true);
    try {
      await fetch("/api/learn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, result, outcome, chosen_option: result.recommendation })
      });
    } finally {
      setLearning(false);
    }
  }

  async function generateModel() {
    setGenerating(true);
    setError("");
    try {
      const response = await fetch("/api/generate-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: decision.question,
          domain: decision.domain,
          objective: decision.objective,
          context: decision.context,
          riskTolerance: decision.riskTolerance,
          timeHorizon: decision.timeHorizon,
          model: decision.model,
          webRealtime: decision.webRealtime
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Generate model failed");
      setDecision(data);
      setResult(null);
      setMapRefresh((value) => value + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generate model failed");
    } finally {
      setGenerating(false);
    }
  }

  async function solveEngineeringText() {
    setEngineeringLoading(true);
    setError("");
    try {
      const response = await fetch("/api/edss-solve-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: engineeringText })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "EDSS solve failed");
      setEngineeringResult(data);
      saveEngineeringHistory(engineeringText, data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "EDSS solve failed");
    } finally {
      setEngineeringLoading(false);
    }
  }

  function persistEngineeringHistory(items: EngineeringHistoryItem[]) {
    setEngineeringHistory(items);
    try {
      window.localStorage.setItem(ENGINEERING_HISTORY_KEY, JSON.stringify(items));
    } catch {
      setError("Không thể lưu history vào localStorage.");
    }
  }

  function titleFromEngineeringText(text: string, result: EngineeringResult) {
    const firstHeading = text.split("\n").map((line) => line.replace(/^#+\s*/, "").trim()).find(Boolean);
    const fallback = result.problem?.context?.title || result.solved?.problem_type || "EDSS case";
    return (firstHeading || fallback).slice(0, 90);
  }

  function saveEngineeringHistory(text: string, data: EngineeringResult) {
    const trimmed = text.trim();
    if (!trimmed) return;
    const item: EngineeringHistoryItem = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      title: titleFromEngineeringText(trimmed, data),
      text: trimmed,
      result: data,
      createdAt: new Date().toISOString(),
      problemType: data.solved?.problem_type || data.problem?.problem_type
    };
    const withoutDuplicate = engineeringHistory.filter((entry) => entry.text.trim() !== trimmed);
    persistEngineeringHistory([item, ...withoutDuplicate].slice(0, MAX_ENGINEERING_HISTORY));
  }

  function loadEngineeringHistory(item: EngineeringHistoryItem) {
    setEngineeringText(item.text);
    setEngineeringResult(item.result);
    setError("");
  }

  function deleteEngineeringHistory(id: string) {
    persistEngineeringHistory(engineeringHistory.filter((item) => item.id !== id));
  }

  function clearEngineeringHistory() {
    persistEngineeringHistory([]);
  }

  function buildClarifiedDecision() {
    if (!result?.clarifying_questions?.length) return decision;
    const addition = result.clarifying_questions
      .map((question) => {
        const answer = answers[question]?.trim();
        return answer ? `Q: ${question}\nA: ${answer}` : "";
      })
      .filter(Boolean)
      .join("\n");
    if (!addition) return decision;
    return {
      ...decision,
      context: `${decision.context}\n\nThông tin bổ sung từ người dùng:\n${addition}`
    };
  }

  function applyClarifications() {
    setDecision(buildClarifiedDecision());
  }

  async function toggleMapFullscreen() {
    const element = mapShellRef.current;
    if (!element) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen().catch(() => undefined);
      setMapFullscreen(false);
      return;
    }
    await element.requestFullscreen?.().catch(() => {
      setMapFullscreen(true);
    });
    setMapFullscreen(true);
  }

  function getRenderedMapSvg() {
    const svg = mapShellRef.current?.querySelector("svg.chart") as SVGSVGElement | null;
    if (!svg) return null;
    if (svg.querySelectorAll("circle,line,path,text").length < 2) return null;
    return svg;
  }

  function downloadMapSvg() {
    const svg = getRenderedMapSvg();
    if (!svg) return;
    const width = Math.max(1, Math.round(svg.clientWidth || svg.getBoundingClientRect().width || 1200));
    const height = Math.max(1, Math.round(svg.clientHeight || svg.getBoundingClientRect().height || 720));
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));
    clone.setAttribute("viewBox", clone.getAttribute("viewBox") || `0 0 ${width} ${height}`);
    clone.setAttribute("version", "1.1");
    clone.querySelectorAll("text").forEach((node) => {
      node.setAttribute("font-family", "Arial, Helvetica, sans-serif");
    });
    clone.insertAdjacentHTML("afterbegin", `<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>`);
    const source = `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `decision-map-${new Date().toISOString().slice(0, 10)}.svg`;
    document.body.appendChild(link);
    link.click();
    window.setTimeout(() => {
      link.remove();
      URL.revokeObjectURL(url);
    }, 0);
  }

  function downloadMapPng() {
    const svg = getRenderedMapSvg();
    if (!svg) return;
    const width = Math.max(1, Math.round(svg.clientWidth || svg.getBoundingClientRect().width || 1200));
    const height = Math.max(1, Math.round(svg.clientHeight || svg.getBoundingClientRect().height || 720));
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));
    clone.insertAdjacentHTML("afterbegin", `<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>`);
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml;charset=utf-8" }));
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, width, height);
      context.drawImage(image, 0, 0);
      URL.revokeObjectURL(url);
      const link = document.createElement("a");
      link.href = canvas.toDataURL("image/png");
      link.download = `decision-map-${new Date().toISOString().slice(0, 10)}.png`;
      link.click();
    };
    image.src = url;
  }

  function toggleMapKind(kind: string) {
    setVisibleMapKinds((current) => (
      current.includes(kind) ? current.filter((item) => item !== kind) : [...current, kind]
    ));
  }

  function toggleWorkbench() {
    setCollapsedRegions((current) => ({
      ...current,
      workbench: !current.workbench,
      insight: current.insight && !current.workbench ? false : current.insight,
    }));
  }

  function toggleInsight() {
    setCollapsedRegions((current) => ({
      ...current,
      insight: !current.insight,
      workbench: current.workbench && !current.insight ? false : current.workbench,
    }));
  }

  return (
    <main className={`shell ${collapsedRegions.sidebar ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <h1>AI-Powered Decision Making</h1>
          <span className="status" title={status.error || "Ollama status"}>
            <span className={`dot ${status.ok ? "ok" : ""}`} />
            Ollama
          </span>
          <button
            className="icon-button layout-toggle"
            title={collapsedRegions.sidebar ? "Mở sidebar" : "Thu sidebar"}
            onClick={() => setCollapsedRegions((current) => ({ ...current, sidebar: !current.sidebar }))}
          >
            {collapsedRegions.sidebar ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>

        <div className="sidebar-content">
          <a href="/edss" className="button" style={{ margin: "0 16px 8px", textAlign: "center", textDecoration: "none", background: "linear-gradient(135deg, #3b82f6, #6366f1)", color: "white", borderColor: "transparent" }}>
            🧠 EDSS Dashboard — Engineering Solver
          </a>

          <section>
            <h2>Bài toán</h2>
            <div className="field">
              <label>Câu hỏi quyết định</label>
              <textarea value={decision.question} onChange={(event) => setDecision({ ...decision, question: event.target.value })} />
            </div>
            <div className="field">
              <label>Lĩnh vực</label>
              <select value={decision.domain} onChange={(event) => setDecision({ ...decision, domain: event.target.value })}>
                <option value="life">Đời sống cá nhân</option>
                <option value="career">Sự nghiệp</option>
                <option value="income">Tăng thu nhập</option>
                <option value="finance">Tài chính</option>
                <option value="health">Sức khỏe</option>
                <option value="relationship">Quan hệ</option>
                <option value="business">Kinh doanh</option>
                <option value="education">Học tập</option>
                <option value="wisdom">Kinh nghiệm sống</option>
                <option value="legal">Pháp lý</option>
                <option value="safety">An toàn</option>
              </select>
            </div>
            <div className="field">
              <label>Mục tiêu tối ưu</label>
              <textarea value={decision.objective} onChange={(event) => setDecision({ ...decision, objective: event.target.value })} />
            </div>
          <div className="field">
            <label>Bối cảnh và ràng buộc</label>
            <textarea value={decision.context} onChange={(event) => setDecision({ ...decision, context: event.target.value })} />
          </div>
          <div className="row">
            <div className="field">
              <label>Khẩu vị rủi ro</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={decision.riskTolerance}
                onChange={(event) => setDecision({ ...decision, riskTolerance: Number(event.target.value) })}
              />
              <span className="small">{decision.riskTolerance.toFixed(2)}</span>
            </div>
            <div className="field">
              <label>Model Ollama</label>
              <select value={decision.model} onChange={(event) => setDecision({ ...decision, model: event.target.value })}>
                {modelOptions.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="row">
            <div className="field">
              <label>Thời hạn quyết định</label>
              <input value={decision.timeHorizon} onChange={(event) => setDecision({ ...decision, timeHorizon: event.target.value })} />
            </div>
            <label className="row small" style={{ alignSelf: "center" }}>
              <input
                style={{ width: 18 }}
                type="checkbox"
                checked={decision.webRealtime}
                onChange={(event) => setDecision({ ...decision, webRealtime: event.target.checked })}
              />
              Web real-time
            </label>
          </div>
          <div className="row">
            <button className="button secondary" onClick={generateModel} disabled={generating || loading}>
              <RefreshCw size={17} />
              {generating ? "Đang tạo..." : "Tạo model"}
            </button>
            <button className="button" onClick={() => analyze()} disabled={loading}>
              <Sparkles size={17} />
              {loading ? "Đang phân tích..." : "Phân tích quyết định"}
            </button>
          </div>
          {error ? <p className="small" style={{ color: "var(--red)" }}>{error}</p> : null}
          </section>

          <section>
            <h2>Học từ kết quả</h2>
            <p className="small">Ghi nhận outcome để hệ thống hiệu chỉnh confidence và case memory cho các lần phân tích sau.</p>
            <div className="row">
              <button className="button secondary" disabled={!result || learning} onClick={() => learn("success")}>Thành công</button>
              <button className="button secondary" disabled={!result || learning} onClick={() => learn("mixed")}>Lẫn lộn</button>
              <button className="button secondary" disabled={!result || learning} onClick={() => learn("failure")}>Thất bại</button>
            </div>
          </section>

          <section>
            <h2>Engineering Solver</h2>
            <p className="small">Paste bài toán kỹ thuật/OR ở đây. Hiện hỗ trợ transportation, quy hoạch động phân bổ tài nguyên, và cây xác suất.</p>
            <div className="field">
              <label>Bài toán kỹ thuật</label>
              <textarea
                value={engineeringText}
                onChange={(event) => setEngineeringText(event.target.value)}
                placeholder="Dán bài toán truyền tải điện, production mix, logistics..."
              />
            </div>
            <button className="button secondary" onClick={solveEngineeringText} disabled={engineeringLoading || !engineeringText.trim()}>
              <Sparkles size={16} />
              {engineeringLoading ? "Đang giải..." : "Giải bằng EDSS"}
            </button>
          </section>

          <section>
            <div className="history-head">
              <div>
                <h2>History bài đã giải</h2>
                <p className="small">Lưu trên trình duyệt local của máy này.</p>
              </div>
              <button className="icon-button" title="Xóa toàn bộ history" onClick={clearEngineeringHistory} disabled={!engineeringHistory.length}>
                <Trash2 size={15} />
              </button>
            </div>
            {engineeringHistory.length ? (
              <div className="history-list">
                {engineeringHistory.map((item) => (
                  <div className="history-item" key={item.id}>
                    <button className="history-main" onClick={() => loadEngineeringHistory(item)} title="Mở lại bài đã giải">
                      <strong>{item.title}</strong>
                      <span>{item.problemType || "unknown"} · {new Date(item.createdAt).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" })}</span>
                    </button>
                    <button className="icon-button" title="Xóa bài này khỏi history" onClick={() => deleteEngineeringHistory(item.id)}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="small">Chưa có bài nào trong history.</p>
            )}
          </section>
        </div>
      </aside>

      <section className="main">
        <div className="toolbar">
          <div>
            <strong className="toolbar-title">{winner ? <span className="winner">Đề xuất: {winner.name}</span> : "Decision workbench"}</strong>
            <div className="small">Expected utility, risk adjustment, regret, sensitivity, VOI, AI critique.</div>
          </div>
          <div className="toolbar-actions">
            <div className="tabs">
              <button className={chartMode === "map" ? "active" : ""} onClick={() => setChartMode("map")}>Map</button>
              <button className={chartMode === "logic" ? "active" : ""} onClick={() => setChartMode("logic")}>Logic</button>
              <button className={chartMode === "score" ? "active" : ""} onClick={() => setChartMode("score")}>Score</button>
              <button className={chartMode === "regret" ? "active" : ""} onClick={() => setChartMode("regret")}>Regret</button>
              <button className={chartMode === "tree" ? "active" : ""} onClick={() => setChartMode("tree")}>Tree</button>
            </div>
          </div>
        </div>

        <div className={`grid ${collapsedRegions.workbench ? "workbench-collapsed" : ""} ${collapsedRegions.insight ? "insight-collapsed" : ""}`}>
          {collapsedRegions.workbench ? (
            <button className="collapsed-rail" title="Mở vùng lựa chọn" onClick={toggleWorkbench}>
              <PanelRightOpen size={16} />
              <span>Lựa chọn</span>
            </button>
          ) : (
          <div className="panel workbench-panel">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
              <h2 style={{ margin: 0 }}>Lựa chọn và tình huống</h2>
              <div className="row">
                <button className="icon-button" title="Thu vùng lựa chọn" onClick={toggleWorkbench}>
                  <PanelLeftClose size={16} />
                </button>
                <button className="button secondary" onClick={() => setDecision({ ...decision, options: [...decision.options, emptyOption(decision.options.length)] })}>
                  <Plus size={16} /> Thêm
                </button>
              </div>
            </div>
            {decision.options.map((option, optionIndex) => {
              const optionResult = resultByOptionId.get(option.id);
              return (
              <div className="option" key={option.id}>
                <div className="option-head">
                  <div className="option-title-wrap">
                    <span className="option-index">Option {optionIndex + 1}</span>
                    <input className="option-title" value={option.name} onChange={(event) => updateOption(option.id, { name: event.target.value })} />
                  </div>
                  {optionResult ? (
                    <div className="score-pill" title="Risk-adjusted score">
                      {optionResult.risk_adjusted_score.toFixed(1)}
                    </div>
                  ) : null}
                  <button
                    className="icon-button"
                    title="Xóa lựa chọn"
                    disabled={decision.options.length <= 2}
                    onClick={() => setDecision({ ...decision, options: decision.options.filter((item) => item.id !== option.id) })}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
                <div className="row">
                  <div className="field">
                    <label>Chi phí</label>
                    <input type="number" value={option.cost} onChange={(event) => updateOption(option.id, { cost: Number(event.target.value) })} />
                  </div>
                  <div className="field">
                    <label>Khả năng đảo ngược</label>
                    <input type="number" min="0" max="1" step="0.05" value={option.reversibility} onChange={(event) => updateOption(option.id, { reversibility: Number(event.target.value) })} />
                  </div>
                </div>
                <div className="scenario-table">
                  <div className="scenario-row scenario-head">
                    <span>Kết quả</span>
                    <span>Xác suất</span>
                    <span>Giá trị</span>
                    <span>Đóng góp</span>
                  </div>
                  {option.scenarios.map((scenario, scenarioIndex) => (
                  <div className="scenario-row" key={`${option.id}-${scenarioIndex}`}>
                    <input
                      value={scenario.name}
                      onChange={(event) => {
                        const scenarios = option.scenarios.map((item, index) => (index === scenarioIndex ? { ...item, name: event.target.value } : item));
                        updateOption(option.id, { scenarios });
                      }}
                    />
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={scenario.probability}
                      onChange={(event) => {
                        const scenarios = option.scenarios.map((item, index) => (index === scenarioIndex ? { ...item, probability: Number(event.target.value) } : item));
                        updateOption(option.id, { scenarios });
                      }}
                    />
                    <input
                      type="number"
                      value={scenario.utility}
                      onChange={(event) => {
                        const scenarios = option.scenarios.map((item, index) => (index === scenarioIndex ? { ...item, utility: Number(event.target.value) } : item));
                        updateOption(option.id, { scenarios });
                      }}
                    />
                    <span className="scenario-contribution">
                      {optionResult?.scenarios?.[scenarioIndex]?.contribution.toFixed(1) ?? "-"}
                    </span>
                  </div>
                  ))}
                </div>
                <button
                  className="button secondary"
                  onClick={() =>
                    updateOption(option.id, {
                      scenarios: [...option.scenarios, { name: `Tình huống ${option.scenarios.length + 1}`, probability: 0.1, utility: 40, evidence: "" }]
                    })
                  }
                >
                  <Plus size={16} /> Thêm tình huống
                </button>
              </div>
              );
            })}
          </div>
          )}

          {collapsedRegions.insight ? (
            <button className="collapsed-rail" title="Mở vùng biểu đồ" onClick={toggleInsight}>
              <PanelLeftOpen size={16} />
              <span>Bản đồ</span>
            </button>
          ) : (
          <div className="panel insight-panel">
            <div className="insight-head">
              <h2>{result ? "Kết quả và bản đồ quyết định" : "Bản đồ quyết định"}</h2>
              <button className="icon-button" title="Thu vùng biểu đồ" onClick={toggleInsight}>
                <PanelRightClose size={16} />
              </button>
            </div>
            {result ? (
              <div className="recommendation-panel">
                <div>
                  <span className="eyebrow">Recommendation</span>
                  <h2>{winner?.name}</h2>
                  <p>{result.summary}</p>
                </div>
                <div className="recommendation-stats">
                  <div>
                    <span>Autonomous route</span>
                    <strong>{result.autonomy ? result.autonomy.selected_solver : "manual"}</strong>
                  </div>
                  <div>
                    <span>Score gap</span>
                    <strong>{scoreGap === null ? "n/a" : scoreGap.toFixed(1)}</strong>
                  </div>
                  <div>
                    <span>Top sensitivity</span>
                    <strong>{topSensitivity ? topSensitivity.variable : "n/a"}</strong>
                  </div>
                  <div>
                    <span>Learning events</span>
                    <strong>{result.autonomy?.learning_profile?.events ?? "n/a"}</strong>
                  </div>
                </div>
              </div>
            ) : null}
            <div ref={mapShellRef} className={`map-shell ${mapFullscreen ? "fullscreen" : ""}`}>
              {chartMode === "map" ? (
                <div className="map-controls" aria-label="Map controls">
                  <div className="map-legend">
                    {[
                      ["scenario", "Scenarios"],
                      ["factor", "Factors"],
                      ["evidence", "Evidence"]
                    ].map(([kind, label]) => (
                      <label key={kind}>
                        <input type="checkbox" checked={visibleMapKinds.includes(kind)} onChange={() => toggleMapKind(kind)} />
                        {label}
                      </label>
                    ))}
                  </div>
                  <button className="icon-button" title="Refresh map" onClick={() => setMapRefresh((value) => value + 1)}>
                    <RefreshCw size={16} />
                  </button>
                  <button className="icon-button" title="Download SVG map" onClick={downloadMapSvg}>
                    <Download size={16} />
                  </button>
                  <button className="icon-button" title="Download PNG map" onClick={downloadMapPng}>
                    <ImageDown size={16} />
                  </button>
                  <button className="icon-button" title={mapFullscreen ? "Tắt fullscreen" : "Fullscreen"} onClick={toggleMapFullscreen}>
                    {mapFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                  </button>
                </div>
              ) : null}
              <DecisionChart
                decision={decision}
                result={result}
                mode={chartMode}
                refreshToken={mapRefresh}
                visibleKinds={visibleMapKinds}
                onNodeSelect={setSelectedMapNode}
              />
              {chartMode === "map" && selectedMapNode ? (
                <div className="map-detail">
                  <strong>{selectedMapNode.label}</strong>
                  <span>{selectedMapNode.kind}</span>
                  <p>{selectedMapNode.detail}</p>
                </div>
              ) : null}
            </div>
            {result ? (
              <>
                <div className="metric-grid" style={{ marginTop: 14 }}>
                  <div className="metric"><span>Best score</span><strong>{winner?.risk_adjusted_score.toFixed(1)}</strong></div>
                  <div className="metric"><span>Expected utility</span><strong>{winner?.expected_utility.toFixed(1)}</strong></div>
                  <div className="metric"><span>Regret</span><strong>{winner?.expected_regret.toFixed(1)}</strong></div>
                  <div className="metric"><span>VOI</span><strong>{result.value_of_information.score.toFixed(1)}</strong></div>
                </div>
                {result.safety ? (
                  <div className={`safety-strip ${result.decision_gate === "needs_clarification" ? "warn" : ""}`}>
                    <strong>Gate: {result.decision_gate === "needs_clarification" ? "Cần hỏi thêm" : "Sẵn sàng phân tích"}</strong>
                    <span>Risk: {result.safety.risk_level}</span>
                    {result.autonomy ? <span>Case: {result.autonomy.case_type} ({Math.round(result.autonomy.confidence * 100)}%)</span> : null}
                    {result.safety.missing_fields.length ? <span>Thiếu: {result.safety.missing_fields.join(", ")}</span> : null}
                  </div>
                ) : null}
                <div className="analysis" style={{ marginTop: 16 }}>
                  <h2><BrainCircuit size={16} /> Phân tích</h2>
                  {result.summary}
                  {"\n\n"}
                  <strong>AI review:</strong>
                  {"\n"}
                  {result.ai_review}
                  {"\n\n"}
                  <strong>Giả định:</strong> {result.assumptions.join("; ")}
                  {"\n"}
                  <strong>Cảnh báo:</strong> {result.warnings.join("; ")}
                  {"\n"}
                  <strong>Học hệ thống:</strong> {result.learning_note}
                  {result.autonomy ? (
                    <>
                      {"\n"}
                      <strong>Autonomous:</strong> {result.autonomy.reason}
                    </>
                  ) : null}
                  {result.monte_carlo?.distributions?.length ? (
                    <>
                      {"\n\n"}
                      <strong>Monte Carlo:</strong>
                      {"\n"}
                      {result.monte_carlo.distributions.map((item) => `- ${item.option}: mean ${item.mean.toFixed(1)}, P10 ${item.p10.toFixed(1)}, P90 ${item.p90.toFixed(1)}, win ${(item.win_rate * 100).toFixed(0)}%`).join("\n")}
                    </>
                  ) : null}
                  {result.book_rag?.results?.length ? (
                    <>
                      {"\n\n"}
                      <strong>PDF RAG:</strong>
                      {"\n"}
                      {result.book_rag.results.map((item) => `- p.${item.page} ${item.section}: ${item.excerpt.slice(0, 180)}...`).join("\n")}
                    </>
                  ) : null}
                  {result.clarifying_questions?.length ? (
                    <>
                      {"\n\n"}
                      <strong>Câu hỏi cần hỏi thêm:</strong>
                    </>
                  ) : null}
                </div>
                {result.clarifying_questions?.length ? (
                  <div style={{ marginTop: 10 }}>
                    {result.clarifying_questions.map((question) => (
                      <div className="field" key={question}>
                        <label>{question}</label>
                        <textarea
                          value={answers[question] || ""}
                          onChange={(event) => setAnswers({ ...answers, [question]: event.target.value })}
                        />
                      </div>
                    ))}
                    <div className="row">
                      <button className="button secondary" onClick={applyClarifications}>
                        Cập nhật context
                      </button>
                      <button
                        className="button"
                        onClick={async () => {
                          const nextDecision = buildClarifiedDecision();
                          setDecision(nextDecision);
                          await analyze(nextDecision);
                        }}
                        disabled={loading}
                      >
                        Phân tích lại
                      </button>
                    </div>
                  </div>
                ) : null}
                <div className="analysis" style={{ marginTop: 12 }}>
                  {result.agent_trace?.length ? (
                    <>
                      <strong>Agent trace:</strong>
                      {"\n"}
                      {result.agent_trace.map((item) => `- ${item.step}: ${typeof item.output === "string" ? item.output : JSON.stringify(item.output)}`).join("\n")}
                    </>
                  ) : null}
                  {result.market_data?.length ? (
                    <>
                      {"\n\n"}
                      <strong>Market data:</strong>
                      {"\n"}
                      {result.market_data.map((item) => `- ${item.ticker}: price ${item.price ?? "n/a"}, 1D ${item.day_change_pct ?? "n/a"}%, 1M ${item.month_change_pct ?? "n/a"}%`).join("\n")}
                    </>
                  ) : null}
                  {result.web_research?.results?.length ? (
                    <>
                      {"\n\n"}
                      <strong>Web evidence:</strong>
                      {"\n"}
                      {result.web_research.results.map((item) => `- [${item.source_score ?? "?"}] ${item.title}: ${item.snippet}`).join("\n")}
                    </>
                  ) : null}
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <button className="button secondary" onClick={() => learn("mixed")} disabled={learning}>
                    <Save size={16} /> Lưu case
                  </button>
                  <span className="small"><Activity size={14} /> Memory được lưu trong backend/data/decision_memory.jsonl</span>
                </div>
              </>
            ) : null}
            {engineeringResult ? <EngineeringSolutionView data={engineeringResult} /> : null}
          </div>
          )}
        </div>
      </section>
    </main>
  );
}
