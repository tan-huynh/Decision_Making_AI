"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, BrainCircuit, Download, ImageDown, Maximize2, Minimize2, Plus, RefreshCw, Save, Sparkles, Trash2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
    assumptions?: string[];
  };
  solved?: {
    problem_type?: string;
    result?: {
      status?: string;
      solver?: string;
      objective_value?: number;
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
};

function EngineeringSolutionView({ data }: { data: EngineeringResult }) {
  const graph = data.problem?.graph;
  const result = data.solved?.result;
  const resourceAllocation = data.problem?.resource_allocation;
  const allocations = result?.allocations || [];
  const plants = Object.keys(graph?.supplies || {});
  const cities = Object.keys(graph?.demands || {});
  const allocationMap = new Map(allocations.map((item) => [`${item.from}->${item.to}`, item]));
  const percent = (value?: number | null) => (value == null ? "N/A" : `${(value * 100).toFixed(2)}%`);

  if (data.status !== "solved" || !result) {
    return (
      <div className="analysis" style={{ marginTop: 16 }}>
        <h2>Engineering Solver Result</h2>
        <p>{data.message || "Không đủ dữ liệu để khuyến nghị chắc chắn."}</p>
        {data.questions?.length ? data.questions.map((question) => <p key={question}>- {question}</p>) : null}
      </div>
    );
  }

  if (result.markdown_report) {
    return (
      <div className="md-view" style={{ marginTop: 16 }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {result.markdown_report}
        </ReactMarkdown>
      </div>
    );
  }

  if (data.solved?.problem_type === "dynamic_programming" && resourceAllocation) {
    const resourceName = resourceAllocation.resource_name || "resource";
    return (
      <div className="solver-result">
        <div className="solver-head">
          <div>
            <h2>Lời giải quy hoạch động</h2>
            <p>Resource allocation dynamic programming</p>
          </div>
          <strong>{result.objective_value?.toFixed(0)}</strong>
        </div>

        <h3>1. Mô hình</h3>
        <p>Trạng thái <code>F_k(s)</code>: lợi nhuận lớn nhất sau <code>k</code> ngày khi đã phân bổ tổng cộng <code>s</code> tấn.</p>
        <p>Công thức truy hồi: <code>F_k(s) = max_x {"{ profit_k(x) + F_{k-1}(s-x) }"}</code>, với <code>x = 0..3</code> và <code>x ≤ s</code>.</p>
        <p>Mục tiêu: maximize <code>F_3(7)</code>.</p>

        <h3>2. Bảng lợi nhuận đầu vào</h3>
        <div className="table-wrap">
          <table className="solver-table">
            <thead>
              <tr><th>Ngày</th>{resourceAllocation.stage_returns[0].map((_, index) => <th key={index}>{index} tấn</th>)}</tr>
            </thead>
            <tbody>
              {resourceAllocation.stage_returns.map((row, index) => (
                <tr key={index}>
                  <th>Ngày {index + 1}</th>
                  {row.map((value, amount) => <td key={amount}>{value}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3>3. Phân bổ tối ưu</h3>
        <div className="table-wrap">
          <table className="solver-table compact">
            <thead><tr><th>Ngày</th><th>Sản xuất</th><th>Lợi nhuận</th></tr></thead>
            <tbody>
              {(result.allocation || []).map((item) => (
                <tr key={item.stage}>
                  <td>Ngày {item.stage}</td>
                  <td>{Number(item[resourceName as keyof typeof item] ?? item.tons ?? item.resource ?? 0)} tấn</td>
                  <td>{item.profit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p><strong>Lợi nhuận tối đa: {result.objective_value?.toFixed(0)} × 10^5 USD.</strong></p>

        <h3>4. Bảng DP</h3>
        {(result.dp_tables || []).map((table) => (
          <div className="table-wrap" key={table.stage}>
            <table className="solver-table compact">
              <thead><tr><th>Ngày {table.stage}</th><th>F(s)</th><th>Chọn x</th></tr></thead>
              <tbody>
                {Object.keys(table.values).map((state) => (
                  <tr key={state}>
                    <td>s = {state}</td>
                    <td>{Number(table.values[state]).toFixed(0)}</td>
                    <td>{table.choice[state]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}

        <h3>5. Khuyến nghị</h3>
        <p>{result.recommendation}</p>
        <p>{data.solved?.recommendation_explanation}</p>
        <p>Đây là quyết định tốt theo mô hình vì nó xét toàn bộ tổ hợp phân bổ 7 tấn qua 3 ngày, không chọn tham lam ngày có lợi nhuận cục bộ cao nhất.</p>
      </div>
    );
  }

  if (data.solved?.problem_type === "decision_tree" && data.problem?.probability_tree) {
    const successDecimal = (result.success_probability ?? 0).toFixed(2);
    const failureDecimal = (result.failure_probability ?? 0).toFixed(2);
    return (
      <div className="solver-result">
        <div className="solver-head">
          <div>
            <h2>Lời giải cây xác suất</h2>
            <p>Binary probability event tree</p>
          </div>
          <strong>{percent(result.queries?.at_least_one_success)}</strong>
        </div>

        <h3>1. Mô hình</h3>
        <p>Gọi <code>T</code> là biến cố khoan trúng dầu, <code>K</code> là biến cố không trúng dầu.</p>
        <p><code>P(T) = {percent(result.success_probability)}</code>, <code>P(K) = 1 - P(T) = {percent(result.failure_probability)}</code>.</p>
        <p>Giả định hai giếng độc lập, nên xác suất của một đường đi bằng tích xác suất các nhánh.</p>

        <h3>2. Sơ đồ hình cây</h3>
        <div className="prob-tree">
          <div className="prob-node root">Bắt đầu</div>
          <div className="prob-level">
            <div className="prob-node success">Giếng 1: Trúng dầu<br /><span>{percent(result.success_probability)}</span></div>
            <div className="prob-node failure">Giếng 1: Không trúng dầu<br /><span>{percent(result.failure_probability)}</span></div>
          </div>
          <div className="prob-level outcomes">
            {(result.outcomes || []).map((outcome) => (
              <div className="prob-node outcome" key={outcome.label}>
                {outcome.events.map((event, index) => <span key={`${event}-${index}`}>Giếng {index + 1}: {event}</span>)}
                <strong>{percent(outcome.probability)}</strong>
              </div>
            ))}
          </div>
        </div>

        <h3>3. Bốn biến cố đơn</h3>
        <div className="table-wrap">
          <table className="solver-table compact">
            <thead><tr><th>Biến cố</th><th>Cách tính</th><th>Xác suất</th></tr></thead>
            <tbody>
              {(result.outcomes || []).map((outcome) => (
                <tr key={outcome.label}>
                  <td>{outcome.label}</td>
                  <td>{outcome.events.map((event) => event === data.problem?.probability_tree?.success_label ? successDecimal : failureDecimal).join(" × ")}</td>
                  <td>{percent(outcome.probability)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3>4. Câu b</h3>
        <p>
          Xác suất trúng dầu ở giếng thứ nhất và không trúng dầu ở giếng thứ hai:
          <br />
          <code>P(T1 ∩ K2) = {successDecimal} × {failureDecimal} = {percent(result.queries?.first_success_second_failure)}</code>.
        </p>

        <h3>5. Câu c</h3>
        <p>
          Xác suất trúng dầu ít nhất một giếng:
          <br />
          <code>P(ít nhất một T) = 1 - P(K1 ∩ K2) = 1 - {failureDecimal}² = {percent(result.queries?.at_least_one_success)}</code>.
        </p>

        <h3>6. Giả định</h3>
        <ul className="solver-list">{data.problem.assumptions?.map((item) => <li key={item}>{item}</li>)}</ul>
      </div>
    );
  }

  if (!graph) {
    return (
      <div className="analysis" style={{ marginTop: 16 }}>
        <h2>Engineering Solver Result</h2>
        {JSON.stringify(data, null, 2)}
      </div>
    );
  }

  return (
    <div className="solver-result">
      <div className="solver-head">
        <div>
          <h2>Lời giải chi tiết</h2>
          <p>Transportation / transmission cost minimization</p>
        </div>
        <strong>{result.objective_value?.toFixed(0)}</strong>
      </div>

      <h3>1. Mô hình toán học</h3>
      <p>Biến quyết định: <code>x_ij</code> là MW truyền từ nhà máy <code>i</code> đến thành phố <code>j</code>.</p>
      <p>Hàm mục tiêu: minimize <code>Σ c_ij x_ij</code>.</p>
      <p>Ràng buộc: tổng phát từ mỗi nhà máy bằng công suất cung cấp; tổng nhận tại mỗi thành phố bằng nhu cầu; <code>x_ij ≥ 0</code>.</p>

      <h3>2. Phân bổ tối ưu</h3>
      <div className="table-wrap">
        <table className="solver-table">
          <thead>
            <tr>
              <th>Nhà máy / Thành phố</th>
              {cities.map((city) => <th key={city}>{city}</th>)}
              <th>Tổng phát</th>
            </tr>
          </thead>
          <tbody>
            {plants.map((plant) => (
              <tr key={plant}>
                <th>{plant}</th>
                {cities.map((city) => {
                  const item = allocationMap.get(`${plant}->${city}`);
                  return <td key={city}>{item?.amount || 0}</td>;
                })}
                <td>{graph.supplies?.[plant]}</td>
              </tr>
            ))}
            <tr>
              <th>Nhu cầu</th>
              {cities.map((city) => <td key={city}>{graph.demands?.[city]}</td>)}
              <td>{cities.reduce((sum, city) => sum + Number(graph.demands?.[city] || 0), 0)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>3. Breakdown chi phí</h3>
      <ul className="solver-list">
        {allocations.map((item) => (
          <li key={`${item.from}-${item.to}`}>
            {item.from} → {item.to}: {item.amount} × {item.unit_cost} = {(item.amount * item.unit_cost).toFixed(0)}
          </li>
        ))}
      </ul>
      <p><strong>Tổng chi phí tối thiểu: {result.objective_value?.toFixed(0)}</strong></p>

      <h3>4. Chứng chỉ tối ưu</h3>
      <p>
        {result.optimality_certificate?.is_optimal
          ? "MODI/reduced cost check: tất cả reduced cost của ô chưa dùng đều không âm, nên nghiệm hiện tại là tối ưu."
          : "Solver chưa có chứng chỉ tối ưu đầy đủ; cần chạy LP/min-cost flow để xác nhận."}
      </p>
      {result.optimality_certificate?.reduced_costs ? (
        <div className="table-wrap">
          <table className="solver-table compact">
            <thead>
              <tr><th>Ô chưa dùng</th><th>Reduced cost</th></tr>
            </thead>
            <tbody>
              {Object.entries(result.optimality_certificate.reduced_costs).map(([route, value]) => (
                <tr key={route}><td>{route}</td><td>{Number(value).toFixed(0)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <h3>5. Khuyến nghị</h3>
      <p>{result.recommendation}</p>
      <p>{data.solved?.recommendation_explanation}</p>
      {data.problem?.assumptions?.length ? (
        <>
          <h3>6. Giả định</h3>
          <ul className="solver-list">{data.problem.assumptions.map((item) => <li key={item}>{item}</li>)}</ul>
        </>
      ) : null}
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
  const [chartMode, setChartMode] = useState<"map" | "score" | "regret" | "tree">("map");
  const [error, setError] = useState("");
  const [engineeringText, setEngineeringText] = useState("");
  const [engineeringResult, setEngineeringResult] = useState<EngineeringResult | null>(null);
  const [engineeringLoading, setEngineeringLoading] = useState(false);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [mapRefresh, setMapRefresh] = useState(0);
  const [mapFullscreen, setMapFullscreen] = useState(false);
  const [visibleMapKinds, setVisibleMapKinds] = useState(["scenario", "factor", "evidence"]);
  const [selectedMapNode, setSelectedMapNode] = useState<{ label: string; kind: string; detail: string } | null>(null);
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

  const winner = useMemo(() => result?.option_results[0], [result]);
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
    } catch (err) {
      setError(err instanceof Error ? err.message : "EDSS solve failed");
    } finally {
      setEngineeringLoading(false);
    }
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

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <h1>AI-Powered Decision Making</h1>
          <span className="status" title={status.error || "Ollama status"}>
            <span className={`dot ${status.ok ? "ok" : ""}`} />
            Ollama
          </span>
        </div>

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
      </aside>

      <section className="main">
        <div className="toolbar">
          <div>
            <strong>{winner ? <span className="winner">Đề xuất: {winner.name}</span> : "Decision workbench"}</strong>
            <div className="small">Expected utility, risk adjustment, regret, sensitivity, VOI, AI critique.</div>
          </div>
          <div className="tabs">
            <button className={chartMode === "map" ? "active" : ""} onClick={() => setChartMode("map")}>Map</button>
            <button className={chartMode === "score" ? "active" : ""} onClick={() => setChartMode("score")}>Score</button>
            <button className={chartMode === "regret" ? "active" : ""} onClick={() => setChartMode("regret")}>Regret</button>
            <button className={chartMode === "tree" ? "active" : ""} onClick={() => setChartMode("tree")}>Tree</button>
          </div>
        </div>

        <div className="grid">
          <div className="panel">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
              <h2 style={{ margin: 0 }}>Lựa chọn và tình huống</h2>
              <button className="button secondary" onClick={() => setDecision({ ...decision, options: [...decision.options, emptyOption(decision.options.length)] })}>
                <Plus size={16} /> Thêm
              </button>
            </div>
            {decision.options.map((option, optionIndex) => (
              <div className="option" key={option.id}>
                <div className="option-head">
                  <input className="option-title" value={option.name} onChange={(event) => updateOption(option.id, { name: event.target.value })} />
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
                {option.scenarios.map((scenario, scenarioIndex) => (
                  <div className="row" key={`${option.id}-${scenarioIndex}`}>
                    <input
                      value={scenario.name}
                      onChange={(event) => {
                        const scenarios = option.scenarios.map((item, index) => (index === scenarioIndex ? { ...item, name: event.target.value } : item));
                        updateOption(option.id, { scenarios });
                      }}
                    />
                    <input
                      title="Xác suất"
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
                      title="Utility"
                      type="number"
                      value={scenario.utility}
                      onChange={(event) => {
                        const scenarios = option.scenarios.map((item, index) => (index === scenarioIndex ? { ...item, utility: Number(event.target.value) } : item));
                        updateOption(option.id, { scenarios });
                      }}
                    />
                  </div>
                ))}
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
            ))}
          </div>

          <div className="panel">
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
        </div>
      </section>
    </main>
  );
}
