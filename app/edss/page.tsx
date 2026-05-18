"use client";

import "./edss.css";

import { useState, useEffect, useRef } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import mermaid from "mermaid";
import {
  Activity, BarChart3, Brain, Calculator,
  GitBranch, HelpCircle, Shield, Target, X, Zap
} from "lucide-react";

mermaid.initialize({ startOnLoad: false, theme: "dark" });

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8008";

type Tab = "solve" | "probability" | "data" | "risk" | "criteria";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "solve", label: "Solver", icon: <Calculator size={16} /> },
  { id: "probability", label: "Xác suất", icon: <Target size={16} /> },
  { id: "data", label: "Dữ liệu", icon: <BarChart3 size={16} /> },
  { id: "risk", label: "Rủi ro", icon: <Shield size={16} /> },
  { id: "criteria", label: "Tiêu chí", icon: <GitBranch size={16} /> },
];

const HELP_TOPICS = [
  {
    id: "lp_step_by_step",
    title: "1. Linear Programming Step-by-Step",
    status: "Implemented",
    computes: "Chuẩn hóa LP, giải LP bằng HiGHS, giải hình học cho 2 biến, simplex tableau cho dạng max Ax <= b, >=/= qua solver report.",
    endpoints: "/edss/solve-text, /edss/solve/lp, /edss/solve/lp/simplex",
  },
  {
    id: "duality",
    title: "2. Duality",
    status: "Implemented",
    computes: "Sinh dual cho primal max Ax <= b, so sánh primal/dual objective, kiểm tra complementary slackness.",
    endpoints: "/edss/solve/lp/duality",
  },
  {
    id: "sensitivity",
    title: "3. Sensitivity Analysis",
    status: "Implemented",
    computes: "Slack, binding constraints, shadow price, reduced cost, khoảng RHS và objective coefficient bằng parametric re-solve.",
    endpoints: "/edss/sensitivity, /edss/solve/lp/sensitivity-ranges",
  },
  {
    id: "mip",
    title: "4. Integer and Mixed-Integer Programming",
    status: "Implemented",
    computes: "Biến integer/binary, capital budgeting/facility template dạng structured, dùng scipy.milp hoặc enumeration nhỏ.",
    endpoints: "/edss/solve/mip",
  },
  {
    id: "goal_programming",
    title: "5. Goal Programming",
    status: "Implemented",
    computes: "Weighted goal programming bằng deviation variables d- và d+, preemptive sequential solve.",
    endpoints: "/edss/solve/goal-programming",
  },
  {
    id: "network",
    title: "6. Network Optimization",
    status: "Implemented",
    computes: "Shortest path, Bellman-Ford, MST, transportation, assignment, max-flow, min-cost-flow.",
    endpoints: "/edss/solve/shortest-path, /edss/solve/bellman-ford, /edss/solve/mst, /edss/solve/max-flow, /edss/solve/min-cost-flow",
  },
  {
    id: "dp",
    title: "7. Dynamic Programming",
    status: "Implemented",
    computes: "Resource allocation DP, production planning DP dạng bảng lợi nhuận, finite-stage DP, inventory DP nhiều kỳ, stochastic inventory simulation.",
    endpoints: "/edss/solve-text, /edss/inventory/dp, /edss/inventory/stochastic-sim",
  },
  {
    id: "inventory",
    title: "8. Inventory Models",
    status: "Implemented",
    computes: "EOQ, quantity discount, production lot size, reorder point, safety stock, newsvendor, deterministic DP, stochastic inventory simulation.",
    endpoints: "/edss/inventory/eoq, /edss/inventory/newsvendor, /edss/inventory/quantity-discount, /edss/inventory/dp, /edss/inventory/stochastic-sim",
  },
  {
    id: "queueing",
    title: "9. Queueing Theory",
    status: "Implemented",
    computes: "M/M/1, M/M/c, M/M/1/K, M/G/1, open queue network, rho, L, Lq, W, Wq và điều kiện ổn định.",
    endpoints: "/edss/queueing/mm1, /edss/queueing/mmc, /edss/queueing/mm1k, /edss/queueing/mg1, /edss/queueing/network",
  },
  {
    id: "markov",
    title: "10. Markov Chain / Random Process",
    status: "Implemented",
    computes: "P^n, steady-state distribution, absorbing Markov chain, expected time to absorption.",
    endpoints: "/edss/markov/n-step, /edss/markov/steady-state, /edss/markov/absorbing",
  },
  {
    id: "simulation",
    title: "11. Simulation",
    status: "Implemented",
    computes: "Monte Carlo, risk samples, VaR/CVaR, inventory stochastic simulation, confidence-oriented output.",
    endpoints: "/analyze, /edss/risk/var, /edss/inventory/stochastic-sim",
  },
  {
    id: "nonlinear",
    title: "12. Nonlinear Programming",
    status: "Implemented",
    computes: "Quadratic NLP structured input, SLSQP solve, KKT notes, convexity eigenvalue check.",
    endpoints: "/edss/solve/nonlinear",
  },
  {
    id: "decision_tree",
    title: "13. Decision Tree",
    status: "Implemented",
    computes: "Decision/chance/outcome tree, rollback expected value, probability tree, Bayes event tree.",
    endpoints: "/edss/solve-text, /edss/decision-criteria",
  },
  {
    id: "influence_diagram",
    title: "14. Influence Diagram",
    status: "Implemented",
    computes: "Decision/chance/value nodes, dependency arcs, topological order, Mermaid report, decision-tree skeleton conversion.",
    endpoints: "/edss/influence-diagram/analyze, /edss/influence-diagram/to-tree",
  },
  {
    id: "probability",
    title: "15. Probability Assessment",
    status: "Implemented",
    computes: "Prior, conditional probability, Bayes update, independent events, distribution PMF/PDF/CDF, distribution fit.",
    endpoints: "/edss/probability/bayes, /edss/probability/independent, /edss/probability/distribution, /edss/probability/fit",
  },
  {
    id: "voi",
    title: "16. Value of Information",
    status: "Implemented",
    computes: "EVPI, EVSI/EVI với likelihood của test/survey, posterior probability và khuyến nghị mua thông tin.",
    endpoints: "/edss/information/evpi, /edss/information/evi",
  },
  {
    id: "utility",
    title: "17. Utility Theory",
    status: "Implemented",
    computes: "Expected utility, exponential utility, certainty equivalent, risk premium, fit risk tolerance từ preference questions.",
    endpoints: "/edss/utility/expected, /edss/utility/exponential-ce, /edss/utility/fit-risk-tolerance",
  },
  {
    id: "mada",
    title: "18. Multi-Attribute Decision Analysis",
    status: "Implemented",
    computes: "Weighted scoring, Pareto frontier, additive utility MAUT, AHP weights và consistency ratio.",
    endpoints: "/edss/solve, /edss/mada/additive-utility, /edss/mada/ahp",
  },
  {
    id: "risk",
    title: "19. Risk Analysis",
    status: "Implemented",
    computes: "Expected value, variance via samples, probability of loss, VaR, CVaR, worst-case sample.",
    endpoints: "/edss/risk/var",
  },
  {
    id: "teaching",
    title: "20. Explanation and Teaching",
    status: "Implemented",
    computes: "Markdown report từng bước cho các solver và behavioral bias audit cho subjective judgment.",
    endpoints: "Tự động trong markdown_report của các solver, /edss/behavioral/audit",
  },
];

async function api(path: string, body: unknown) {
  const res = await fetch(`${BACKEND}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

/* ═══════════════════════════════════════════════════════════════
   Markdown Renderer
   ═══════════════════════════════════════════════════════════════ */

function MermaidView({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const cleanChart = normalizeMermaidChart(chart);

  useEffect(() => {
    let cancelled = false;
    const id = `mermaid-${Math.random().toString(36).slice(2)}`;
    if (!ref.current) return;
    ref.current.textContent = "";
    mermaid
      .render(id, cleanChart)
      .then(({ svg }) => {
        if (!cancelled && ref.current) ref.current.innerHTML = svg;
      })
      .catch((err) => {
        if (ref.current) {
          ref.current.textContent = `Mermaid render error: ${err instanceof Error ? err.message : String(err)}`;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [cleanChart]);

  return <div ref={ref} className="mermaid-wrapper" style={{ margin: "1.5rem 0", display: "flex", justifyContent: "center", background: "var(--bg-secondary)", padding: "1rem", borderRadius: "8px" }} />;
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

function MarkdownView({ content }: { content: string }) {
  return (
    <div className="md-view">
      <link href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" rel="stylesheet" />
      <ReactMarkdown 
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        urlTransform={markdownUrlTransform}
        components={{
          img({ src, alt, ...props }) {
            if (!src) return null;
            return <img src={src} alt={alt || ""} {...props} />;
          },
          pre({ children, ...props }) {
            const raw = String((children as any)?.props?.children ?? "");
            const className = String((children as any)?.props?.className ?? "");
            if (className.includes("language-mermaid") || isMermaidChart(raw)) {
              return <MermaidView chart={raw} />;
            }
            return <pre {...props}>{children}</pre>;
          },
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const lang = match ? match[1] : "";
            const raw = String(children).replace(/\n$/, "");
            if (lang === "mermaid" || isMermaidChart(raw)) {
              return <MermaidView chart={raw} />;
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function HelpModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="edss-help-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="edss-help-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="edss-help-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="edss-help-head">
          <div>
            <h2 id="edss-help-title"><HelpCircle size={20} /> 20 chủ đề chương trình có thể tính toán</h2>
            <p>
              Các mục dưới đây cho biết EDSS đang dùng solver thật ở đâu, module nào đã đầy đủ,
              và module nào còn ở mức partial/scaffold.
            </p>
          </div>
          <button className="edss-icon-btn" onClick={onClose} aria-label="Đóng help">
            <X size={18} />
          </button>
        </div>

        <div className="edss-help-summary">
          <span><strong>Implemented</strong>: solver/report đang dùng được trực tiếp.</span>
          <span><strong>Partial</strong>: đã có engine chính nhưng chưa full textbook mọi biến thể.</span>
        </div>

        <div className="edss-help-grid">
          {HELP_TOPICS.map((topic) => (
            <article key={topic.id} className="edss-help-card">
              <div className="edss-help-card-head">
                <h3>{topic.title}</h3>
                <span className={`edss-help-status ${topic.status.toLowerCase()}`}>{topic.status}</span>
              </div>
              <p>{topic.computes}</p>
              <div className="edss-help-endpoint">{topic.endpoints}</div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Convert any JSON result to readable markdown
   ═══════════════════════════════════════════════════════════════ */

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
        // Table
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

function recognitionGateMarkdown(data: Record<string, any>): string {
  const gate = data?.recognition_gate;
  if (!gate) return "";
  const evidence = Array.isArray(gate.evidence) ? gate.evidence : [];
  const required = Array.isArray(gate.required_slots) ? gate.required_slots : [];
  const filled = Array.isArray(gate.filled_slots) ? gate.filled_slots : [];
  const missing = Array.isArray(gate.missing_slots) ? gate.missing_slots : [];
  return [
    "## 1. Nhận dạng dạng toán",
    "",
    `- **Dạng toán chính:** ${gate.recognized_problem_type || "unknown"}`,
    `- **Dạng toán phụ:** ${gate.recognized_subtype || "unknown"}`,
    `- **Mức tin cậy:** ${gate.confidence ?? "n/a"}`,
    `- **Quyết định gate:** ${gate.decision_to_solve || "unknown"}`,
    "",
    "**Dấu hiệu nhận dạng:**",
    ...(evidence.length ? evidence.map((item: string) => `- ${item}`) : ["- Chưa đủ tín hiệu rõ ràng."]),
    "",
    "**Dữ liệu cần có:**",
    ...(required.length ? required.map((item: string) => `- ${item}`) : ["- Chưa xác định."]),
    "",
    "**Dữ liệu đã có:**",
    ...(filled.length ? filled.map((item: string) => `- ${item}`) : ["- Chưa đủ slot bắt buộc."]),
    "",
    "**Dữ liệu còn thiếu nếu có:**",
    ...(missing.length ? missing.map((item: string) => `- ${item}`) : ["- Không có slot bắt buộc bị thiếu."]),
    "",
    gate.decision_to_solve === "solve"
      ? "**Kết luận:** Đủ điều kiện để xây mô hình, chọn solver và giải."
      : "**Kết luận:** Chưa đủ điều kiện để đưa lời giải cuối cùng; cần làm rõ trước.",
    "",
  ].join("\n");
}

/* ═══════════════════════════════════════════════════════════════
   Solver Tab
   ═══════════════════════════════════════════════════════════════ */

function SolverTab() {
  const [text, setText] = useState("");
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(false);

  async function solve() {
    setLoading(true);
    setMarkdown("");
    try {
      const data = await api("/edss/solve-text", { text });

      // Check for markdown_report in LP result
      const lpResult = data?.solved?.result;
      const gateMd = recognitionGateMarkdown(data);
      if (lpResult?.markdown_report) {
        setMarkdown(gateMd + lpResult.markdown_report);
      } else if (data?.status === "solved") {
        // Generate markdown from the JSON result
        let md = gateMd || `# Lời giải\n\n**Status**: ${data.status}\n\n`;
        if (data.solved?.problem_type) {
          md += `**Loại bài toán**: ${data.solved.problem_type}\n\n`;
        }
        if (data.problem?.formulation) {
          md += `### 🤖 Mô hình được AI trích xuất:\n\`\`\`\n${data.problem.formulation}\n\`\`\`\n\n`;
        }
        if (data.solved?.result) {
          md += jsonToMarkdown(data.solved.result);
        }
        if (data.solved?.recommendation_explanation) {
          md += `\n\n## Khuyến nghị\n\n${data.solved.recommendation_explanation}\n`;
        }
        setMarkdown(md);
      } else if (data?.status === "needs_clarification") {
        let md = `${gateMd}## ⚠️ Cần thêm thông tin\n\n${data.message || ""}\n\n`;
        if (data.questions) {
          md += "### Câu hỏi:\n\n";
          for (const q of data.questions) md += `- ${q}\n`;
        }
        setMarkdown(md);
      } else {
        setMarkdown(`## ❌ Lỗi\n\n\`\`\`json\n${JSON.stringify(data, null, 2)}\n\`\`\``);
      }
    } catch {
      setMarkdown("## ❌ Lỗi kết nối\n\nKhông thể kết nối đến backend. Kiểm tra server đang chạy.");
    }
    setLoading(false);
  }

  return (
    <div className="edss-tab">
      <h2><Zap size={20} /> Engineering Solver</h2>
      <p className="hint">
        Nhập bài toán bằng ngôn ngữ tự nhiên. Hệ thống tự nhận diện loại bài toán và áp dụng
        phương pháp phù hợp: LP, Transportation, Assignment, DP, Decision Tree, Network Flow.
      </p>
      <textarea
        className="edss-input"
        rows={8}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={"Ví dụ: Một máy bay chở hàng có 3 khoang...\nHoặc paste bài toán transportation, DP, xác suất..."}
      />
      <button className="edss-btn primary" onClick={solve} disabled={loading || !text.trim()}>
        {loading ? "Đang giải..." : "Giải bài toán"}
      </button>
      {markdown && <MarkdownView content={markdown} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Probability Tab
   ═══════════════════════════════════════════════════════════════ */

function ProbabilityTab() {
  const [mode, setMode] = useState<"bayes" | "dist" | "independent">("bayes");
  const [prior, setPrior] = useState(0.3);
  const [sensitivity, setSensitivity] = useState(0.9);
  const [fpr, setFpr] = useState(0.2);
  const [distName, setDistName] = useState("normal");
  const [mean, setMean] = useState(100);
  const [std, setStd] = useState(15);
  const [xVals, setXVals] = useState("85,100,115");
  const [probs, setProbs] = useState("0.5,0.3,0.8");
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(false);

  async function compute() {
    setLoading(true);
    try {
      let data;
      if (mode === "bayes") {
        data = await api("/edss/probability/bayes", { prior, sensitivity, false_positive_rate: fpr, observed_positive: true });
      } else if (mode === "dist") {
        data = await api("/edss/probability/distribution", { distribution: distName, params: { mean, std }, x: xVals.split(",").map(Number) });
      } else {
        data = await api("/edss/probability/independent", { probabilities: probs.split(",").map(Number) });
      }
      setMarkdown(`# Probability Result\n\n${jsonToMarkdown(data)}`);
    } catch { setMarkdown("## ❌ Lỗi kết nối"); }
    setLoading(false);
  }

  return (
    <div className="edss-tab">
      <h2><Target size={20} /> Probability Engine</h2>
      <div className="edss-subtabs">
        <button className={mode === "bayes" ? "active" : ""} onClick={() => setMode("bayes")}>Bayes</button>
        <button className={mode === "dist" ? "active" : ""} onClick={() => setMode("dist")}>Distribution</button>
        <button className={mode === "independent" ? "active" : ""} onClick={() => setMode("independent")}>Joint</button>
      </div>
      {mode === "bayes" && (
        <div className="edss-form">
          <label>Prior P(H) <input type="number" step="0.01" min="0" max="1" value={prior} onChange={(e) => setPrior(Number(e.target.value))} /></label>
          <label>Sensitivity P(E+|H) <input type="number" step="0.01" min="0" max="1" value={sensitivity} onChange={(e) => setSensitivity(Number(e.target.value))} /></label>
          <label>False Positive P(E+|¬H) <input type="number" step="0.01" min="0" max="1" value={fpr} onChange={(e) => setFpr(Number(e.target.value))} /></label>
        </div>
      )}
      {mode === "dist" && (
        <div className="edss-form">
          <label>Distribution
            <select value={distName} onChange={(e) => setDistName(e.target.value)}>
              <option value="normal">Normal</option><option value="exponential">Exponential</option>
              <option value="poisson">Poisson</option><option value="binomial">Binomial</option><option value="uniform">Uniform</option>
            </select>
          </label>
          <label>Mean/Mu <input type="number" value={mean} onChange={(e) => setMean(Number(e.target.value))} /></label>
          <label>Std/Rate <input type="number" value={std} onChange={(e) => setStd(Number(e.target.value))} /></label>
          <label>X values (comma-sep) <input value={xVals} onChange={(e) => setXVals(e.target.value)} /></label>
        </div>
      )}
      {mode === "independent" && (
        <div className="edss-form">
          <label>Probabilities (comma-sep) <input value={probs} onChange={(e) => setProbs(e.target.value)} /></label>
        </div>
      )}
      <button className="edss-btn primary" onClick={compute} disabled={loading}>{loading ? "Computing..." : "Tính toán"}</button>
      {markdown && <MarkdownView content={markdown} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Data Analysis Tab
   ═══════════════════════════════════════════════════════════════ */

function DataTab() {
  const [rawData, setRawData] = useState("10,20,30,40,50,60,70,80,90,100");
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(false);

  async function analyze(endpoint: string) {
    setLoading(true);
    try {
      const nums = rawData.split(",").map(Number).filter((n) => !isNaN(n));
      let data;
      if (endpoint === "fit") {
        data = await api("/edss/probability/fit", { data: nums });
      } else {
        data = await api(`/edss/data/${endpoint}`, { data: nums, label: "input_data" });
      }
      setMarkdown(`# Data Analysis\n\n${jsonToMarkdown(data)}`);
    } catch { setMarkdown("## ❌ Lỗi kết nối"); }
    setLoading(false);
  }

  return (
    <div className="edss-tab">
      <h2><BarChart3 size={20} /> Data Analysis</h2>
      <p className="hint">Nhập dữ liệu số cách nhau bằng dấu phẩy.</p>
      <textarea className="edss-input" rows={3} value={rawData} onChange={(e) => setRawData(e.target.value)} placeholder="Nhập số..." />
      <div className="edss-btn-group">
        <button className="edss-btn" onClick={() => analyze("stats")} disabled={loading}>Thống kê</button>
        <button className="edss-btn" onClick={() => analyze("histogram")} disabled={loading}>Histogram</button>
        <button className="edss-btn" onClick={() => analyze("fit")} disabled={loading}>Fit Distribution</button>
      </div>
      {markdown && <MarkdownView content={markdown} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Risk Tab
   ═══════════════════════════════════════════════════════════════ */

function RiskTab() {
  const [samples, setSamples] = useState("");
  const [confidence, setConfidence] = useState(0.95);
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(false);

  async function compute() {
    setLoading(true);
    try {
      let nums: number[];
      if (samples.trim()) {
        nums = samples.split(",").map(Number).filter((n) => !isNaN(n));
      } else {
        nums = Array.from({ length: 1000 }, () => (Math.random() - 0.3) * 200);
      }
      const data = await api("/edss/risk/var", { samples: nums, confidence });
      setMarkdown(`# Risk Analysis — VaR/CVaR\n\n${jsonToMarkdown(data)}`);
    } catch { setMarkdown("## ❌ Lỗi kết nối"); }
    setLoading(false);
  }

  return (
    <div className="edss-tab">
      <h2><Shield size={20} /> Risk Analysis — VaR / CVaR</h2>
      <p className="hint">Nhập samples (profit/loss) hoặc để trống cho demo data.</p>
      <textarea className="edss-input" rows={3} value={samples} onChange={(e) => setSamples(e.target.value)} placeholder="Nhập samples..." />
      <div className="edss-form">
        <label>Confidence Level <input type="number" step="0.01" min="0.5" max="0.99" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></label>
      </div>
      <button className="edss-btn primary" onClick={compute} disabled={loading}>{loading ? "Computing..." : "Tính VaR/CVaR"}</button>
      {markdown && <MarkdownView content={markdown} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Decision Criteria Tab
   ═══════════════════════════════════════════════════════════════ */

function CriteriaTab() {
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(false);
  const [payoffText, setPayoffText] = useState("Drill:Oil=600,NoOil=-100\nNoDrill:Oil=0,NoOil=0");
  const [probText, setProbText] = useState("Oil=0.4,NoOil=0.6");

  async function compute() {
    setLoading(true);
    try {
      const states: { name: string; probability: number }[] = [];
      probText.split(",").forEach((p) => { const [n, v] = p.trim().split("="); if (n && v) states.push({ name: n.trim(), probability: Number(v) }); });
      const alternatives: { name: string }[] = [];
      const payoff_matrix: { alternative: string; state: string; payoff: number }[] = [];
      payoffText.split("\n").forEach((line) => {
        const [altName, rest] = line.split(":");
        if (!altName || !rest) return;
        const alt = altName.trim();
        alternatives.push({ name: alt });
        rest.split(",").forEach((pair) => { const [s, v] = pair.trim().split("="); if (s && v) payoff_matrix.push({ alternative: alt, state: s.trim(), payoff: Number(v) }); });
      });
      const data = await api("/edss/decision-criteria", { alternatives, states, payoff_matrix, context: { title: "Decision criteria analysis" } });
      setMarkdown(`# Decision Criteria Analysis\n\n${jsonToMarkdown(data)}`);
    } catch { setMarkdown("## ❌ Lỗi kết nối"); }
    setLoading(false);
  }

  return (
    <div className="edss-tab">
      <h2><GitBranch size={20} /> Decision Criteria</h2>
      <p className="hint">So sánh 5 tiêu chí: Maximin, Maximax, Minimax Regret, Hurwicz, Expected Value.</p>
      <div className="edss-form">
        <label>States & Probabilities <input value={probText} onChange={(e) => setProbText(e.target.value)} /></label>
        <label>Payoff Matrix (Alt:State=Payoff)
          <textarea className="edss-input" rows={4} value={payoffText} onChange={(e) => setPayoffText(e.target.value)} />
        </label>
      </div>
      <button className="edss-btn primary" onClick={compute} disabled={loading}>{loading ? "Computing..." : "So sánh tiêu chí"}</button>
      {markdown && <MarkdownView content={markdown} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Main Dashboard
   ═══════════════════════════════════════════════════════════════ */

export default function EDSSDashboard() {
  const [tab, setTab] = useState<Tab>("solve");
  const [showHelp, setShowHelp] = useState(false);

  return (
    <main className="edss-shell">
      <header className="edss-header">
        <div className="edss-brand">
          <Brain size={28} />
          <div>
            <h1>Engineering Decision Intelligence System</h1>
            <p>ECE 307 — Techniques for Engineering Decisions</p>
          </div>
        </div>
        <div className="edss-header-actions">
          <button className="edss-btn secondary" onClick={() => setShowHelp(true)}>
            <HelpCircle size={16} /> Help
          </button>
          <a href="/" className="edss-btn secondary">← Decision Workbench</a>
        </div>
      </header>

      <nav className="edss-nav">
        {TABS.map((t) => (
          <button key={t.id} className={`edss-nav-btn ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
            {t.icon}<span>{t.label}</span>
          </button>
        ))}
      </nav>

      <div className="edss-content">
        {tab === "solve" && <SolverTab />}
        {tab === "probability" && <ProbabilityTab />}
        {tab === "data" && <DataTab />}
        {tab === "risk" && <RiskTab />}
        {tab === "criteria" && <CriteriaTab />}
      </div>

      <footer className="edss-footer">
        <p>Good Decision ≠ Lucky Outcome · Solver thật, không LLM tính toán · Mọi bước được audit</p>
      </footer>

      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
    </main>
  );
}
