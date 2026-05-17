"use client";

import "./edss.css";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity, BarChart3, Brain, Calculator,
  GitBranch, Shield, Target, Zap
} from "lucide-react";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8008";

type Tab = "solve" | "probability" | "data" | "risk" | "criteria";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "solve", label: "Solver", icon: <Calculator size={16} /> },
  { id: "probability", label: "Xác suất", icon: <Target size={16} /> },
  { id: "data", label: "Dữ liệu", icon: <BarChart3 size={16} /> },
  { id: "risk", label: "Rủi ro", icon: <Shield size={16} /> },
  { id: "criteria", label: "Tiêu chí", icon: <GitBranch size={16} /> },
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

function MarkdownView({ content }: { content: string }) {
  return (
    <div className="md-view">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
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
      if (lpResult?.markdown_report) {
        setMarkdown(lpResult.markdown_report);
      } else if (data?.status === "solved") {
        // Generate markdown from the JSON result
        let md = `# Lời giải\n\n**Status**: ${data.status}\n\n`;
        if (data.solved?.problem_type) {
          md += `**Loại bài toán**: ${data.solved.problem_type}\n\n`;
        }
        if (data.solved?.result) {
          md += jsonToMarkdown(data.solved.result);
        }
        if (data.solved?.recommendation_explanation) {
          md += `\n\n## Khuyến nghị\n\n${data.solved.recommendation_explanation}\n`;
        }
        setMarkdown(md);
      } else if (data?.status === "needs_clarification") {
        let md = `## ⚠️ Cần thêm thông tin\n\n${data.message || ""}\n\n`;
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
        <a href="/" className="edss-btn secondary">← Decision Workbench</a>
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
    </main>
  );
}
