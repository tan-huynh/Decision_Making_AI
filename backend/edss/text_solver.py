from __future__ import annotations

import json
import re
from typing import Any

from .router import solve_problem
from .cargo_lp_parser import parse_cargo_lp_text


def parse_probability_tree_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    has_probability_language = any(token in lowered for token in ["xác suất", "xac suat", "probability", "biến cố", "bien co"])
    has_tree_language = any(token in lowered for token in ["hình cây", "decision tree", "sơ đồ", "so do", "tree"])
    if not has_probability_language and not has_tree_language:
        return None

    percent_match = re.search(r"(\d+(?:[,.]\d+)?)\s*%", text)
    if not percent_match:
        return None
    success_probability = float(percent_match.group(1).replace(",", ".")) / 100

    trials = 2
    numeric_trial_match = re.search(r"(?:khoan|drill)\s+(\d+)\s+(?:giếng|gieng|wells?)", text, flags=re.I)
    if numeric_trial_match:
        trials = int(numeric_trial_match.group(1))
    elif re.search(r"(?:khoan|drill)\s+hai\s+(?:giếng|gieng)", text, flags=re.I):
        trials = 2

    if trials < 2:
        return None

    return {
        "context": {
            "title": "Oil drilling probability tree",
            "domain": "probability",
            "decision_maker": "exploration analyst",
            "objective_direction": "compute",
            "unit": "probability",
            "description": text[:5000],
        },
        "problem_type": "decision_tree",
        "probability_tree": {
            "success_probability": success_probability,
            "trials": trials,
            "success_label": "Trúng dầu",
            "failure_label": "Không trúng dầu",
        },
        "assumptions": [
            "Xác suất trúng dầu của mỗi giếng là như nhau.",
            "Kết quả khoan giữa các giếng độc lập nếu đề bài không nêu phụ thuộc địa chất.",
            "Mỗi giếng chỉ có hai trạng thái: trúng dầu hoặc không trúng dầu.",
        ],
    }


def parse_power_transportation_text(text: str) -> dict[str, Any] | None:
    supply_matches = re.findall(r"Nhà máy\s+(\d+)\s*\|\s*(\d+(?:\.\d+)?)\s*MW", text, flags=re.I)
    demand_matches = re.findall(r"Thành phố\s+(\d+)\s*\|\s*(\d+(?:\.\d+)?)\s*MW", text, flags=re.I)
    cost_matches = re.findall(r"C(\d)(\d)\s*=\s*(\d+(?:\.\d+)?)", text, flags=re.I)
    if len(supply_matches) < 2 or len(demand_matches) < 2 or len(cost_matches) < 4:
        return None
    supplies = {f"P{i}": float(value) for i, value in supply_matches}
    demands = {f"C{j}": float(value) for j, value in demand_matches}
    costs: dict[str, dict[str, float]] = {}
    for i, j, value in cost_matches:
        costs.setdefault(f"P{i}", {})[f"C{j}"] = float(value)
    return {
        "context": {
            "title": "Power transmission transportation problem",
            "domain": "energy",
            "decision_maker": "system operator",
            "objective_direction": "minimize",
            "unit": "cost",
            "description": text[:5000],
        },
        "problem_type": "transportation",
        "graph": {"supplies": supplies, "demands": demands, "costs": costs},
        "assumptions": [
            "Total supply equals total demand.",
            "Transmission cost is linear per MW.",
            "No line capacity constraints beyond plant supply and city demand.",
        ],
    }


def parse_resource_allocation_dp_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "quy hoạch động" not in lowered and "dynamic" not in lowered:
        return None
    total_match = re.search(r"(\d+(?:\.\d+)?)\s*tấn", text, flags=re.I)
    profit_rows = []
    for line in text.splitlines():
        normalized = line.lower()
        if "lợi nhuận" not in normalized and "loi nhuan" not in normalized:
            continue
        if "sản xuất / lợi nhuận" in normalized or "san xuat / loi nhuan" in normalized:
            continue
        if "sản xuất (tấn)" in normalized or "san xuat (tan)" in normalized:
            continue
        numbers = [float(value) for value in re.findall(r"(?<!\^)(?<!\d)(\d+(?:\.\d+)?)", line)]
        if len(numbers) >= 4:
            profit_rows.append(numbers[-4:])
    if len(profit_rows) > 1 and profit_rows[0] == [0.0, 1.0, 2.0, 3.0]:
        profit_rows = profit_rows[1:]
    if not total_match or len(profit_rows) < 2:
        return None
    total_resource = int(float(total_match.group(1)))
    return {
        "context": {
            "title": "Steel production dynamic programming problem",
            "domain": "manufacturing",
            "decision_maker": "production planner",
            "objective_direction": "maximize",
            "unit": "10^5 USD",
            "description": text[:5000],
        },
        "problem_type": "dynamic_programming",
        "resource_allocation": {
            "resource_name": "tons",
            "total_resource": total_resource,
            "stage_returns": profit_rows,
        },
        "assumptions": [
            "Tổng sản lượng 7 tấn phải được phân bổ hết trong 3 ngày.",
            "Mỗi ngày có thể sản xuất 0-3 tấn theo bảng lợi nhuận.",
            "Lợi nhuận từng ngày độc lập và cộng được.",
        ],
    }


def parse_shortest_path_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "dijkstra" not in lowered and "đường đi ngắn" not in lowered and "shortest path" not in lowered:
        return None

    cleaned_text = re.sub(r"[*_`]", "", lowered)
    source_match = re.search(r"(?:từ|from)\s+(?:(?:đỉnh|điểm|node|vertex)\s+)?([a-z0-9]+)", cleaned_text)
    target_match = re.search(r"(?:đến|to)\s+(?:(?:đỉnh|điểm|node|vertex)\s+)?([a-z0-9]+)", cleaned_text)
    
    source = None
    if source_match:
        source = source_match.group(1).upper() if source_match.group(1).isalpha() else source_match.group(1)
        
    target = None
    if target_match:
        target = target_match.group(1).upper() if target_match.group(1).isalpha() else target_match.group(1)

    is_undirected = "vô hướng" in lowered or "undirected" in lowered

    edges = []
    for line in text.splitlines():
        if "---" in line:
            continue
        m = re.search(r"(?:\|\s*)?([a-zA-Z0-9]+)\s*(?:-|->|—)\s*([a-zA-Z0-9]+)\s*(?:\||:)\s*(\d+(?:\.\d+)?)\s*(?:\|)?", line)
        if m:
            u, v, w = m.groups()
            u_name = u.upper() if u.isalpha() else u
            v_name = v.upper() if v.isalpha() else v
            directed = False if is_undirected else ("->" in line)
            edges.append({"from": u_name, "to": v_name, "cost": float(w), "directed": directed})

    if not source or not target or not edges:
        return None

    return {
        "context": {
            "title": "Shortest Path Problem",
            "domain": "network",
            "decision_maker": "planner",
            "objective_direction": "minimize",
            "unit": "distance",
            "description": text[:3000],
        },
        "problem_type": "shortest_path",
        "graph": {
            "source": source,
            "target": target,
            "edges": edges,
        },
        "assumptions": [
            "Chi phí các cạnh lấy theo trọng số cho trước.",
            "Thuật toán tìm đường đi ngắn nhất (ví dụ: Dijkstra)."
        ],
    }


# ── LLM-based General LP Parser ──────────────────────────────────────────


LP_SYSTEM_PROMPT = """You are a mathematical modeler. Extract the LP from the problem text.
Return ONLY valid JSON with this EXACT structure (no markdown, no explanation):
{
  "title": "short title",
  "sense": "maximize" or "minimize",
  "n_index_groups": [{"name": "group name", "items": ["item1", "item2"]}],
  "variable_description": "what x_ij represents",
  "variables": [{"name": "x11", "group_i": 0, "group_j": 0}],
  "objective_coefficients": [510, 510, 510, 680, 680, 680],
  "constraints_leq": [
    {"name": "constraint name", "coefficients": [1,0,0,1,0,0], "rhs": 100}
  ],
  "constraints_eq": [
    {"name": "balance", "coefficients": [15,-9,0,15,-9,0], "rhs": 0}
  ],
  "assumptions": ["assumption 1"]
}
Variables must be listed in order: x11,x12,...,x1m,x21,x22,...,xnm.
ALL coefficient arrays must have length = number of variables.
Do NOT include any text outside the JSON."""


async def parse_lp_with_llm(text: str, model: str = "qwen3:8b") -> dict[str, Any] | None:
    """Use Ollama LLM to extract LP structure from arbitrary text."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": LP_SYSTEM_PROMPT},
                        {"role": "user", "content": text[:4000]},
                    ],
                    "options": {"temperature": 0.1, "num_predict": 3000},
                },
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
    except Exception:
        return None

    # Extract JSON from response
    # Remove think tags if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    # Validate and convert to raw LP format
    obj_c = data.get("objective_coefficients", [])
    if not obj_c:
        return None

    n = len(obj_c)
    var_names = [v["name"] for v in data.get("variables", [])]
    if len(var_names) != n:
        var_names = [f"x{i+1}" for i in range(n)]

    A_ub, b_ub, ub_names = [], [], []
    for con in data.get("constraints_leq", []):
        coeffs = con.get("coefficients", [])
        if len(coeffs) == n:
            A_ub.append([float(c) for c in coeffs])
            b_ub.append(float(con["rhs"]))
            ub_names.append(con.get("name", ""))

    A_eq, b_eq, eq_names = [], [], []
    for con in data.get("constraints_eq", []):
        coeffs = con.get("coefficients", [])
        if len(coeffs) == n:
            A_eq.append([float(c) for c in coeffs])
            b_eq.append(float(con["rhs"]))
            eq_names.append(con.get("name", ""))

    # Build formulation text
    sense = data.get("sense", "maximize")
    formulation = f"{sense} Z = c·x\n"
    formulation += f"Biến: {data.get('variable_description', ', '.join(var_names))}\n"
    formulation += f"Số biến: {n}, Bất đẳng thức: {len(A_ub)}, Đẳng thức: {len(A_eq)}"

    steps = [
        f"1. LLM nhận diện bài toán: {data.get('title', 'LP')}",
        f"2. Trích xuất {n} biến quyết định",
        f"3. Xây dựng {len(A_ub)} ràng buộc ≤ và {len(A_eq)} ràng buộc =",
        f"4. Giải bằng scipy HiGHS solver",
    ]

    return {
        "context": {
            "title": data.get("title", "Linear Programming"),
            "domain": "optimization",
            "decision_maker": "analyst",
            "objective_direction": sense,
            "unit": "$",
            "description": text[:3000],
        },
        "problem_type": "linear_programming",
        "c": [float(c) for c in obj_c],
        "sense": sense,
        "A_ub": A_ub if A_ub else None,
        "b_ub": b_ub if b_ub else None,
        "A_eq": A_eq if A_eq else None,
        "b_eq": b_eq if b_eq else None,
        "bounds": [[0, None]] * n,
        "variable_names": var_names,
        "constraint_names_ub": ub_names,
        "constraint_names_eq": eq_names,
        "formulation": formulation,
        "steps": steps,
        "assumptions": data.get("assumptions", ["LP solved with scipy HiGHS."]),
    }


# ── Main Entry Point ─────────────────────────────────────────────────────


def solve_text_problem(text: str) -> dict[str, Any]:
    """Solve text problem: try specific parsers first, then cargo LP, then return pending for LLM."""
    parsed = (
        parse_probability_tree_text(text)
        or parse_resource_allocation_dp_text(text)
        or parse_power_transportation_text(text)
        or parse_cargo_lp_text(text)
        or parse_shortest_path_text(text)
    )
    if parsed:
        # For raw matrix LP (cargo etc), solve directly via LP solver
        if "c" in parsed:
            from .linear_programming import solve_lp
            result = solve_lp(parsed)
            return {
                "status": "solved",
                "problem": parsed,
                "solved": {
                    "problem_type": "linear_programming",
                    "result": result,
                    "recommendation_explanation": result.get("recommendation", ""),
                },
            }
        solved = solve_problem(parsed)
        return {"status": "solved", "problem": parsed, "solved": solved}

    # Return pending — frontend will call async LLM endpoint
    return {
        "status": "pending_llm",
        "message": "Đang phân tích bài toán bằng AI...",
        "text": text,
    }


async def solve_text_problem_async(text: str, model: str = "qwen3:8b") -> dict[str, Any]:
    """Async version: tries specific parsers, then LLM-based extraction."""
    # Try sync parsers first
    sync_result = solve_text_problem(text)
    if sync_result["status"] == "solved":
        return sync_result

    # Try LLM-based general LP parser
    parsed = await parse_lp_with_llm(text, model=model)
    if parsed and "c" in parsed:
        from .linear_programming import solve_lp
        result = solve_lp(parsed)
        return {
            "status": "solved",
            "problem": parsed,
            "solved": {
                "problem_type": "linear_programming",
                "result": result,
                "recommendation_explanation": result.get("recommendation", ""),
            },
        }

    return {
        "status": "needs_clarification",
        "message": "Không thể trích xuất mô hình toán học từ bài toán. Hãy cung cấp rõ: biến quyết định, hàm mục tiêu, và các ràng buộc.",
        "questions": [
            "Biến quyết định là gì? (ví dụ: x1 = sản lượng sản phẩm A)",
            "Hàm mục tiêu: maximize hay minimize cái gì?",
            "Liệt kê các ràng buộc với hệ số cụ thể.",
        ],
    }

