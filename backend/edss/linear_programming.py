"""Linear Programming Solver — scipy.optimize.linprog + pure-Python fallback.

Uses HiGHS solver via scipy for production-quality LP. Falls back to
vertex enumeration for small problems when scipy is unavailable.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any

# Try scipy first; fall back to pure-Python vertex enumeration
try:
    from scipy.optimize import linprog as scipy_linprog  # type: ignore[import-untyped]

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


EPS = 1e-8


def solve_lp(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve an LP problem. Prefers scipy HiGHS; falls back to vertex enum."""
    # Check for raw matrix form first
    if "c" in problem and ("A_ub" in problem or "A_eq" in problem):
        return solve_lp_raw(problem)

    variables = problem.get("variables", [])
    if not variables:
        raise ValueError("LP requires decision variables.")
    names = [item["name"] for item in variables]

    if HAS_SCIPY:
        return _solve_with_scipy(problem, names, variables)
    return _solve_vertex_enum(problem, names, variables)


def solve_lp_raw(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve LP from raw scipy matrix format.

    Expected keys:
        c: list[float] — objective coefficients
        sense: "maximize" | "minimize"
        A_ub, b_ub: inequality constraints (A_ub @ x <= b_ub)
        A_eq, b_eq: equality constraints (A_eq @ x = b_eq)
        bounds: list of [lo, hi] or None
        variable_names: list[str] — names for display
        constraint_names_ub: list[str] — names for ≤ constraints
        constraint_names_eq: list[str] — names for = constraints
        formulation: str — math model text for display
        steps: list[str] — solution steps for display
    """
    if not HAS_SCIPY:
        raise ValueError("scipy required for raw matrix LP.")

    c_raw = [float(v) for v in problem["c"]]
    sense = problem.get("sense", "maximize")
    n = len(c_raw)

    c_min = [-v for v in c_raw] if sense == "maximize" else c_raw[:]

    A_ub = problem.get("A_ub")
    b_ub = problem.get("b_ub")
    A_eq = problem.get("A_eq")
    b_eq = problem.get("b_eq")

    raw_bounds = problem.get("bounds")
    if raw_bounds:
        bounds = [(b[0] if b[0] is not None else 0, b[1] if len(b) > 1 and b[1] is not None else None) for b in raw_bounds]
    else:
        bounds = [(0, None)] * n

    result = scipy_linprog(
        c_min,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        status = _linprog_failure_status(result)
        return {
            "status": status,
            "message": result.message,
            "solver": "scipy_highs",
            "formulation": problem.get("formulation", ""),
            "recommendation": _failure_recommendation(status),
            "markdown_report": _build_lp_failure_markdown(problem, status, result.message),
        }

    x = result.x
    obj_value = float(-result.fun if sense == "maximize" else result.fun)
    var_names = problem.get("variable_names", [f"x{i}" for i in range(n)])

    # Build solution dict (only non-zero)
    solution = {}
    for i, name in enumerate(var_names):
        if abs(x[i]) > 1e-6:
            solution[name] = round(x[i], 6)

    # Constraint analysis
    binding_ub = []
    slacks_ub = {}
    shadow_prices = {}
    ub_names = problem.get("constraint_names_ub", [f"ub_{i}" for i in range(len(A_ub or []))])
    if A_ub and b_ub:
        for idx, (row, rhs) in enumerate(zip(A_ub, b_ub)):
            lhs = sum(row[j] * x[j] for j in range(n))
            slack = rhs - lhs
            cname = ub_names[idx] if idx < len(ub_names) else f"ub_{idx}"
            slacks_ub[cname] = round(slack, 6)
            if abs(slack) <= 1e-5:
                binding_ub.append(cname)
            # Try to get shadow prices
            if hasattr(result, "ineq_lin") and result.ineq_lin is not None:
                try:
                    dual = float(result.ineq_lin.marginals[idx])
                    shadow_prices[cname] = round(-dual if sense == "maximize" else dual, 6)
                except (IndexError, AttributeError, TypeError):
                    pass

    eq_names = problem.get("constraint_names_eq", [f"eq_{i}" for i in range(len(A_eq or []))])
    if A_eq:
        for idx, cname in enumerate(eq_names):
            binding_ub.append(cname)

    # Build detailed markdown report
    md = _build_lp_markdown(
        problem, var_names, x, obj_value, sense,
        solution, slacks_ub, binding_ub, shadow_prices,
        A_ub, b_ub, ub_names, A_eq, b_eq, eq_names, c_raw, n,
    )

    return {
        "status": "optimal",
        "solver": "scipy_highs",
        "objective_value": round(obj_value, 6),
        "solution": solution,
        "all_values": {var_names[i]: round(x[i], 6) for i in range(n)},
        "binding_constraints": binding_ub,
        "slacks": slacks_ub,
        "shadow_prices": shadow_prices,
        "formulation": problem.get("formulation", ""),
        "steps": problem.get("steps", []),
        "summary_tables": problem.get("summary_tables", {}),
        "assumptions": problem.get("assumptions", [
            "LP solved with scipy HiGHS solver.",
        ]),
        "recommendation": f"Nghiệm tối ưu: Z* = {round(obj_value, 2)}",
        "markdown_report": md,
    }


def _linprog_failure_status(result: Any) -> str:
    status_code = getattr(result, "status", None)
    message = (getattr(result, "message", "") or "").lower()
    if status_code == 2 or "infeasible" in message:
        return "infeasible"
    if status_code == 3 or "unbounded" in message:
        return "unbounded"
    if status_code == 4:
        return "solver_error"
    return "failed"


def _failure_recommendation(status: str) -> str:
    if status == "infeasible":
        return "Không có nghiệm khả thi; cần kiểm tra các ràng buộc mâu thuẫn hoặc RHS quá chặt."
    if status == "unbounded":
        return "Bài toán không bị chặn; cần thêm ràng buộc giới hạn biến/tài nguyên để có nghiệm tối ưu hữu hạn."
    return "Solver không tìm được nghiệm; cần kiểm tra mô hình và dữ liệu đầu vào."


def _build_lp_failure_markdown(problem: dict[str, Any], status: str, message: str | None) -> str:
    ctx = problem.get("context", {})
    title = ctx.get("title", "Linear Programming Solution")
    formulation = problem.get("formulation", "")
    lines = [f"# {title}", "", f"**Solver**: scipy HiGHS · **Status**: {status}", ""]
    lines.append("## 1. Mô hình đã parse")
    lines.append("")
    if formulation:
        lines.append("```text")
        lines.append(formulation)
        lines.append("```")
        lines.append("")
    lines.append("## 2. Kết luận solver")
    lines.append("")
    if status == "infeasible":
        lines.append("Bài toán **không có nghiệm khả thi**. Nghĩa là không tồn tại bộ giá trị biến nào thỏa tất cả ràng buộc cùng lúc.")
        lines.append("")
        lines.append("Cách kiểm tra tiếp theo:")
        lines.append("- Xem có ràng buộc mâu thuẫn như `x <= a` và `x >= b` với `b > a` không.")
        lines.append("- Kiểm tra dấu `<=`, `>=`, `=` có bị nhập sai không.")
        lines.append("- Kiểm tra đơn vị đo/RHS của từng ràng buộc.")
    elif status == "unbounded":
        lines.append("Bài toán **không bị chặn**. Hàm mục tiêu có thể cải thiện vô hạn theo một hướng khả thi, nên không tồn tại nghiệm tối ưu hữu hạn.")
        lines.append("")
        lines.append("Cách khắc phục:")
        lines.append("- Thêm giới hạn tài nguyên/công suất/ngân sách cho các biến đang tăng vô hạn.")
        lines.append("- Kiểm tra có thiếu ràng buộc upper bound không.")
        lines.append("- Kiểm tra objective direction maximize/minimize có đúng không.")
    else:
        lines.append("Solver không trả về nghiệm tối ưu. Cần kiểm tra mô hình đầu vào hoặc thử solver khác.")
    if message:
        lines.append("")
        lines.append("## 3. Thông báo kỹ thuật")
        lines.append("")
        lines.append(f"```text\n{message}\n```")
    return "\n".join(lines)


def _build_lp_markdown(
    problem: dict, var_names: list, x, obj_value: float, sense: str,
    solution: dict, slacks: dict, binding: list, shadow_prices: dict,
    A_ub, b_ub, ub_names, A_eq, b_eq, eq_names, c_raw, n: int,
) -> str:
    """Generate a comprehensive markdown solution report."""
    ctx = problem.get("context", {})
    title = ctx.get("title", "Linear Programming Solution")
    tables = problem.get("summary_tables", {})
    formulation = problem.get("formulation", "")
    steps = problem.get("steps", [])
    assumptions = problem.get("assumptions", [])

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Solver**: scipy HiGHS · **Status**: Optimal")
    lines.append("")

    # Section 1: Problem Summary
    lines.append("## 1. Tóm tắt bài toán")
    lines.append("")

    # Input data tables
    for tbl_name, rows in tables.items():
        if not isinstance(rows, list) or not rows:
            continue
        if not isinstance(rows[0], dict):
            continue
        keys = list(rows[0].keys())
        lines.append(f"**{tbl_name.replace('_', ' ').title()}:**")
        lines.append("")
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join("---:" for _ in keys) + " |")
        for row in rows:
            vals = [str(row.get(k, "")) for k in keys]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # Section 2: Decision Variables
    lines.append("## 2. Biến quyết định")
    lines.append("")
    lines.append(f"Tổng cộng **{n} biến**: " + ", ".join(var_names))
    lines.append("")

    # Section 3: Mathematical Formulation
    lines.append("## 3. Mô hình toán học")
    lines.append("")
    if formulation:
        lines.append("```")
        lines.append(formulation)
        lines.append("```")
        lines.append("")

    if n == 2 and A_ub and b_ub and not A_eq:
        lines.extend(_build_two_variable_geometric_solution(var_names, A_ub, b_ub, ub_names, c_raw))
    else:
        lines.extend(_build_general_lp_solution_notes(var_names, A_ub, b_ub, ub_names, A_eq, b_eq, eq_names, c_raw, sense))

    # Section 4: Steps
    if steps:
        lines.append("## 4. Các bước giải")
        lines.append("")
        for s in steps:
            lines.append(f"- {s}")
        lines.append("")

    # Section 5: OPTIMAL SOLUTION
    lines.append("## 5. ✅ Nghiệm tối ưu")
    lines.append("")
    lines.append(f"### **Z* = ${obj_value:,.2f}**")
    lines.append("")

    # Solution table (non-zero only)
    if solution:
        lines.append("| Biến | Giá trị (tấn) |")
        lines.append("|---|---:|")
        for name, val in solution.items():
            lines.append(f"| {name} | {val:.4f} |")
        lines.append("")

    # Section 6: Detailed Calculation - verify Z*
    lines.append("## 6. 📐 Kiểm tra phép tính")
    lines.append("")
    lines.append("### Tính Z* từ nghiệm:")
    lines.append("")
    obj_parts = []
    total = 0.0
    for name, val in solution.items():
        idx = var_names.index(name) if name in var_names else -1
        if idx >= 0:
            coef = c_raw[idx]
            contrib = coef * val
            total += contrib
            obj_parts.append(f"- {name}: {coef} × {val:.4f} = **{contrib:.2f}**")
    for part in obj_parts:
        lines.append(part)
    lines.append(f"- **Tổng Z* = {total:.2f}** ✓")
    lines.append("")

    # Constraint verification
    lines.append("### Kiểm tra ràng buộc:")
    lines.append("")
    if A_ub and b_ub:
        lines.append("| Ràng buộc | LHS | Dấu | RHS | Slack | Status |")
        lines.append("|---|---:|---|---:|---:|---|")
        for idx, (row, rhs) in enumerate(zip(A_ub, b_ub)):
            lhs = sum(row[j] * x[j] for j in range(n))
            slack = rhs - lhs
            cname = ub_names[idx] if idx < len(ub_names) else f"ub_{idx}"
            status = "🔒 Binding" if abs(slack) < 1e-5 else "✓ OK"
            lines.append(f"| {cname} | {lhs:.4f} | ≤ | {rhs:.1f} | {slack:.4f} | {status} |")
        lines.append("")

    if A_eq and b_eq:
        lines.append("| Ràng buộc (=) | LHS | = | RHS | Status |")
        lines.append("|---|---:|---|---:|---|")
        for idx, (row, rhs) in enumerate(zip(A_eq, b_eq)):
            lhs = sum(row[j] * x[j] for j in range(n))
            diff = abs(lhs - rhs)
            cname = eq_names[idx] if idx < len(eq_names) else f"eq_{idx}"
            status = "✓ Thỏa" if diff < 1e-5 else f"✗ Sai ({diff:.6f})"
            lines.append(f"| {cname} | {lhs:.4f} | = | {rhs:.1f} | {status} |")
        lines.append("")

    # Section 7: Binding constraints summary
    if binding or slacks:
        lines.append("## 7. 🔒 Phân tích Ràng buộc (Constraint Analysis)")
        lines.append("")
        lines.append("> **💡 Giải thích thuật ngữ:**")
        lines.append("> - **Ràng buộc chặt (Binding):** Là những nguồn tài nguyên đã bị sử dụng cạn kiệt (hoặc đạt mức tối thiểu bắt buộc) đến tận giới hạn (Slack = 0). Đây là các **nút thắt cổ chai** của hệ thống, mọi sự nới lỏng ở đây sẽ trực tiếp cải thiện lợi nhuận/chi phí.")
        lines.append("> - **Ràng buộc không chặt (Non-binding):** Là những nguồn tài nguyên vẫn còn dư thừa (Slack > 0). Việc nạp thêm tài nguyên này lúc này cũng vô ích vì nó không phải là nút thắt.")
        lines.append("")

        if binding:
            lines.append("### 🔴 Các Ràng buộc chặt (Nút thắt):")
            for b in binding:
                lines.append(f"- **{b}** (đã cạn kiệt)")
            lines.append("")

        if slacks:
            slack_list = [(k, v) for k, v in slacks.items() if v > 1e-5]
            if slack_list:
                lines.append("### 🟢 Các Ràng buộc không chặt (Còn dư):")
                for name, val in slack_list:
                    lines.append(f"- **{name}**: dư thừa **{val:.4f}** đơn vị")
                lines.append("")

    # Section 8: Assumptions
    if assumptions:
        lines.append("## 8. ⚠️ Giả định")
        lines.append("")
        for a in assumptions:
            if a:
                lines.append(f"- {a}")
        lines.append("")

    return "\n".join(lines)


def _build_general_lp_solution_notes(
    var_names: list[str],
    a_ub,
    b_ub,
    ub_names: list[str],
    a_eq,
    b_eq,
    eq_names: list[str],
    c_raw: list[float],
    sense: str,
) -> list[str]:
    lines: list[str] = []
    lines.append("## Phần giải tổng quát cho LP nhiều biến")
    lines.append("")
    lines.append("Bài toán được đưa về dạng ma trận để solver HiGHS giải bằng các biến không âm.")
    lines.append("")
    lines.append("### 1. Dạng ma trận")
    lines.append("")
    lines.append(f"- Objective direction: **{sense}**")
    lines.append(f"- Vector biến: `x = ({', '.join(var_names)})`")
    lines.append(f"- Vector hệ số mục tiêu: `c = ({', '.join(_fmt_fraction(v) for v in c_raw)})`")
    lines.append("")
    if a_ub and b_ub:
        lines.append("Ràng buộc bất đẳng thức `A_ub x <= b_ub`:")
        lines.append("")
        lines.append("| Tên | Hệ số | RHS |")
        lines.append("|---|---|---:|")
        for idx, (row, rhs) in enumerate(zip(a_ub, b_ub)):
            name = ub_names[idx] if idx < len(ub_names) and ub_names[idx] else f"ub_{idx + 1}"
            lines.append(f"| {name} | ({', '.join(_fmt_fraction(v) for v in row)}) | {_fmt_fraction(rhs)} |")
        lines.append("")
    if a_eq and b_eq:
        lines.append("Ràng buộc đẳng thức `A_eq x = b_eq`:")
        lines.append("")
        lines.append("| Tên | Hệ số | RHS |")
        lines.append("|---|---|---:|")
        for idx, (row, rhs) in enumerate(zip(a_eq, b_eq)):
            name = eq_names[idx] if idx < len(eq_names) and eq_names[idx] else f"eq_{idx + 1}"
            lines.append(f"| {name} | ({', '.join(_fmt_fraction(v) for v in row)}) | {_fmt_fraction(rhs)} |")
        lines.append("")
    lines.append("### 2. Diễn giải")
    lines.append("")
    lines.append("- Ràng buộc `=` luôn là ràng buộc chặt nếu nghiệm khả thi.")
    lines.append("- Với `<=`, slack = RHS - LHS; slack bằng 0 nghĩa là binding.")
    lines.append("- Với bài toán maximize, shadow price dương cho biết objective tăng bao nhiêu khi nới RHS thêm 1 đơn vị, trong vùng nhạy cục bộ.")
    lines.append("")
    return lines


def _fmt_fraction(value: float, max_denominator: int = 1000) -> str:
    if abs(value) < 1e-10:
        return "0"
    fraction = Fraction(float(value)).limit_denominator(max_denominator)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _fmt_fraction_decimal(value: float) -> str:
    return f"{_fmt_fraction(value)} ≈ {value:.4f}"


def _fmt_point(point: tuple[float, float]) -> str:
    return f"({_fmt_fraction(point[0])}, {_fmt_fraction(point[1])})"


def _line_expr(row: list[float], names: list[str]) -> str:
    parts = []
    for coef, name in zip(row, names):
        if abs(coef) < 1e-10:
            continue
        sign = "-" if coef < 0 else "+"
        magnitude = abs(coef)
        term = name if abs(magnitude - 1.0) < 1e-10 else f"{_fmt_fraction(magnitude)}{name}"
        parts.append((sign, term))
    if not parts:
        return "0"
    first_sign, first_term = parts[0]
    expr = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in parts[1:]:
        expr += f" {sign} {term}"
    return expr


def _intersect_two_lines(row_a: list[float], rhs_a: float, row_b: list[float], rhs_b: float) -> tuple[float, float] | None:
    det = row_a[0] * row_b[1] - row_a[1] * row_b[0]
    if abs(det) < 1e-10:
        return None
    x_val = (rhs_a * row_b[1] - row_a[1] * rhs_b) / det
    y_val = (row_a[0] * rhs_b - rhs_a * row_b[0]) / det
    return (x_val, y_val)


def _is_feasible_2d(point: tuple[float, float], a_ub: list[list[float]], b_ub: list[float]) -> bool:
    if point[0] < -1e-8 or point[1] < -1e-8:
        return False
    return all(row[0] * point[0] + row[1] * point[1] <= rhs + 1e-8 for row, rhs in zip(a_ub, b_ub))


def _build_two_variable_geometric_solution(
    var_names: list[str],
    a_ub: list[list[float]],
    b_ub: list[float],
    ub_names: list[str],
    c_raw: list[float],
) -> list[str]:
    lines: list[str] = []
    x_name, y_name = var_names[0], var_names[1]

    lines.append("## Phần giải chi tiết bằng phương pháp hình học")
    lines.append("")
    lines.append("Vì bài toán có 2 biến quyết định và các ràng buộc tuyến tính, ta có thể tìm nghiệm tối ưu bằng cách xét các đỉnh của miền nghiệm.")
    lines.append("")

    lines.append("### 1. Xác định các đường biên")
    lines.append("")
    for idx, (row, rhs) in enumerate(zip(a_ub, b_ub)):
        cname = ub_names[idx] if idx < len(ub_names) and ub_names[idx] else f"c{idx + 1}"
        expr = _line_expr(row, var_names)
        lines.append(f"Từ ràng buộc **{cname}**:")
        lines.append("")
        lines.append(f"```text\n{expr} = {_fmt_fraction(rhs)}\n```")
        if abs(row[0]) > 1e-10:
            x_intercept = rhs / row[0]
            lines.append(f"- Nếu {y_name} = 0: {expr} = {_fmt_fraction(rhs)} ⇒ {x_name} = {_fmt_fraction_decimal(x_intercept)}")
        else:
            x_intercept = None
            lines.append(f"- Đường biên song song trục {x_name}, không có giao điểm {y_name}=0 hữu hạn.")
        if abs(row[1]) > 1e-10:
            y_intercept = rhs / row[1]
            lines.append(f"- Nếu {x_name} = 0: {expr} = {_fmt_fraction(rhs)} ⇒ {y_name} = {_fmt_fraction_decimal(y_intercept)}")
        else:
            y_intercept = None
            lines.append(f"- Đường biên song song trục {y_name}, không có giao điểm {x_name}=0 hữu hạn.")
        if x_intercept is not None and y_intercept is not None:
            lines.append(f"- Đường biên đi qua `{(_fmt_point((x_intercept, 0)))}` và `{(_fmt_point((0, y_intercept)))}`.")
        lines.append("")

    candidates: list[tuple[float, float]] = [(0.0, 0.0)]
    for row, rhs in zip(a_ub, b_ub):
        if abs(row[0]) > 1e-10:
            candidates.append((rhs / row[0], 0.0))
        if abs(row[1]) > 1e-10:
            candidates.append((0.0, rhs / row[1]))
    for (row_a, rhs_a), (row_b, rhs_b) in combinations(list(zip(a_ub, b_ub)), 2):
        point = _intersect_two_lines(row_a, rhs_a, row_b, rhs_b)
        if point is not None:
            candidates.append(point)

    feasible_vertices: list[tuple[float, float]] = []
    for point in candidates:
        if not _is_feasible_2d(point, a_ub, b_ub):
            continue
        if not any(abs(point[0] - existing[0]) < 1e-7 and abs(point[1] - existing[1]) < 1e-7 for existing in feasible_vertices):
            feasible_vertices.append(point)
    feasible_vertices.sort(key=lambda item: (item[0], item[1]))

    lines.append("### 2. Tìm các đỉnh của miền nghiệm")
    lines.append("")
    lines.append(f"Vì {x_name} ≥ 0, {y_name} ≥ 0, miền nghiệm nằm trong góc phần tư thứ nhất.")
    lines.append("")
    lines.append("Các đỉnh khả thi là:")
    lines.append("")
    for point in feasible_vertices:
        lines.append(f"- `{_fmt_point(point)}`")
    lines.append("")

    if len(a_ub) >= 2:
        point = _intersect_two_lines(a_ub[0], b_ub[0], a_ub[1], b_ub[1])
        if point is not None:
            lines.append("### 3. Tính giao điểm của hai đường ràng buộc chính")
            lines.append("")
            lines.append("Giải hệ:")
            lines.append("")
            lines.append("```text")
            lines.append(f"{_line_expr(a_ub[0], var_names)} = {_fmt_fraction(b_ub[0])}")
            lines.append(f"{_line_expr(a_ub[1], var_names)} = {_fmt_fraction(b_ub[1])}")
            lines.append("```")
            lines.append("")
            lines.append(f"Kết quả giao điểm: `{_fmt_point(point)}` = `({_fmt_fraction_decimal(point[0])}, {_fmt_fraction_decimal(point[1])})`.")
            lines.append("")

    lines.append("### 4. Tính giá trị hàm mục tiêu tại các đỉnh")
    lines.append("")
    lines.append(f"Hàm mục tiêu: `Z = {_line_expr(c_raw, var_names)}`")
    lines.append("")
    lines.append("| Đỉnh | Giá trị Z |")
    lines.append("|---|---:|")
    best_point = None
    best_value = float("-inf")
    for point in feasible_vertices:
        value = c_raw[0] * point[0] + c_raw[1] * point[1]
        if value > best_value:
            best_value = value
            best_point = point
        lines.append(f"| {_fmt_point(point)} | {_fmt_fraction(value)} ≈ {value:.4f} |")
    lines.append("")
    if best_point is not None:
        lines.append("### 5. Đáp án cuối cùng theo phương pháp hình học")
        lines.append("")
        lines.append(f"- `{x_name}* = {_fmt_fraction(best_point[0])}`")
        lines.append(f"- `{y_name}* = {_fmt_fraction(best_point[1])}`")
        lines.append(f"- `Zmax = {_fmt_fraction(best_value)} ≈ {best_value:.4f}`")
        lines.append("")
    return lines


# ── scipy HiGHS Solver ────────────────────────────────────────────────────


def _solve_with_scipy(
    problem: dict[str, Any],
    names: list[str],
    variables: list[dict[str, Any]],
    _skip_shadow: bool = False,
) -> dict[str, Any]:
    n = len(names)
    obj = problem.get("objective") or {}
    sense = obj.get("sense", "maximize")
    constant = float(obj.get("constant", 0))

    # Objective coefficients (linprog minimizes, so negate for maximize)
    c = [float(obj.get("coefficients", {}).get(name, 0)) for name in names]
    if sense == "maximize":
        c_min = [-ci for ci in c]
    else:
        c_min = c[:]

    # Constraints: separate <= and = types
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    constraint_names: list[str] = []
    constraint_types: list[str] = []

    for con in problem.get("constraints", []):
        coeffs = [float(con.get("coefficients", {}).get(name, 0)) for name in names]
        rhs = float(con.get("rhs", 0))
        op = con.get("operator", "<=")
        cname = con.get("name", "")

        if op == "<=":
            a_ub.append(coeffs)
            b_ub.append(rhs)
        elif op == ">=":
            a_ub.append([-ci for ci in coeffs])
            b_ub.append(-rhs)
        elif op == "=":
            a_eq.append(coeffs)
            b_eq.append(rhs)
        constraint_names.append(cname)
        constraint_types.append(op)

    # Variable bounds
    bounds = []
    for var in variables:
        lb = float(var.get("lower_bound", 0))
        ub = var.get("upper_bound")
        bounds.append((lb, float(ub) if ub is not None else None))

    # Solve
    result = scipy_linprog(
        c_min,
        A_ub=a_ub if a_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=a_eq if a_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        status = "infeasible" if "infeasible" in result.message.lower() else "unbounded" if "unbounded" in result.message.lower() else "failed"
        return {"status": status, "message": result.message, "missing_data": []}

    # Extract solution
    x = list(result.x)
    obj_value = -result.fun + constant if sense == "maximize" else result.fun + constant

    # Compute slacks and binding constraints
    slacks: dict[str, float] = {}
    binding: list[str] = []
    shadow_prices: dict[str, float] = {}

    ub_idx = 0
    eq_idx = 0
    for i, con in enumerate(problem.get("constraints", [])):
        cname = con.get("name", f"c{i}")
        coeffs = [float(con.get("coefficients", {}).get(name, 0)) for name in names]
        rhs = float(con.get("rhs", 0))
        op = con.get("operator", "<=")
        lhs = sum(a * xi for a, xi in zip(coeffs, x))

        if op == "<=":
            slack = rhs - lhs
            slacks[cname] = round(slack, 6)
            if abs(slack) <= 1e-5:
                binding.append(cname)
            # Dual from ineq_marginals
            if hasattr(result, "ineq_lin") and result.ineq_lin is not None:
                try:
                    dual = float(result.ineq_lin.marginals[ub_idx])
                    shadow_prices[cname] = round(-dual if sense == "maximize" else dual, 6)
                except (IndexError, AttributeError, TypeError):
                    pass
            ub_idx += 1
        elif op == ">=":
            slack = lhs - rhs
            slacks[cname] = round(slack, 6)
            if abs(slack) <= 1e-5:
                binding.append(cname)
            if hasattr(result, "ineq_lin") and result.ineq_lin is not None:
                try:
                    dual = float(result.ineq_lin.marginals[ub_idx])
                    shadow_prices[cname] = round(dual if sense == "maximize" else -dual, 6)
                except (IndexError, AttributeError, TypeError):
                    pass
            ub_idx += 1
        elif op == "=":
            slacks[cname] = 0.0
            binding.append(cname)
            if hasattr(result, "eqlin") and result.eqlin is not None:
                try:
                    dual = float(result.eqlin.marginals[eq_idx])
                    shadow_prices[cname] = round(-dual if sense == "maximize" else dual, 6)
                except (IndexError, AttributeError, TypeError):
                    pass
            eq_idx += 1

    # Fallback shadow prices via finite-difference if duals unavailable
    if not shadow_prices and not _skip_shadow:
        shadow_prices = _approximate_shadow_prices_scipy(problem, names, variables, obj_value, sense)

    return {
        "status": "optimal",
        "solver": "scipy_highs",
        "objective_value": round(obj_value, 6),
        "solution": {name: round(x[i], 6) for i, name in enumerate(names)},
        "binding_constraints": binding,
        "slacks": slacks,
        "shadow_prices": shadow_prices,
        "assumptions": [
            "LP solved with scipy HiGHS interior-point/simplex solver.",
            "Shadow prices indicate marginal value of relaxing each binding constraint by 1 unit.",
        ],
        "recommendation": f"Nghiệm tối ưu có objective = {round(obj_value, 3)}.",
    }


def _approximate_shadow_prices_scipy(
    problem: dict[str, Any],
    names: list[str],
    variables: list[dict[str, Any]],
    base_value: float,
    sense: str,
    delta: float = 1.0,
) -> dict[str, float]:
    """Finite-difference shadow prices as fallback."""
    import copy
    prices: dict[str, float] = {}
    for i, con in enumerate(problem.get("constraints", [])):
        perturbed = copy.deepcopy(problem)
        perturbed["constraints"][i]["rhs"] = float(perturbed["constraints"][i].get("rhs", 0)) + delta
        try:
            result = _solve_with_scipy(perturbed, names, variables, _skip_shadow=True)
            if result.get("status") == "optimal":
                prices[con.get("name", f"c{i}")] = round(
                    (result["objective_value"] - base_value) / delta, 4
                )
        except Exception:
            prices[con.get("name", f"c{i}")] = 0.0
    return prices


# ── Pure-Python Vertex Enumeration (fallback) ─────────────────────────────


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < EPS:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        a[col] = [value / div for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [a[row][i] - factor * a[col][i] for i in range(n + 1)]
    return [a[row][n] for row in range(n)]


def canonical_constraints(problem: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    constraints = []
    for item in problem.get("constraints", []):
        coeffs = [float(item.get("coefficients", {}).get(name, 0)) for name in names]
        op = item.get("operator", "<=")
        rhs = float(item.get("rhs", 0))
        if op == ">=":
            coeffs = [-value for value in coeffs]
            rhs = -rhs
        constraints.append({**item, "coeff_vector": coeffs, "rhs": rhs, "operator": "<="})
    for index, variable in enumerate(problem.get("variables", [])):
        lb = float(variable.get("lower_bound", 0))
        coeffs = [0.0] * len(names)
        coeffs[index] = -1.0
        constraints.append({"name": f"{names[index]}_lower_bound", "coeff_vector": coeffs, "rhs": -lb, "operator": "<="})
        if variable.get("upper_bound") is not None:
            coeffs = [0.0] * len(names)
            coeffs[index] = 1.0
            constraints.append({"name": f"{names[index]}_upper_bound", "coeff_vector": coeffs, "rhs": float(variable["upper_bound"]), "operator": "<="})
    return constraints


def feasible(point: list[float], constraints: list[dict[str, Any]]) -> bool:
    return all(sum(a * x for a, x in zip(item["coeff_vector"], point)) <= float(item["rhs"]) + 1e-6 for item in constraints)


def objective_value(problem: dict[str, Any], names: list[str], point: list[float]) -> float:
    objective = problem.get("objective") or {}
    coeffs = objective.get("coefficients", {})
    return float(objective.get("constant", 0)) + sum(float(coeffs.get(name, 0)) * point[index] for index, name in enumerate(names))


def _solve_vertex_enum(
    problem: dict[str, Any],
    names: list[str],
    variables: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(names)
    constraints = canonical_constraints(problem, names)
    if len(constraints) < n:
        raise ValueError("LP requires at least as many constraints as variables.")

    candidates = []
    for combo in combinations(constraints, n):
        solution = solve_linear_system([item["coeff_vector"] for item in combo], [float(item["rhs"]) for item in combo])
        if solution is not None and feasible(solution, constraints):
            candidates.append(solution)
    if not candidates:
        return {"status": "infeasible", "message": "No feasible vertex found.", "missing_data": []}

    sense = (problem.get("objective") or {}).get("sense", "maximize")
    reverse = sense == "maximize"
    best = sorted(candidates, key=lambda point: objective_value(problem, names, point), reverse=reverse)[0]
    best_value = objective_value(problem, names, best)
    slacks: dict[str, float] = {}
    binding: list[str] = []
    for item in constraints:
        lhs = sum(a * x for a, x in zip(item["coeff_vector"], best))
        slack = float(item["rhs"]) - lhs
        slacks[item["name"]] = slack
        if abs(slack) <= 1e-5:
            binding.append(item["name"])

    shadow_prices = _approximate_shadow_prices_vertex(problem, names, best_value)
    return {
        "status": "optimal",
        "solver": "pure_python_vertex_lp",
        "objective_value": best_value,
        "solution": {name: best[index] for index, name in enumerate(names)},
        "binding_constraints": binding,
        "slacks": slacks,
        "shadow_prices": shadow_prices,
        "assumptions": [
            "LP solved via vertex enumeration (pure-Python fallback). Install scipy for HiGHS.",
            "Shadow prices are finite-difference approximations.",
        ],
        "recommendation": f"Nghiệm tối ưu có objective = {best_value:.3f}.",
    }


def _approximate_shadow_prices_vertex(problem: dict[str, Any], names: list[str], base_value: float) -> dict[str, float]:
    prices = {}
    for index, constraint in enumerate(problem.get("constraints", [])):
        import copy
        perturbed = copy.deepcopy(problem)
        perturbed["constraints"][index]["rhs"] = float(perturbed["constraints"][index].get("rhs", 0)) + 1.0
        try:
            result = _solve_vertex_enum(perturbed, names, problem.get("variables", []))
            if result.get("status") == "optimal":
                prices[constraint.get("name", f"c{index}")] = result["objective_value"] - base_value
        except Exception:
            prices[constraint.get("name", f"c{index}")] = 0.0
    return prices
