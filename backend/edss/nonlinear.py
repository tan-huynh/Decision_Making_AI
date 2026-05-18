from __future__ import annotations

import math
import random
from itertools import permutations
from typing import Any, Callable
from urllib.parse import quote

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

from .teaching_report import wrap_teaching_report


def solve_nonlinear(problem: dict[str, Any]) -> dict[str, Any]:
    """Numerical nonlinear programming with callable registry or quadratic spec.

    Supported structured objective:
      {"type":"quadratic", "Q":[[...]], "c":[...], "constant":0}
    Constraints use scipy dict-like payloads:
      {"type":"ineq", "coefficients":[...], "rhs": 5} means rhs - a^T x >= 0
      {"type":"eq", "coefficients":[...], "rhs": 5}
    """
    sense = problem.get("sense", "minimize")
    names = problem.get("variable_names", [f"x{i+1}" for i in range(len(problem.get("initial", [])))])
    x0 = np.asarray(problem.get("initial", [0.0] * len(names)), dtype=float)
    objective = _objective(problem.get("objective", {}), sense)
    constraints = [_constraint(item) for item in problem.get("constraints", [])]
    bounds = problem.get("bounds")
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
    status = "optimal_local" if result.success else "failed"
    raw_value = float(_objective(problem.get("objective", {}), "minimize")(result.x))
    objective_value = -raw_value if sense == "maximize" else raw_value
    solution = {name: round(float(value), 8) for name, value in zip(names, result.x)}
    sections = [
        ("Mô hình", "Bài toán NLP được giải bằng SLSQP. Nếu objective/constraints không lồi, nghiệm chỉ được đảm bảo là local optimum."),
        ("KKT conditions", _kkt_notes(problem)),
        ("Kết quả số", f"Solver success = `{result.success}`; message = `{result.message}`; solution = `{solution}`."),
    ]
    return {
        "status": status,
        "solver": "scipy_slsqp",
        "objective_value": round(objective_value, 8),
        "solution": solution,
        "local_optimum_warning": True,
        "message": result.message,
        "markdown_report": wrap_teaching_report("Nonlinear Programming", "NLP / KKT", sections, f"Objective = {objective_value:.6f}; x = {solution}"),
    }


def solve_circle_packing_box(radii: list[float], attempts: int = 120, seed: int = 7) -> dict[str, Any]:
    """Minimize box perimeter for non-overlapping circles in an axis-aligned box.

    Variables are A, B, X_i, Y_i. The model is nonconvex, so this uses deterministic
    multi-start SLSQP and reports the best feasible local optimum found.
    """
    if len(radii) < 2 or any(r <= 0 for r in radii):
        raise ValueError("Circle packing requires at least two positive radii.")

    n = len(radii)
    max_r = max(radii)
    upper = 2 * sum(radii)

    def objective(v: np.ndarray) -> float:
        return float(2 * (v[0] + v[1]))

    constraints = []
    for i, r in enumerate(radii):
        xi = 2 + 2 * i
        yi = xi + 1
        constraints.append({"type": "ineq", "fun": lambda v, xi=xi, r=r: float(v[xi] - r)})
        constraints.append({"type": "ineq", "fun": lambda v, yi=yi, r=r: float(v[yi] - r)})
        constraints.append({"type": "ineq", "fun": lambda v, xi=xi, r=r: float(v[1] - r - v[xi])})
        constraints.append({"type": "ineq", "fun": lambda v, yi=yi, r=r: float(v[0] - r - v[yi])})
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = 2 + 2 * i, 3 + 2 * i
            xj, yj = 2 + 2 * j, 3 + 2 * j
            min_dist = radii[i] + radii[j]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda v, xi=xi, yi=yi, xj=xj, yj=yj, d=min_dist: float(
                        (v[xi] - v[xj]) ** 2 + (v[yi] - v[yj]) ** 2 - d**2
                    ),
                }
            )

    bounds = [(2 * max_r, upper), (2 * max_r, upper)]
    for r in radii:
        bounds.extend([(r, upper - r), (r, upper - r)])

    starts = _circle_packing_starts(radii, attempts, seed)
    best = None
    results = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10, "disp": False},
        )
        feasible = result.success and _circle_solution_feasible(result.x, radii)
        results.append({"success": bool(result.success), "feasible": feasible, "objective": float(objective(result.x)), "message": str(result.message)})
        if feasible and (best is None or objective(result.x) < objective(best.x)):
            best = result

    if best is None:
        return {
            "status": "failed",
            "solver": "scipy_slsqp_multistart",
            "message": "Không tìm được nghiệm feasible cho circle packing NLP.",
            "attempts": len(starts),
            "attempt_summaries": results[:10],
        }

    v = best.x
    circles = [
        {"circle": i + 1, "radius": radii[i], "x": round(float(v[2 + 2 * i]), 6), "y": round(float(v[3 + 2 * i]), 6)}
        for i in range(n)
    ]
    checks = _circle_checks(v, radii)
    tangent_check = _three_circle_tangent_layout_check(radii) if n == 3 else None
    svg = _circle_packing_svg(v, radii)
    report = _circle_packing_report(v, radii, circles, checks, len(starts), svg, tangent_check)
    return {
        "status": "optimal_local",
        "solver": "scipy_slsqp_multistart_circle_packing",
        "objective_value": round(float(objective(v)), 6),
        "local_optimum_warning": True,
        "attempts": len(starts),
        "box": {"A_height": round(float(v[0]), 6), "B_width": round(float(v[1]), 6), "perimeter": round(float(objective(v)), 6)},
        "circles": circles,
        "constraint_checks": checks,
        "tangent_layout_check": tangent_check,
        "svg": svg,
        "svg_data_uri": "data:image/svg+xml;utf8," + quote(svg),
        "markdown_report": report,
    }


def _objective(spec: dict[str, Any], sense: str) -> Callable[[np.ndarray], float]:
    sign = -1.0 if sense == "maximize" else 1.0
    if spec.get("type", "quadratic") == "quadratic":
        q = np.asarray(spec.get("Q"), dtype=float)
        c = np.asarray(spec.get("c", [0.0] * q.shape[0]), dtype=float)
        constant = float(spec.get("constant", 0))

        def fn(x: np.ndarray) -> float:
            return sign * float(0.5 * x @ q @ x + c @ x + constant)

        return fn
    raise ValueError("Hiện hỗ trợ objective type quadratic.")


def _constraint(spec: dict[str, Any]) -> dict[str, Any]:
    coefficients = np.asarray(spec["coefficients"], dtype=float)
    rhs = float(spec.get("rhs", 0))
    if spec.get("type", "ineq") == "eq":
        return {"type": "eq", "fun": lambda x, a=coefficients, b=rhs: float(a @ x - b)}
    return {"type": "ineq", "fun": lambda x, a=coefficients, b=rhs: float(b - a @ x)}


def _kkt_notes(problem: dict[str, Any]) -> str:
    lines = [
        "Với bài toán minimize `f(x)` và ràng buộc `g_i(x) <= 0`, KKT gồm:",
        "- Stationarity: `∇f(x*) + Σ λ_i ∇g_i(x*) + Σ μ_j ∇h_j(x*) = 0`",
        "- Primal feasibility: mọi ràng buộc thỏa.",
        "- Dual feasibility: `λ_i >= 0`.",
        "- Complementary slackness: `λ_i g_i(x*) = 0`.",
    ]
    obj = problem.get("objective", {})
    if obj.get("type", "quadratic") == "quadratic":
        q = obj.get("Q")
        c = obj.get("c")
        lines.append(f"Ở đây `∇f(x)=Qx+c`, với `Q={q}`, `c={c}`.")
        try:
            eig = np.linalg.eigvals(np.asarray(q, dtype=float))
            lines.append(f"Convexity check: eigenvalues(Q) = {[round(float(v.real), 6) for v in eig]}; Q PSD thì nghiệm local là global.")
        except Exception:
            pass
    return "\n".join(lines)


def _circle_packing_starts(radii: list[float], attempts: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    upper = 2 * sum(radii)
    max_r = max(radii)
    starts: list[list[float]] = []

    for perm in permutations(range(len(radii))):
        width = sum(2 * radii[i] for i in perm)
        height = 2 * max_r
        centers = [None] * len(radii)
        x = 0.0
        for idx in perm:
            centers[idx] = (x + radii[idx], radii[idx])
            x += 2 * radii[idx]
        starts.append(_pack_start(height, width, centers))

        centers = [None] * len(radii)
        y = 0.0
        for idx in perm:
            centers[idx] = (radii[idx], y + radii[idx])
            y += 2 * radii[idx]
        starts.append(_pack_start(sum(2 * radii[i] for i in perm), 2 * max_r, centers))

    # Tangent triangle starts for each permutation.
    for perm in permutations(range(len(radii))):
        i, j, k = perm[:3]
        ri, rj, rk = radii[i], radii[j], radii[k]
        xi, yi = ri, ri
        xj, yj = xi + ri + rj, rj
        dik, djk = ri + rk, rj + rk
        xk = (dik**2 - djk**2 + xj**2 - xi**2 + yj**2 - yi**2) / (2 * (xj - xi))
        y_sq = max(0.0, dik**2 - (xk - xi) ** 2)
        yk = yi + math.sqrt(y_sq)
        centers = [None] * len(radii)
        centers[i], centers[j], centers[k] = (xi, yi), (xj, yj), (xk, yk)
        min_x = min(centers[t][0] - radii[t] for t in range(len(radii)))  # type: ignore[index]
        min_y = min(centers[t][1] - radii[t] for t in range(len(radii)))  # type: ignore[index]
        shifted = [(centers[t][0] - min_x, centers[t][1] - min_y) for t in range(len(radii))]  # type: ignore[index]
        width = max(shifted[t][0] + radii[t] for t in range(len(radii)))
        height = max(shifted[t][1] + radii[t] for t in range(len(radii)))
        starts.append(_pack_start(height, width, shifted))

    for _ in range(max(0, attempts - len(starts))):
        height = rng.uniform(2 * max_r, upper)
        width = rng.uniform(2 * max_r, upper)
        centers = [(rng.uniform(r, width - r), rng.uniform(r, height - r)) for r in radii]
        starts.append(_pack_start(height, width, centers))
    return starts


def _three_circle_tangent_layout_check(radii: list[float]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for perm in permutations(range(3)):
        ordered_radii = [radii[i] for i in perm]
        centers = _tangent_triangle_centers(ordered_radii)
        result = minimize_scalar(
            lambda theta, r=ordered_radii, c=centers: _rotated_bounding_perimeter(r, c, theta),
            bounds=(0, math.pi),
            method="bounded",
            options={"xatol": 1e-12},
        )
        candidate = _rotated_bounding_box_solution(ordered_radii, centers, float(result.x), perm)
        if best is None or candidate["perimeter"] < best["perimeter"]:
            best = candidate
    assert best is not None
    return best


def _tangent_triangle_centers(radii: list[float]) -> list[tuple[float, float]]:
    r1, r2, r3 = radii
    c1 = (0.0, 0.0)
    c2 = (r1 + r2, 0.0)
    d13 = r1 + r3
    d23 = r2 + r3
    x3 = (d13**2 - d23**2 + c2[0] ** 2) / (2 * c2[0])
    y3 = math.sqrt(max(0.0, d13**2 - x3**2))
    return [c1, c2, (x3, y3)]


def _rotated_bounding_perimeter(radii: list[float], centers: list[tuple[float, float]], theta: float) -> float:
    xs, ys = _rotated_extents(radii, centers, theta)
    return 2 * ((max(xs) - min(xs)) + (max(ys) - min(ys)))


def _rotated_extents(radii: list[float], centers: list[tuple[float, float]], theta: float) -> tuple[list[float], list[float]]:
    ct, st = math.cos(theta), math.sin(theta)
    xs: list[float] = []
    ys: list[float] = []
    for (x, y), radius in zip(centers, radii):
        xr = ct * x - st * y
        yr = st * x + ct * y
        xs.extend([xr - radius, xr + radius])
        ys.extend([yr - radius, yr + radius])
    return xs, ys


def _rotated_bounding_box_solution(
    ordered_radii: list[float],
    centers: list[tuple[float, float]],
    theta: float,
    permutation: tuple[int, int, int],
) -> dict[str, Any]:
    ct, st = math.cos(theta), math.sin(theta)
    xs, ys = _rotated_extents(ordered_radii, centers, theta)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    mapped: list[dict[str, Any]] = []
    for local_idx, original_idx in enumerate(permutation):
        x, y = centers[local_idx]
        xr = ct * x - st * y - min_x
        yr = st * x + ct * y - min_y
        mapped.append(
            {
                "circle": original_idx + 1,
                "radius": ordered_radii[local_idx],
                "x": round(xr, 6),
                "y": round(yr, 6),
            }
        )
    height = max_y - min_y
    width = max_x - min_x
    return {
        "method": "three_circle_mutual_tangent_rotation",
        "rotation_radians": round(theta, 10),
        "height": round(height, 6),
        "width": round(width, 6),
        "perimeter": round(2 * (height + width), 6),
        "permutation": [i + 1 for i in permutation],
        "circles": sorted(mapped, key=lambda item: item["circle"]),
        "interpretation": "For three circles, a strong candidate optimum occurs when all circles are mutually tangent and the bounding rectangle is tight after rotation.",
    }


def _pack_start(height: float, width: float, centers: list[tuple[float, float] | None]) -> list[float]:
    values = [max(height, 1.0), max(width, 1.0)]
    for center in centers:
        if center is None:
            values.extend([1.0, 1.0])
        else:
            values.extend([center[0], center[1]])
    return values


def _circle_solution_feasible(v: np.ndarray, radii: list[float]) -> bool:
    return all(item["slack"] >= -1e-5 for item in _circle_checks(v, radii))


def _circle_checks(v: np.ndarray, radii: list[float]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    a, b = float(v[0]), float(v[1])
    for i, r in enumerate(radii):
        x, y = float(v[2 + 2 * i]), float(v[3 + 2 * i])
        checks.extend(
            [
                {"constraint": f"circle_{i+1}_left", "lhs": x, "rhs": r, "slack": round(x - r, 6)},
                {"constraint": f"circle_{i+1}_bottom", "lhs": y, "rhs": r, "slack": round(y - r, 6)},
                {"constraint": f"circle_{i+1}_right", "lhs": b - x, "rhs": r, "slack": round(b - x - r, 6)},
                {"constraint": f"circle_{i+1}_top", "lhs": a - y, "rhs": r, "slack": round(a - y - r, 6)},
            ]
        )
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dx = float(v[2 + 2 * i] - v[2 + 2 * j])
            dy = float(v[3 + 2 * i] - v[3 + 2 * j])
            distance = math.sqrt(dx * dx + dy * dy)
            required = radii[i] + radii[j]
            checks.append(
                {
                    "constraint": f"circle_{i+1}_{j+1}_nonoverlap",
                    "distance": round(distance, 6),
                    "required": required,
                    "slack": round(distance - required, 6),
                }
            )
    return checks


def _circle_packing_svg(v: np.ndarray, radii: list[float]) -> str:
    height = float(v[0])
    width = float(v[1])
    margin = 18
    scale = 7.0
    svg_w = width * scale + 2 * margin
    svg_h = height * scale + 2 * margin
    colors = ["#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#f87171"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" height="{svg_h:.0f}" viewBox="0 0 {svg_w:.2f} {svg_h:.2f}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<rect x="{margin}" y="{margin}" width="{width * scale:.2f}" height="{height * scale:.2f}" fill="#111827" stroke="#93c5fd" stroke-width="2"/>',
    ]
    for i, radius in enumerate(radii):
        x = margin + float(v[2 + 2 * i]) * scale
        y = margin + (height - float(v[3 + 2 * i])) * scale
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius * scale:.2f}" fill="{colors[i % len(colors)]}" fill-opacity="0.32" stroke="{colors[i % len(colors)]}" stroke-width="2"/>'
        )
        parts.append(f'<text x="{x:.2f}" y="{y + 4:.2f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#e5e7eb">C{i+1}</text>')
    parts.append(
        f'<text x="{margin}" y="{svg_h - 4:.2f}" font-family="Arial" font-size="12" fill="#cbd5e1">A={height:.2f} cm, B={width:.2f} cm, P={2*(height+width):.2f} cm</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _circle_packing_report(
    v: np.ndarray,
    radii: list[float],
    circles: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    attempts: int,
    svg: str,
    tangent_check: dict[str, Any] | None,
) -> str:
    a, b = float(v[0]), float(v[1])
    data_uri = "data:image/svg+xml;utf8," + quote(svg)
    lines = [
        "# Designing Boxes Problem - Nonlinear Programming",
        "",
        "**Dạng bài:** Nonlinear circle packing / box design.",
        "",
        "## 1. Mô hình",
        "",
        "Minimize `z = 2(A+B)` với biến `A, B, X_i, Y_i`.",
        "",
        "Ràng buộc: mỗi circle nằm trong box và khoảng cách tâm giữa hai circle không nhỏ hơn tổng bán kính.",
        "",
        "## 2. Phương pháp giải",
        "",
        f"Dùng SLSQP multi-start với {attempts} điểm khởi tạo vì bài toán phi tuyến không lồi. Kết quả là local optimum tốt nhất tìm được; các ràng buộc được kiểm tra lại sau solve.",
        "",
        "## 3. Hình kiểm tra layout",
        "",
        f"![Circle packing layout]({data_uri})",
        "",
        "## 4. Nghiệm tìm được",
        "",
        f"- Height `A` = **{a:.4f} cm**",
        f"- Width `B` = **{b:.4f} cm**",
        f"- Perimeter `2(A+B)` = **{2 * (a + b):.4f} cm**",
        "",
        "| Circle | Radius | X | Y |",
        "|---:|---:|---:|---:|",
    ]
    for circle in circles:
        lines.append(f"| {circle['circle']} | {circle['radius']:.4f} | {circle['x']:.4f} | {circle['y']:.4f} |")
    if tangent_check:
        gap = abs(2 * (a + b) - float(tangent_check["perimeter"]))
        lines.extend(
            [
                "",
                "## 5. Global/Layout sanity check cho 3 circles",
                "",
                "Với 3 circles, một ứng viên rất mạnh là cấu hình cả ba circle tiếp xúc nhau, sau đó xoay cụm tangent triangle để bounding rectangle có chu vi nhỏ nhất.",
                "",
                f"- Tangent-layout perimeter = **{float(tangent_check['perimeter']):.4f} cm**",
                f"- SLSQP best perimeter = **{2 * (a + b):.4f} cm**",
                f"- Difference = **{gap:.8f} cm**",
                "",
                "Nếu difference gần 0, nghiệm số khớp với cấu hình tiếp xúc hình học mạnh nhất cho bài 3 circles.",
            ]
        )
    lines.extend(["", "## 6. Kiểm tra ràng buộc", "", "| Constraint | Slack / Distance Margin | Status |", "|---|---:|---|"])
    for check in checks:
        status = "OK" if check["slack"] >= -1e-5 else "VIOLATED"
        lines.append(f"| {check['constraint']} | {check['slack']:.6f} | {status} |")
    lines.extend(
        [
            "",
            "## 7. Kết luận",
            "",
            "Nghiệm NLP khả thi, có hình layout và tất cả ràng buộc đều được kiểm tra. Với 3 circles, tangent-layout sanity check giúp củng cố kết luận; nếu cần chứng minh global tuyệt đối, cần thêm global optimization/bounds formal.",
        ]
    )
    return "\n".join(lines)
