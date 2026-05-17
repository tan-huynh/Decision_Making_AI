"""Cargo / Multi-index LP parser for text_solver.

Parses airplane cargo loading problems into raw scipy LP format.
"""
from __future__ import annotations
import re
from typing import Any


def parse_cargo_lp_text(text: str) -> dict[str, Any] | None:
    """Parse cargo/airplane loading LP from Vietnamese/English text."""
    low = text.lower()
    is_cargo = any(k in low for k in [
        "khoang", "máy bay", "may bay", "cargo", "compartment",
        "hàng hóa", "hang hoa", "xếp hàng", "xep hang",
    ])
    is_lp = any(k in low for k in [
        "quy hoạch tuyến tính", "linear programming", "lợi nhuận",
        "maximize", "minimize", "tối đa", "tối ưu",
    ])
    if not (is_cargo and is_lp):
        return None

    # Extract compartment data from tables
    compartments = _parse_compartments(text)
    goods = _parse_goods(text)

    if len(compartments) < 2 or len(goods) < 2:
        return None

    has_balance = any(k in low for k in [
        "tỉ lệ", "ti le", "cân bằng", "can bang", "proportion",
        "như nhau", "nhu nhau", "bằng nhau", "bang nhau",
    ])

    return _build_cargo_lp(compartments, goods, has_balance, text)


def _parse_compartments(text: str) -> list[dict[str, Any]]:
    """Extract compartment weight/volume limits."""
    comps = []
    # Pattern: name | weight | volume
    for line in text.splitlines():
        # Try table row: | name | weight | volume |
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 3:
            name = parts[0].lower()
            nums = re.findall(r"[\d]+(?:[.,]\d+)?", " ".join(parts[1:]))
            if len(nums) >= 2 and any(k in name for k in [
                "trước", "truoc", "front", "trung", "center", "sau", "rear", "back"
            ]):
                comps.append({
                    "name": parts[0].strip(),
                    "weight_limit": float(nums[0].replace(",", ".")),
                    "volume_limit": float(nums[1].replace(",", ".")),
                })
    return comps


def _parse_goods(text: str) -> list[dict[str, Any]]:
    """Extract goods: weight available, volume/ton, profit/ton."""
    goods = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 4:
            nums = re.findall(r"[\d]+(?:[.,]\d+)?", " ".join(parts))
            # First number might be the good index
            if len(nums) >= 4:
                idx = int(nums[0])
                if 1 <= idx <= 20:
                    goods.append({
                        "name": f"Loại {idx}",
                        "available": float(nums[1].replace(",", ".")),
                        "volume_rate": float(nums[2].replace(",", ".")),
                        "profit_rate": float(nums[3].replace(",", ".")),
                    })
    return goods


def _build_cargo_lp(
    compartments: list[dict], goods: list[dict],
    has_balance: bool, raw_text: str,
) -> dict[str, Any]:
    """Build raw scipy LP from parsed cargo data."""
    nc = len(compartments)
    ng = len(goods)
    n = ng * nc  # total variables: x_ij

    # Variable names: x_ij
    var_names = []
    for i in range(ng):
        for j in range(nc):
            var_names.append(f"x{i+1}{j+1}")

    # Objective: maximize sum of profit_i * x_ij
    c = []
    for i in range(ng):
        for j in range(nc):
            c.append(goods[i]["profit_rate"])

    # Build constraints
    A_ub, b_ub, ub_names = [], [], []
    A_eq, b_eq, eq_names = [], [], []

    # (a) Weight capacity per compartment
    for j in range(nc):
        row = [0.0] * n
        for i in range(ng):
            row[i * nc + j] = 1.0
        A_ub.append(row)
        b_ub.append(compartments[j]["weight_limit"])
        ub_names.append(f"Trọng lượng {compartments[j]['name']}")

    # (b) Volume capacity per compartment
    for j in range(nc):
        row = [0.0] * n
        for i in range(ng):
            row[i * nc + j] = goods[i]["volume_rate"]
        A_ub.append(row)
        b_ub.append(compartments[j]["volume_limit"])
        ub_names.append(f"Thể tích {compartments[j]['name']}")

    # (c) Available goods
    for i in range(ng):
        row = [0.0] * n
        for j in range(nc):
            row[i * nc + j] = 1.0
        A_ub.append(row)
        b_ub.append(goods[i]["available"])
        ub_names.append(f"Hàng {goods[i]['name']}")

    # (d) Balance: weight_j / capacity_j equal across compartments
    if has_balance:
        for j in range(1, nc):
            row = [0.0] * n
            cap0 = compartments[0]["weight_limit"]
            capj = compartments[j]["weight_limit"]
            for i in range(ng):
                row[i * nc + 0] = capj    # capj * x_i0
                row[i * nc + j] = -cap0   # -cap0 * x_ij
            A_eq.append(row)
            b_eq.append(0.0)
            eq_names.append(f"Cân bằng {compartments[0]['name']}-{compartments[j]['name']}")

    # Formulation text
    obj_terms = " + ".join(f"{g['profit_rate']}·Σⱼx{i+1}ⱼ" for i, g in enumerate(goods))
    formulation_lines = [
        f"Max Z = {obj_terms}",
        "",
        "Biến quyết định: x_ij = tấn hàng loại i xếp vào khoang j",
        f"  i = 1..{ng} (loại hàng), j = 1..{nc} (khoang)",
        f"  Tổng: {n} biến",
        "",
        "Ràng buộc:",
    ]
    for j, comp in enumerate(compartments):
        terms = " + ".join(f"x{i+1}{j+1}" for i in range(ng))
        formulation_lines.append(f"  Trọng lượng {comp['name']}: {terms} ≤ {comp['weight_limit']}")
    for j, comp in enumerate(compartments):
        terms = " + ".join(f"{g['volume_rate']}·x{i+1}{j+1}" for i, g in enumerate(goods))
        formulation_lines.append(f"  Thể tích {comp['name']}: {terms} ≤ {comp['volume_limit']}")
    for i, g in enumerate(goods):
        terms = " + ".join(f"x{i+1}{j+1}" for j in range(nc))
        formulation_lines.append(f"  Hàng {g['name']}: {terms} ≤ {g['available']}")
    if has_balance:
        formulation_lines.append("  Cân bằng tỉ lệ: W_j/Cap_j bằng nhau ∀j")

    # Steps for display
    steps = [
        f"1. Đặt {n} biến quyết định x_ij (i=1..{ng}, j=1..{nc})",
        f"2. Hàm mục tiêu: Max Z = {obj_terms}",
        f"3. {nc} ràng buộc trọng lượng khoang (≤)",
        f"4. {nc} ràng buộc thể tích khoang (≤)",
        f"5. {ng} ràng buộc hàng sẵn có (≤)",
    ]
    if has_balance:
        steps.append(f"6. {nc-1} ràng buộc cân bằng tỉ lệ (=)")
    steps.append(f"Tổng: {n} biến, {len(A_ub)} bất đẳng thức, {len(A_eq)} đẳng thức")

    return {
        "context": {
            "title": "Cargo Loading LP — Xếp hàng hóa",
            "domain": "logistics",
            "decision_maker": "logistics planner",
            "objective_direction": "maximize",
            "unit": "$/tấn",
            "description": raw_text[:3000],
        },
        "problem_type": "linear_programming",
        "c": c,
        "sense": "maximize",
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq if A_eq else None,
        "b_eq": b_eq if b_eq else None,
        "bounds": [[0, None]] * n,
        "variable_names": var_names,
        "constraint_names_ub": ub_names,
        "constraint_names_eq": eq_names,
        "formulation": "\n".join(formulation_lines),
        "steps": steps,
        "summary_tables": {
            "compartments": compartments,
            "goods": goods,
        },
        "assumptions": [
            "Hàng hóa có thể chia nhỏ (biến liên tục, không nguyên).",
            "Chi phí vận chuyển tuyến tính theo trọng lượng.",
            "Thể tích = thể tích riêng × trọng lượng.",
            "Ràng buộc cân bằng: tỉ lệ tải/giới hạn bằng nhau ở mọi khoang." if has_balance else "",
        ],
    }
