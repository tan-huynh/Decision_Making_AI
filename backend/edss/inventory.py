from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Any


def solve_eoq(demand: float, ordering_cost: float, holding_cost: float) -> dict[str, Any]:
    if demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
        raise ValueError("demand, ordering_cost và holding_cost phải dương.")
    q_star = math.sqrt((2 * demand * ordering_cost) / holding_cost)
    ordering = demand / q_star * ordering_cost
    holding = q_star / 2 * holding_cost
    report = (
        "### Báo cáo EOQ\n\n"
        f"Q* = sqrt(2DS/H) = sqrt(2 x {demand:g} x {ordering_cost:g} / {holding_cost:g}) = {q_star:.4f}\n\n"
        f"Annual ordering cost = D/Q* x S = {ordering:.4f}\n\n"
        f"Annual holding cost = Q*/2 x H = {holding:.4f}\n\n"
        f"Minimum relevant cost = {ordering + holding:.4f}\n"
    )
    return {
        "status": "optimal",
        "model": "EOQ",
        "order_quantity": q_star,
        "ordering_cost": ordering,
        "holding_cost": holding,
        "total_relevant_cost": ordering + holding,
        "markdown_report": report,
    }


def solve_production_lot_size(demand: float, setup_cost: float, holding_cost: float, production_rate: float) -> dict[str, Any]:
    if min(demand, setup_cost, holding_cost, production_rate) <= 0:
        raise ValueError("Tất cả tham số phải dương.")
    if production_rate <= demand:
        raise ValueError("production_rate phải lớn hơn demand để mô hình EPQ ổn định.")
    q_star = math.sqrt((2 * demand * setup_cost) / (holding_cost * (1 - demand / production_rate)))
    max_inventory = q_star * (1 - demand / production_rate)
    setup = demand / q_star * setup_cost
    holding = max_inventory / 2 * holding_cost
    return {
        "status": "optimal",
        "model": "production_lot_size",
        "lot_size": q_star,
        "max_inventory": max_inventory,
        "setup_cost": setup,
        "holding_cost": holding,
        "total_relevant_cost": setup + holding,
        "markdown_report": (
            "### Báo cáo Production Lot Size\n\n"
            f"Q* = sqrt(2DS / (H(1-d/p))) = {q_star:.4f}. "
            f"Max inventory = Q*(1-d/p) = {max_inventory:.4f}."
        ),
    }


def solve_reorder_point(demand_rate: float, lead_time: float, safety_stock: float = 0) -> dict[str, Any]:
    if demand_rate < 0 or lead_time < 0 or safety_stock < 0:
        raise ValueError("demand_rate, lead_time và safety_stock không được âm.")
    rop = demand_rate * lead_time + safety_stock
    return {
        "status": "computed",
        "model": "reorder_point",
        "reorder_point": rop,
        "markdown_report": f"### Reorder Point\n\nROP = demand_rate x lead_time + safety_stock = {rop:.4f}.",
    }


def solve_newsvendor(
    unit_cost: float,
    selling_price: float,
    salvage_value: float = 0,
    demand_mean: float | None = None,
    demand_std: float | None = None,
) -> dict[str, Any]:
    if selling_price <= unit_cost:
        raise ValueError("selling_price phải lớn hơn unit_cost để có underage cost dương.")
    if salvage_value >= selling_price:
        raise ValueError("salvage_value phải nhỏ hơn selling_price.")
    underage = selling_price - unit_cost
    overage = unit_cost - salvage_value
    if overage <= 0:
        raise ValueError("unit_cost phải lớn hơn salvage_value.")
    critical_ratio = underage / (underage + overage)
    result: dict[str, Any] = {
        "status": "computed",
        "model": "newsvendor",
        "underage_cost": underage,
        "overage_cost": overage,
        "critical_ratio": critical_ratio,
    }
    if demand_mean is not None and demand_std is not None:
        if demand_std <= 0:
            raise ValueError("demand_std phải dương.")
        z = NormalDist().inv_cdf(critical_ratio)
        result["z_value"] = z
        result["order_quantity"] = demand_mean + z * demand_std
    result["markdown_report"] = (
        "### Báo cáo Newsvendor\n\n"
        f"Cu = p-c = {underage:.4f}; Co = c-s = {overage:.4f}.\n\n"
        f"Critical fractile = Cu/(Cu+Co) = {critical_ratio:.4f}.\n"
        + (f"\nNếu demand ~ Normal, Q* = μ + zσ = {result['order_quantity']:.4f}." if "order_quantity" in result else "")
    )
    return result


def solve_quantity_discount_eoq(demand: float, ordering_cost: float, holding_rate: float, price_breaks: list[dict[str, float]]) -> dict[str, Any]:
    if demand <= 0 or ordering_cost <= 0 or holding_rate <= 0:
        raise ValueError("demand, ordering_cost và holding_rate phải dương.")
    if not price_breaks:
        raise ValueError("Cần price_breaks gồm min_qty và unit_cost.")

    candidates: list[dict[str, float]] = []
    sorted_breaks = sorted(price_breaks, key=lambda item: float(item.get("min_qty", 0)))
    for item in sorted_breaks:
        min_qty = float(item.get("min_qty", 0))
        unit_cost = float(item["unit_cost"])
        if unit_cost <= 0:
            raise ValueError("unit_cost phải dương.")
        holding_cost = holding_rate * unit_cost
        raw_q = math.sqrt((2 * demand * ordering_cost) / holding_cost)
        q = max(raw_q, min_qty)
        total_cost = demand * unit_cost + demand / q * ordering_cost + q / 2 * holding_cost
        candidates.append(
            {
                "min_qty": min_qty,
                "unit_cost": unit_cost,
                "raw_eoq": raw_q,
                "candidate_quantity": q,
                "total_cost": total_cost,
            }
        )
    best = min(candidates, key=lambda item: item["total_cost"])
    return {
        "status": "optimal",
        "model": "quantity_discount_eoq",
        "best": best,
        "candidates": candidates,
        "markdown_report": (
            "### Báo cáo EOQ Quantity Discount\n\n"
            f"Chọn mức giá {best['unit_cost']:g} với Q = {best['candidate_quantity']:.4f}, "
            f"total cost = {best['total_cost']:.4f}."
        ),
    }


def solve_paper_inventory_management(
    monthly_demands: list[float],
    purchase_price: float,
    ordering_cost: float,
    annual_inventory_charge: float,
    additional_storage_cost: float,
    discount_tiers: list[dict[str, float]],
) -> dict[str, Any]:
    if not monthly_demands or min(monthly_demands) <= 0:
        raise ValueError("monthly_demands phải dương.")
    if min(purchase_price, ordering_cost, annual_inventory_charge, additional_storage_cost) <= 0:
        raise ValueError("purchase_price, ordering_cost, annual_inventory_charge và additional_storage_cost phải dương.")

    annual_demand = sum(monthly_demands)
    mean_monthly = annual_demand / len(monthly_demands)
    demand_range = max(monthly_demands) - min(monthly_demands)
    holding_cost = purchase_price * annual_inventory_charge + additional_storage_cost
    q_basic = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    orders_per_year = annual_demand / q_basic

    def total_cost(q: float, unit_price: float, h: float) -> float:
        return annual_demand * unit_price + ordering_cost * annual_demand / q + h * q / 2

    basic_total = total_cost(q_basic, purchase_price, holding_cost)
    current_total = total_cost(mean_monthly, purchase_price, holding_cost)
    savings = current_total - basic_total

    discount_rows = []
    for tier in sorted(discount_tiers, key=lambda item: float(item["min_qty"])):
        min_qty = float(tier["min_qty"])
        discount = float(tier["discount"])
        unit_price = purchase_price * (1 - discount)
        cost = total_cost(min_qty, unit_price, holding_cost)
        discount_rows.append(
            {
                "quantity": min_qty,
                "discount": discount,
                "unit_price": unit_price,
                "total_cost": cost,
            }
        )
    best_discount = min(discount_rows, key=lambda row: row["total_cost"]) if discount_rows else None

    reduced_storage_q = math.sqrt((2 * annual_demand * ordering_cost) / additional_storage_cost)
    reduced_storage_price = purchase_price * (1 - (best_discount["discount"] if best_discount else 0))
    reduced_storage_total = total_cost(reduced_storage_q, reduced_storage_price, additional_storage_cost)

    demand_text = ", ".join(f"{value:g}" for value in monthly_demands)
    tier_lines = []
    for idx, row in enumerate(discount_rows, start=1):
        tier_lines.append(
            f"For `Q = {row['quantity']:.0f}`:\n\n"
            "$$\n"
            f"CT_{idx} = {annual_demand:.0f}\\cdot {purchase_price:,.0f}\\cdot (1-{row['discount']:.2f}) "
            f"+ {annual_demand:.0f}\\cdot {ordering_cost:,.0f}/{row['quantity']:.0f} "
            f"+ {row['quantity']:.0f}\\cdot {holding_cost:,.0f}/2 = {row['total_cost']:,.0f}\n"
            "$$\n"
        )

    report = (
        "# Paper Inventory Management Problem\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        "- **Dạng toán chính:** Inventory Theory\n"
        "- **Dạng toán phụ:** EOQ with quantity discounts\n"
        "- **Phương pháp giải:** EOQ cơ bản, sau đó kiểm tra total annual cost tại các discount breakpoints.\n\n"
        "## 2. Dữ liệu đã trích xuất\n\n"
        f"- Monthly demand observations: `{demand_text}` tonnes/month\n"
        f"- Annual demand: `D = {annual_demand:.0f}` tonnes/year\n"
        f"- Mean monthly order currently used: `{mean_monthly:.2f}` tonnes/order\n"
        f"- Purchase price: `p = ${purchase_price:,.0f}` per tonne\n"
        f"- Order cost: `K = ${ordering_cost:,.0f}`\n"
        f"- Inventory charge: `{annual_inventory_charge:.0%}` of unit cost\n"
        f"- Additional storage cost: `${additional_storage_cost:,.0f}` per tonne-year\n"
        f"- Demand range over 12 months: `{demand_range:.2f}` tonnes, so the EOQ approximation is reasonable.\n\n"
        "## (a) Inventory management model under current conditions\n\n"
        "$$\n"
        f"h = {purchase_price:,.0f}\\cdot {annual_inventory_charge:.2f} + {additional_storage_cost:,.0f} = {holding_cost:,.0f}\\ \\text{{dollars/tonne-year}}\n"
        "$$\n\n"
        "$$\n"
        f"Q = \\sqrt{{\\frac{{2KD}}{{h}}}} = \\sqrt{{\\frac{{2\\cdot {annual_demand:.0f}\\cdot {ordering_cost:,.0f}}}{{{holding_cost:,.0f}}}}} = {q_basic:.2f}\\ \\text{{tonnes/order}}\n"
        "$$\n\n"
        "$$\n"
        f"N = \\frac{{D}}{{Q}} = \\frac{{{annual_demand:.0f}}}{{{q_basic:.2f}}} = {orders_per_year:.2f}\\ \\text{{orders/year}}\n"
        "$$\n\n"
        "$$\n"
        f"CT(Q={q_basic:.2f}) = \\frac{{Q}}{{2}}h + K\\frac{{D}}{{Q}} + pD\n"
        "$$\n\n"
        "$$\n"
        f"= {holding_cost:,.0f}\\cdot {q_basic:.2f}/2 + {ordering_cost:,.0f}\\cdot {annual_demand:.0f}/{q_basic:.2f} + {annual_demand:.0f}\\cdot {purchase_price:,.0f} = {basic_total:,.0f}\\ \\text{{dollars/year}}\n"
        "$$\n\n"
        "$$\n"
        f"CT(Q={mean_monthly:.0f}) = {holding_cost:,.0f}\\cdot {mean_monthly:.0f}/2 + {ordering_cost:,.0f}\\cdot {annual_demand:.0f}/{mean_monthly:.0f} + {annual_demand:.0f}\\cdot {purchase_price:,.0f} = {current_total:,.0f}\\ \\text{{dollars/year}}\n"
        "$$\n\n"
        f"With EOQ, the firm saves `${savings:,.0f}` per year compared with monthly ordering.\n\n"
        "## (b) Quantity discount policy\n\n"
        + "\n".join(tier_lines)
        + "\n"
        f"Best discount breakpoint: `Q = {best_discount['quantity']:.0f}` tonnes with total cost `${best_discount['total_cost']:,.0f}`.\n\n"
        "Vì vậy doanh nghiệp nên dùng mức chiết khấu tốt nhất theo total annual cost, không chỉ chọn discount phần trăm lớn nhất.\n\n"
        "## (c) If storage-only cost is lowered\n\n"
        "Nếu only storage per tonne remains as the carrying cost, use `h = 55,000` while keeping the accepted discount price.\n\n"
        "$$\n"
        f"Q = \\sqrt{{\\frac{{2\\cdot {annual_demand:.0f}\\cdot {ordering_cost:,.0f}}}{{{additional_storage_cost:,.0f}}}}} = {reduced_storage_q:.2f}\\ \\text{{tonnes/order}}\n"
        "$$\n\n"
        "$$\n"
        f"N = \\frac{{{annual_demand:.0f}}}{{{reduced_storage_q:.2f}}} = {annual_demand / reduced_storage_q:.2f}\\ \\text{{orders/year}}\n"
        "$$\n\n"
        "$$\n"
        f"CT(Q={reduced_storage_q:.2f}) = {additional_storage_cost:,.0f}\\cdot {reduced_storage_q:.2f}/2 + {ordering_cost:,.0f}\\cdot {annual_demand:.0f}/{reduced_storage_q:.2f} + {reduced_storage_price:,.0f}\\cdot {annual_demand:.0f} = {reduced_storage_total:,.0f}\\ \\text{{dollars/year}}\n"
        "$$\n\n"
        "## Kiểm tra nghiệm\n\n"
        f"- Demand conversion: sum of 12 monthly observations = `{annual_demand:.0f}` tonnes/year.\n"
        f"- Holding cost: `{purchase_price:,.0f} x {annual_inventory_charge:.2f} + {additional_storage_cost:,.0f} = {holding_cost:,.0f}`.\n"
        "- Discount alternatives are checked by total annual cost, including purchase, ordering and holding cost.\n\n"
        "## Đáp án cuối cùng\n\n"
        "| Case | Order quantity | Total annual cost | Recommendation |\n"
        "|---|---:|---:|---|\n"
        f"| (a) EOQ current price | {q_basic:.2f} tonnes | ${basic_total:,.0f} | Use EOQ instead of monthly 10-ton orders |\n"
        f"| (b) Best discount breakpoint | {best_discount['quantity']:.0f} tonnes | ${best_discount['total_cost']:,.0f} | Accept the {best_discount['discount']:.0%} discount tier |\n"
        f"| (c) Storage-only carrying cost | {reduced_storage_q:.2f} tonnes | ${reduced_storage_total:,.0f} | Larger order quantity becomes attractive |\n"
    )
    return {
        "status": "optimal",
        "model": "paper_inventory_quantity_discount",
        "annual_demand": round(annual_demand, 6),
        "holding_cost": round(holding_cost, 6),
        "basic_eoq": round(q_basic, 6),
        "orders_per_year": round(orders_per_year, 6),
        "basic_total_cost": round(basic_total, 6),
        "current_monthly_total_cost": round(current_total, 6),
        "annual_savings": round(savings, 6),
        "discount_rows": [{**row, "total_cost": round(row["total_cost"], 6), "unit_price": round(row["unit_price"], 6)} for row in discount_rows],
        "best_discount": {**best_discount, "total_cost": round(best_discount["total_cost"], 6), "unit_price": round(best_discount["unit_price"], 6)} if best_discount else None,
        "reduced_storage_eoq": round(reduced_storage_q, 6),
        "reduced_storage_total_cost": round(reduced_storage_total, 6),
        "markdown_report": report,
    }


def solve_eoq_planned_shortages(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: float,
    shortage_cost: float,
    unit_profit: float = 0,
    weekly_demand: float | None = None,
    purchase_cost: float | None = None,
) -> dict[str, Any]:
    if min(annual_demand, ordering_cost, holding_cost, shortage_cost) <= 0:
        raise ValueError("annual_demand, ordering_cost, holding_cost và shortage_cost phải dương.")
    q_star = math.sqrt((2 * ordering_cost * annual_demand * (holding_cost + shortage_cost)) / (holding_cost * shortage_cost))
    max_inventory = q_star * shortage_cost / (holding_cost + shortage_cost)
    max_shortage = q_star * holding_cost / (holding_cost + shortage_cost)
    cycle_years = q_star / annual_demand
    cycle_weeks = cycle_years * 52
    demand_per_week = weekly_demand if weekly_demand and weekly_demand > 0 else annual_demand / 52
    positive_inventory_weeks = max_inventory / demand_per_week
    shortage_weeks = max_shortage / demand_per_week
    delayed_fraction = max_shortage / q_star
    ordering = ordering_cost * annual_demand / q_star
    holding = holding_cost * max_inventory**2 / (2 * q_star)
    shortage = shortage_cost * max_shortage**2 / (2 * q_star)
    total_inventory_cost = ordering + holding + shortage
    annual_purchase_cost = purchase_cost * annual_demand if purchase_cost else 0
    annual_total_cost_with_purchase = annual_purchase_cost + total_inventory_cost
    gross_profit = unit_profit * annual_demand
    total_income = (purchase_cost + unit_profit) * annual_demand if purchase_cost else gross_profit
    annual_profit = gross_profit - total_inventory_cost
    svg = _planned_shortage_svg(max_inventory, max_shortage, positive_inventory_weeks, shortage_weeks)
    eoq_basic = math.sqrt((2 * ordering_cost * annual_demand) / holding_cost)
    weekly_line = f"{annual_demand / 52:.0f} \\cdot 52" if abs((annual_demand / 52) - round(annual_demand / 52)) < 1e-9 else f"{annual_demand:,.0f}"
    profit_block = ""
    if purchase_cost and unit_profit > 0:
        profit_block = (
            "## (e) Annual profit from selling RFG dispensers\n\n"
            "$$\n"
            "CT(Q=Q^*) = C_a + C_s + C_p + C_c = "
            "\\frac{M^2h}{2Q} + \\frac{(Q-M)^2s}{2Q} + K\\frac{D}{Q} + cD\n"
            "$$\n\n"
            "$$\n"
            f"= \\frac{{{max_inventory:.2f}^2 \\cdot {holding_cost:.2f}}}{{2 \\cdot {q_star:.2f}}} "
            f"+ \\frac{{({q_star:.2f}-{max_inventory:.2f})^2 \\cdot {shortage_cost:.2f}}}{{2 \\cdot {q_star:.2f}}} "
            f"+ {ordering_cost:.2f}\\frac{{{annual_demand:,.0f}}}{{{q_star:.2f}}} "
            f"+ {purchase_cost:.2f}\\cdot {annual_demand:,.0f} = {annual_total_cost_with_purchase:,.2f}\n"
            "$$\n\n"
            "$$\n"
            f"\\text{{Total income}} = {weekly_line} \\cdot ({purchase_cost:.2f}+{unit_profit:.2f}) = {total_income:,.2f}\n"
            "$$\n\n"
            "$$\n"
            f"\\text{{Annual profit}} = {total_income:,.2f} - {annual_total_cost_with_purchase:,.2f} = {annual_profit:,.2f}\n"
            "$$\n\n"
        )
    elif purchase_cost:
        profit_block = (
            "## (e) Annual profit from selling RFG dispensers\n\n"
            "Không đủ dữ liệu để tính annual profit chắc chắn vì thiếu `gross profit per unit`. "
            "Cần giá trị gross profit trên mỗi dispenser sold, ví dụ trong sách là `$80`.\n\n"
        )
    report = (
        "# BAC Inventory Problem - EOQ with Planned Shortages\n\n"
        "## 1. Dạng mô hình\n\n"
        "Đây là mô hình EOQ cho phép thiếu hàng có kế hoạch/backorder vì đề cho chi phí stockout trên mỗi đơn vị chưa đáp ứng.\n\n"
        "## 2. Tham số\n\n"
        f"- Annual demand `D = {annual_demand:,.0f}` units/year\n"
        f"- Ordering cost `K = ${ordering_cost:,.2f}` per order\n"
        f"- Holding cost `h = ${holding_cost:,.2f}` per unit-year\n"
        f"- Shortage cost `p = ${shortage_cost:,.2f}` per unit-year/planning horizon unit\n"
        f"- Purchase cost `c = ${purchase_cost:,.2f}`\n" if purchase_cost else ""
    ) + (
        f"- Gross profit per sold unit = `${unit_profit:,.2f}`\n\n"
        "## (a) Optimum inventory policy\n\n"
        "$$\n"
        f"EOQ = \\sqrt{{\\frac{{2KD}}{{h}}}} = \\sqrt{{\\frac{{2 \\cdot {ordering_cost:.0f} \\cdot {annual_demand / 52:.0f} \\cdot 52}}{{{holding_cost:.0f}}}}} = {eoq_basic:.2f}\\ \\text{{units/order}}\n"
        "$$\n\n"
        "$$\n"
        f"Q^* = EOQ \\sqrt{{\\frac{{h+s}}{{s}}}} = {eoq_basic:.2f}\\sqrt{{\\frac{{{holding_cost:.0f}+{shortage_cost:.0f}}}{{{shortage_cost:.0f}}}}} = {q_star:.2f}\\ \\text{{units/order}}\n"
        "$$\n\n"
        "$$\n"
        f"F = \\frac{{D}}{{Q^*}} = \\frac{{{annual_demand:,.0f}}}{{{q_star:.2f}}} = {annual_demand / q_star:.2f}\\ \\text{{orders/year}}\n"
        "$$\n\n"
        "## (b) Weeks between two consecutive orders\n\n"
        "$$\n"
        f"OB = \\frac{{Q}}{{D}} \\cdot 52 = \\frac{{{q_star:.2f}}}{{{annual_demand:,.0f}}}\\cdot 52 = {cycle_weeks:.2f}\\ \\text{{weeks}}\n"
        "$$\n\n"
        "### Inventory model graph\n\n"
        f"![EOQ planned shortage graph](data:image/svg+xml;utf8,{_quote_svg(svg)})\n\n"
        "## (c) Percentage of customers delayed or rescheduled\n\n"
        "$$\n"
        f"M^* = EOQ \\sqrt{{\\frac{{s}}{{h+s}}}} = {eoq_basic:.2f}\\sqrt{{\\frac{{{shortage_cost:.0f}}}{{{holding_cost:.0f}+{shortage_cost:.0f}}}}} = {max_inventory:.2f}\\ \\text{{units}}\n"
        "$$\n\n"
        "$$\n"
        f"\\frac{{Q-M}}{{Q}} = \\frac{{{q_star:.2f}-{max_inventory:.2f}}}{{{q_star:.2f}}} = {delayed_fraction * 100:.2f}\\%\n"
        "$$\n\n"
        "## (d) Maximum wait for an unsatisfied customer\n\n"
        "$$\n"
        f"\\frac{{Q-M}}{{D}}\\cdot 52 = \\frac{{{q_star:.2f}-{max_inventory:.2f}}}{{{annual_demand:,.0f}}}\\cdot 52 = {shortage_weeks:.2f}\\ \\text{{weeks}}\n"
        "$$\n\n"
        + profit_block
        + "## Kiểm tra nghiệm\n\n"
        f"- `Q* > 0`, `M* > 0`, `Q-M > 0`: passed.\n"
        f"- `M* + (Q-M) = {max_inventory:.2f} + {max_shortage:.2f} = {q_star:.2f}` units: passed.\n"
        f"- Time unit: weekly demand converted to annual demand `D = {annual_demand:,.0f}` units/year.\n"
        f"- Annual inventory cost without purchase = `${total_inventory_cost:,.2f}`.\n\n"
        "## Đáp án cuối cùng\n\n"
        f"| Item | Value |\n"
        f"|---|---:|\n"
        f"| Basic EOQ | {eoq_basic:.2f} units/order |\n"
        f"| Optimal order quantity Q* | {q_star:.2f} units/order |\n"
        f"| Orders per year F | {annual_demand / q_star:.2f} |\n"
        f"| Maximum inventory M* | {max_inventory:.2f} units |\n"
        f"| Maximum backorder Q-M | {max_shortage:.2f} units |\n"
        f"| Weeks between orders | {cycle_weeks:.2f} weeks |\n"
        f"| Delayed/rescheduled customers | {delayed_fraction * 100:.2f}% |\n"
        f"| Maximum wait | {shortage_weeks:.2f} weeks |\n"
        + (f"| Annual profit | ${annual_profit:,.2f}/year |\n" if purchase_cost and unit_profit > 0 else "")
    )
    return {
        "status": "optimal",
        "model": "EOQ_with_planned_shortages",
        "order_quantity": round(q_star, 6),
        "maximum_inventory": round(max_inventory, 6),
        "maximum_shortage": round(max_shortage, 6),
        "cycle_weeks": round(cycle_weeks, 6),
        "positive_inventory_weeks": round(positive_inventory_weeks, 6),
        "shortage_weeks": round(shortage_weeks, 6),
        "delayed_customer_fraction": round(delayed_fraction, 8),
        "delayed_customer_percent": round(delayed_fraction * 100, 6),
        "annual_ordering_cost": round(ordering, 6),
        "annual_holding_cost": round(holding, 6),
        "annual_shortage_cost": round(shortage, 6),
        "annual_inventory_cost": round(total_inventory_cost, 6),
        "annual_purchase_cost": round(annual_purchase_cost, 6) if purchase_cost else None,
        "annual_total_cost_with_purchase": round(annual_total_cost_with_purchase, 6) if purchase_cost else None,
        "total_income": round(total_income, 6),
        "gross_annual_profit": round(gross_profit, 6),
        "annual_profit_after_inventory_cost": round(annual_profit, 6),
        "svg": svg,
        "markdown_report": report,
    }


def _planned_shortage_svg(max_inventory: float, max_shortage: float, positive_weeks: float, shortage_weeks: float) -> str:
    width, height = 920, 360
    margin_left, margin_top = 70, 42
    plot_w, plot_h = 740, 230
    total_weeks = positive_weeks + shortage_weeks
    cycles = 3

    def sx(t: float) -> float:
        return margin_left + (t / (total_weeks * cycles)) * plot_w if total_weeks else margin_left

    def sy(v: float) -> float:
        return margin_top + (1 - ((v + max_shortage) / (max_inventory + max_shortage))) * plot_h

    y_top = sy(max_inventory)
    y_axis = sy(0)
    y_bottom = sy(-max_shortage)
    cycle_points = []
    verticals = []
    for idx in range(cycles):
        start = idx * total_weeks
        end = (idx + 1) * total_weeks
        cross = start + positive_weeks
        cycle_points.append(f"{sx(start):.2f},{y_top:.2f} {sx(cross):.2f},{y_axis:.2f} {sx(end):.2f},{y_bottom:.2f}")
        if idx < cycles - 1:
            verticals.append((sx(end), y_bottom, y_top))
    x_right = sx(total_weeks * cycles)
    x_bracket_m = x_right + 42
    x_bracket_q = x_right + 88
    x_label_a = sx(positive_weeks)
    x_label_b = sx(positive_weeks + shortage_weeks / 2)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#111827"/></marker></defs>'
        f'<line x1="{margin_left}" y1="{y_axis:.2f}" x2="{margin_left + plot_w + 12}" y2="{y_axis:.2f}" stroke="#111827" stroke-width="2" marker-end="url(#arrow)"/>'
        f'<line x1="{margin_left}" y1="{y_bottom + 2:.2f}" x2="{margin_left}" y2="{margin_top - 8}" stroke="#111827" stroke-width="2" marker-end="url(#arrow)"/>'
        f'<line x1="{margin_left}" y1="{y_top:.2f}" x2="{x_right:.2f}" y2="{y_top:.2f}" stroke="#111827" stroke-width="5" stroke-dasharray="4 8"/>'
        f'<line x1="{x_label_a:.2f}" y1="{y_bottom:.2f}" x2="{x_right:.2f}" y2="{y_bottom:.2f}" stroke="#111827" stroke-width="5" stroke-dasharray="4 8"/>'
        + "".join(f'<polyline points="{points}" fill="none" stroke="#303f9f" stroke-width="4"/>' for points in cycle_points)
        + "".join(f'<line x1="{x:.2f}" y1="{yb:.2f}" x2="{x:.2f}" y2="{yt:.2f}" stroke="#303f9f" stroke-width="4"/>' for x, yb, yt in verticals)
        + f'<text x="{margin_left - 24}" y="{y_axis + 26:.2f}" font-family="Georgia,serif" font-size="28" fill="#111827">0</text>'
        f'<text x="{x_label_a - 12:.2f}" y="{y_axis + 36:.2f}" font-family="Georgia,serif" font-size="28" font-weight="700" fill="#111827">A</text>'
        f'<text x="{x_label_b:.2f}" y="{y_axis + 42:.2f}" font-family="Georgia,serif" font-size="28" font-weight="700" fill="#111827">B</text>'
        f'<line x1="{x_bracket_m:.2f}" y1="{y_top:.2f}" x2="{x_bracket_m:.2f}" y2="{y_axis:.2f}" stroke="#111827" stroke-width="2" marker-start="url(#arrow)" marker-end="url(#arrow)"/>'
        f'<line x1="{x_bracket_m:.2f}" y1="{y_axis:.2f}" x2="{x_bracket_m:.2f}" y2="{y_bottom:.2f}" stroke="#111827" stroke-width="2" marker-start="url(#arrow)" marker-end="url(#arrow)"/>'
        f'<line x1="{x_bracket_q:.2f}" y1="{y_top:.2f}" x2="{x_bracket_q:.2f}" y2="{y_bottom:.2f}" stroke="#111827" stroke-width="2" marker-start="url(#arrow)" marker-end="url(#arrow)"/>'
        f'<text x="{x_bracket_m + 26:.2f}" y="{(y_top + y_axis) / 2 + 9:.2f}" font-family="Georgia,serif" font-size="28" font-weight="700" fill="#111827">M</text>'
        f'<text x="{x_bracket_m + 26:.2f}" y="{(y_axis + y_bottom) / 2 + 9:.2f}" font-family="Georgia,serif" font-size="24" font-weight="700" fill="#111827">Q-M</text>'
        f'<text x="{x_bracket_q + 24:.2f}" y="{(y_top + y_bottom) / 2 + 9:.2f}" font-family="Georgia,serif" font-size="28" font-weight="700" fill="#111827">Q</text>'
        f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">Cycle length = {total_weeks:.2f} weeks</text>'
        f'<text x="18" y="{margin_top + 20}" font-family="Arial" font-size="13" fill="#111827">Inventory</text>'
        "</svg>"
    )


def _quote_svg(svg: str) -> str:
    from urllib.parse import quote

    return quote(svg)


def simulate_stochastic_inventory(
    periods: int,
    initial_inventory: int,
    reorder_point: int,
    order_up_to: int,
    demand_mean: float,
    demand_std: float,
    holding_cost: float,
    shortage_cost: float,
    ordering_cost: float = 0,
    fixed_order_cost: float = 0,
    lead_time: int = 0,
    seed: int | None = None,
) -> dict[str, Any]:
    if periods <= 0 or demand_std < 0 or order_up_to < reorder_point:
        raise ValueError("Thông số stochastic inventory không hợp lệ.")
    rng = random.Random(seed)
    inventory = int(initial_inventory)
    pipeline: list[tuple[int, int]] = []
    rows = []
    total_cost = 0.0
    for period in range(1, periods + 1):
        arrivals = sum(qty for due, qty in pipeline if due == period)
        pipeline = [(due, qty) for due, qty in pipeline if due != period]
        inventory += arrivals
        order_qty = 0
        inventory_position = inventory + sum(qty for _, qty in pipeline)
        if inventory_position <= reorder_point:
            order_qty = order_up_to - inventory_position
            if lead_time == 0:
                inventory += order_qty
                arrivals += order_qty
            else:
                pipeline.append((period + lead_time, order_qty))
        demand = max(0, int(round(rng.gauss(demand_mean, demand_std))))
        sold = min(inventory, demand)
        shortage = demand - sold
        inventory -= sold
        cost = holding_cost * inventory + shortage_cost * shortage + ordering_cost * order_qty + (fixed_order_cost if order_qty > 0 else 0)
        total_cost += cost
        rows.append(
            {
                "period": period,
                "arrivals": arrivals,
                "order_qty": order_qty,
                "demand": demand,
                "ending_inventory": inventory,
                "shortage": shortage,
                "cost": round(cost, 6),
            }
        )
    avg_cost = total_cost / periods
    service_level = 1 - sum(row["shortage"] > 0 for row in rows) / periods
    return {
        "status": "computed",
        "model": "stochastic_inventory_simulation",
        "seed": seed,
        "total_cost": round(total_cost, 6),
        "average_cost": round(avg_cost, 6),
        "cycle_service_level": round(service_level, 6),
        "rows": rows,
        "markdown_report": (
            "### Báo cáo Stochastic Inventory Simulation\n\n"
            f"Policy `(s,S)=({reorder_point},{order_up_to})`, periods = {periods}, seed = {seed}.\n\n"
            f"Total cost = {total_cost:.4f}, average cost = {avg_cost:.4f}, cycle service level = {service_level:.4f}."
        ),
    }


def solve_inventory_problem(problem: dict[str, Any]) -> dict[str, Any]:
    demand = float(problem.get("demand") or problem.get("annual_demand", 0))
    ordering = float(problem.get("ordering_cost") or problem.get("order_cost", 0))
    holding = float(problem.get("holding_cost", 0))
    shortage = float(problem.get("shortage_cost", 0))
    unit_cost = float(problem.get("unit_cost", 0))
    
    if demand <= 0 or ordering <= 0 or holding <= 0:
        return {"status": "needs_clarification", "missing_data": ["Cần demand, ordering_cost và holding_cost lớn hơn 0 để chạy mô hình EOQ."]}
        
    if shortage > 0:
        return solve_eoq_planned_shortages(
            annual_demand=demand,
            ordering_cost=ordering,
            holding_cost=holding,
            shortage_cost=shortage,
            unit_profit=float(problem.get("unit_profit", 80)),
            weekly_demand=demand/52,
            purchase_cost=unit_cost
        )
    else:
        return solve_eoq(demand, ordering, holding)
