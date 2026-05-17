"""Assignment Solver — Hungarian method (scipy) + DP fallback.

Uses scipy.optimize.linear_sum_assignment for O(n³) Hungarian method.
Falls back to exact DP with bitmask for small matrices when scipy unavailable.
Handles rectangular matrices by padding with dummy rows/columns.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def solve_assignment(costs: list[list[float]], maximize: bool = False) -> dict[str, Any]:
    """Solve the assignment problem.

    Supports rectangular matrices (agents ≠ tasks) by padding.
    Uses Hungarian method (scipy) when available; DP fallback for small matrices.
    """
    if not costs or not costs[0]:
        raise ValueError("Assignment requires a non-empty cost matrix.")

    rows = len(costs)
    cols = len(costs[0])

    # Pad to square if needed
    padded, original_rows, original_cols = _pad_to_square(costs)

    if HAS_SCIPY:
        return _solve_hungarian(padded, original_rows, original_cols, costs, maximize)

    # DP fallback: only for small matrices (≤20 columns due to bitmask)
    if len(padded) <= 20:
        return _solve_dp(padded, original_rows, original_cols, costs, maximize)

    raise ValueError(
        f"Matrix too large ({rows}×{cols}) for DP fallback. Install scipy for Hungarian method."
    )


def _pad_to_square(costs: list[list[float]]) -> tuple[list[list[float]], int, int]:
    """Pad rectangular matrix to square with zero-cost dummy rows/columns."""
    rows = len(costs)
    cols = len(costs[0])
    original_rows = rows
    original_cols = cols
    n = max(rows, cols)

    padded = [row[:] for row in costs]

    # Add dummy rows if needed
    for _ in range(n - rows):
        padded.append([0.0] * cols)

    # Add dummy columns if needed
    if cols < n:
        for row in padded:
            row.extend([0.0] * (n - cols))

    return padded, original_rows, original_cols


def _solve_hungarian(
    padded: list[list[float]],
    original_rows: int,
    original_cols: int,
    original_costs: list[list[float]],
    maximize: bool,
) -> dict[str, Any]:
    """Solve using scipy's linear_sum_assignment (Hungarian method)."""
    import numpy as np

    cost_matrix = np.array(padded, dtype=float)
    row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=maximize)

    # Filter out dummy assignments
    assignments: list[dict[str, Any]] = []
    total = 0.0
    for r, c in zip(row_idx, col_idx):
        if r < original_rows and c < original_cols:
            cost = float(original_costs[r][c])
            assignments.append({"row": int(r), "column": int(c), "cost": cost})
            total += cost

    was_padded = original_rows != original_cols
    return {
        "status": "optimal",
        "solver": "scipy_hungarian",
        "objective_value": round(total, 6),
        "assignment": assignments,
        "was_padded": was_padded,
        "original_size": f"{original_rows}×{original_cols}",
        "recommendation": (
            f"Assignment tối ưu (Hungarian O(n³)) có tổng "
            f"{'lợi ích' if maximize else 'chi phí'} = {total:.3f}."
            + (f" Ma trận đã được pad từ {original_rows}×{original_cols} thành vuông." if was_padded else "")
        ),
    }


def _solve_dp(
    padded: list[list[float]],
    original_rows: int,
    original_cols: int,
    original_costs: list[list[float]],
    maximize: bool,
) -> dict[str, Any]:
    """Solve using exact DP with bitmask (fallback for small matrices)."""
    n = len(padded)

    @lru_cache(maxsize=None)
    def dp(row: int, mask: int) -> tuple[float, tuple[int, ...]]:
        if row == n:
            return 0.0, ()
        best_value = float("-inf") if maximize else float("inf")
        best_assign: tuple[int, ...] = ()
        for col in range(n):
            if mask & (1 << col):
                continue
            rest_value, rest_assign = dp(row + 1, mask | (1 << col))
            value = float(padded[row][col]) + rest_value
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_value = value
                best_assign = (col,) + rest_assign
        return best_value, best_assign

    _, full_assignment = dp(0, 0)

    # Filter out dummy assignments
    assignments: list[dict[str, Any]] = []
    total = 0.0
    for r, c in enumerate(full_assignment):
        if r < original_rows and c < original_cols:
            cost = float(original_costs[r][c])
            assignments.append({"row": r, "column": c, "cost": cost})
            total += cost

    return {
        "status": "optimal",
        "solver": "exact_dp_assignment",
        "objective_value": round(total, 6),
        "assignment": assignments,
        "original_size": f"{original_rows}×{original_cols}",
        "recommendation": (
            f"Assignment tối ưu (DP exact) có tổng "
            f"{'lợi ích' if maximize else 'chi phí'} = {total:.3f}."
        ),
    }
