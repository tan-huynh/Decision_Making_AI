"""Audit & Traceability Module — Log every step of decision analysis.

Records inputs, solver selections, results, and timestamps for
full reproducibility. Distinguishes decision quality from outcome luck.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


AUDIT_DIR = Path(__file__).resolve().parent.parent / "data" / "audit"


def create_audit_trail(problem_id: str | None = None) -> dict[str, Any]:
    """Create a new audit trail for a decision problem."""
    trail_id = str(uuid4())
    return {
        "trail_id": trail_id,
        "problem_id": problem_id or trail_id,
        "created_at": time.time(),
        "steps": [],
    }


def log_step(
    trail: dict[str, Any],
    step_name: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a step to the audit trail."""
    step = {
        "step_id": str(uuid4()),
        "step_name": step_name,
        "timestamp": time.time(),
        "input_summary": _summarize(input_data),
        "output_summary": _summarize(output_data),
        "metadata": metadata or {},
    }
    trail["steps"].append(step)
    return step


def save_audit_trail(trail: dict[str, Any]) -> str:
    """Persist audit trail to disk as JSON."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{trail['problem_id']}_{int(trail['created_at'])}.json"
    filepath = AUDIT_DIR / filename
    filepath.write_text(json.dumps(trail, indent=2, ensure_ascii=False, default=str))
    return str(filepath)


def load_audit_trail(problem_id: str) -> dict[str, Any] | None:
    """Load most recent audit trail for a problem."""
    if not AUDIT_DIR.exists():
        return None
    candidates = sorted(AUDIT_DIR.glob(f"{problem_id}_*.json"), reverse=True)
    if not candidates:
        return None
    return json.loads(candidates[0].read_text())


def decision_quality_assessment(
    trail: dict[str, Any],
    actual_outcome: str = "",
) -> dict[str, Any]:
    """Assess decision quality separate from outcome.

    A good decision can have a bad outcome (unlucky) and vice versa.
    """
    steps = trail.get("steps", [])
    had_validation = any(s["step_name"] == "model_validation" for s in steps)
    had_sensitivity = any(s["step_name"] == "sensitivity_analysis" for s in steps)
    had_risk = any(s["step_name"] == "risk_analysis" for s in steps)
    had_solver = any(s["step_name"] == "solver_execution" for s in steps)

    quality_score = 0
    factors: list[str] = []

    if had_solver:
        quality_score += 30
        factors.append("✓ Solver thật được sử dụng (không dùng LLM tính toán)")
    else:
        factors.append("✗ Không có solver result — kết luận thiếu căn cứ toán học")

    if had_validation:
        quality_score += 20
        factors.append("✓ Model đã được validate trước khi solve")
    else:
        factors.append("✗ Model chưa validate — có thể có lỗi constraint/sign")

    if had_sensitivity:
        quality_score += 25
        factors.append("✓ Sensitivity analysis đã chạy")
    else:
        factors.append("○ Chưa chạy sensitivity analysis")

    if had_risk:
        quality_score += 25
        factors.append("✓ Risk analysis đã chạy")
    else:
        factors.append("○ Chưa chạy risk analysis")

    assessment = "robust" if quality_score >= 80 else "adequate" if quality_score >= 50 else "insufficient"

    return {
        "decision_quality_score": quality_score,
        "assessment": assessment,
        "factors": factors,
        "actual_outcome": actual_outcome,
        "note": (
            "Decision quality đánh giá quy trình ra quyết định, "
            "không phải kết quả thực tế. Kết quả tốt có thể do may mắn; "
            "quyết định tốt dựa trên model đúng, dữ liệu đủ và phân tích rủi ro."
        ),
    }


def _summarize(data: Any, max_len: int = 500) -> str:
    """Create a summary string of data for audit logging."""
    if data is None:
        return ""
    s = str(data)
    return s[:max_len] + "..." if len(s) > max_len else s
