from __future__ import annotations

from typing import Any


def wrap_teaching_report(title: str, problem_type: str, sections: list[tuple[str, str]], answer: str, checks: list[str] | None = None) -> str:
    lines = [f"# {title}", "", f"**Dạng bài:** {problem_type}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    lines.append("## Đáp án cuối cùng")
    lines.append("")
    lines.append(answer.strip())
    lines.append("")
    if checks:
        lines.append("## Kiểm tra tính hợp lệ")
        lines.append("")
        for check in checks:
            lines.append(f"- {check}")
    return "\n".join(lines).strip()


def missing_data_report(title: str, questions: list[str]) -> dict[str, Any]:
    markdown = wrap_teaching_report(
        title,
        "needs_clarification",
        [("Dữ liệu còn thiếu", "\n".join(f"- {q}" for q in questions))],
        "Không đủ dữ liệu để khuyến nghị chắc chắn.",
    )
    return {"status": "needs_clarification", "questions": questions, "markdown_report": markdown}
