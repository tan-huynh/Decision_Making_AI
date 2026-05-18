from __future__ import annotations

from typing import Any


BIAS_CHECKS = [
    ("anchoring", ["first number", "initial estimate", "giá đầu tiên", "neo"]),
    ("confirmation_bias", ["confirm", "ủng hộ ý kiến", "chỉ tìm bằng chứng", "xác nhận"]),
    ("overconfidence", ["chắc chắn", "100%", "không thể sai", "sure"]),
    ("loss_aversion", ["sợ mất", "loss", "thua lỗ", "mất tiền"]),
    ("availability", ["vừa xảy ra", "gần đây", "recent", "nhớ nhất"]),
]


def behavioral_bias_audit(description: str, judgments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    text = description.lower()
    hits = []
    for bias, keywords in BIAS_CHECKS:
        matched = [kw for kw in keywords if kw in text]
        if matched:
            hits.append({"bias": bias, "matched": matched, "mitigation": _mitigation(bias)})
    calibration = []
    for item in judgments or []:
        confidence = float(item.get("confidence", 0))
        outcome = item.get("outcome")
        if confidence > 0.8 and outcome is False:
            calibration.append({"issue": "possible_overconfidence", "judgment": item})
    return {
        "status": "computed",
        "bias_hits": hits,
        "calibration_flags": calibration,
        "questions": [
            "Bạn có đang dựa vào con số neo ban đầu không?",
            "Có bằng chứng nào phản bác phương án bạn thích không?",
            "Nếu kết quả xấu xảy ra, nguyên nhân khả dĩ nhất là gì?",
            "Xác suất bạn đưa ra có được kiểm tra bằng base rate dữ liệu ngoài chưa?",
        ],
        "markdown_report": _report(hits, calibration),
    }


def _mitigation(bias: str) -> str:
    return {
        "anchoring": "Tạo estimate độc lập trước khi xem giá trị neo; dùng nhiều nguồn dữ liệu.",
        "confirmation_bias": "Tìm evidence phản bác và yêu cầu premortem.",
        "overconfidence": "Dùng calibration history và khoảng tin cậy rộng hơn.",
        "loss_aversion": "Tách loss thực tế khỏi loss cảm xúc; so sánh expected utility.",
        "availability": "Dùng base rate thay vì sự kiện dễ nhớ gần đây.",
    }.get(bias, "Dùng checklist bias và dữ liệu base rate.")


def _report(hits: list[dict[str, Any]], calibration: list[dict[str, Any]]) -> str:
    lines = ["### Behavioral / Subjective Judgment Audit", ""]
    if hits:
        lines.append("| Bias | Mitigation |")
        lines.append("|---|---|")
        for hit in hits:
            lines.append(f"| {hit['bias']} | {hit['mitigation']} |")
    else:
        lines.append("Không phát hiện bias keyword rõ ràng trong mô tả, nhưng vẫn nên chạy premortem/base-rate check.")
    if calibration:
        lines.append("")
        lines.append(f"Calibration flags: {len(calibration)}.")
    return "\n".join(lines)
