from __future__ import annotations

from typing import Any

from agentic_research import is_medication_question


HIGH_STAKES = {"health", "legal", "safety", "finance"}

REQUIRED_KEYWORDS = {
    "health": ["tuổi", "liều", "thuốc", "dị ứng", "bệnh nền"],
    "legal": ["khu vực", "jurisdiction", "deadline", "hợp đồng", "tài liệu"],
    "finance": ["số tiền", "thời hạn", "lỗ tối đa", "thanh khoản"],
    "safety": ["mức độ nguy hiểm", "khẩn cấp", "ai bị ảnh hưởng"],
}


def safety_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    domain = str(payload.get("domain", "life")).lower()
    text = " ".join(str(payload.get(key, "")).lower() for key in ["question", "objective", "context"])
    missing = []
    for keyword in REQUIRED_KEYWORDS.get(domain, []):
        if keyword.lower() not in text:
            missing.append(keyword)

    risk_level = "low"
    requires_clarification = False
    warnings: list[str] = []
    if domain in HIGH_STAKES:
        risk_level = "medium"
        warnings.append("Đây là lĩnh vực high-stakes; kết quả chỉ là hỗ trợ ra quyết định.")
    if domain in {"health", "legal", "safety"} and missing:
        risk_level = "high"
        requires_clarification = True
    if is_medication_question(payload):
        risk_level = "high"
        requires_clarification = True
        warnings.append("Với thuốc, cần xác nhận hoạt chất, liều, thuốc dùng chung, dị ứng và bệnh nền trước khi khuyến nghị dùng/không dùng.")
    if any(word in text for word in ["cấp cứu", "khẩn cấp", "đau ngực", "khó thở", "tự tử", "bạo lực"]):
        risk_level = "urgent"
        requires_clarification = True
        warnings.append("Có dấu hiệu khẩn cấp; cần liên hệ dịch vụ khẩn cấp hoặc chuyên gia ngay.")

    return {
        "risk_level": risk_level,
        "requires_clarification": requires_clarification,
        "missing_fields": missing,
        "warnings": warnings,
    }
