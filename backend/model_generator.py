from __future__ import annotations

import hashlib
from typing import Any


DOMAIN_DEFAULTS = {
    "income": [
        ("Tăng lương trong công việc hiện tại", 8, 0.75, ["Được tăng lương", "Chỉ tăng nhỏ", "Không được duyệt"]),
        ("Freelance ngoài giờ", 15, 0.65, ["Có khách đều", "Có vài dự án nhỏ", "Không có traction"]),
        ("Học kỹ năng mới để đổi việc", 20, 0.7, ["Đổi việc thành công", "Tăng năng lực nhưng chưa đổi", "Mất thời gian"]),
    ],
    "career": [
        ("Ở lại và tối ưu vai trò hiện tại", 8, 0.8, ["Thăng tiến", "Ổn định", "Chậm phát triển"]),
        ("Chuyển sang cơ hội mới", 18, 0.55, ["Tăng trưởng mạnh", "Không phù hợp", "Rủi ro thử việc"]),
        ("Học thêm trước khi chuyển", 14, 0.85, ["Sẵn sàng hơn", "Mất momentum", "Phát hiện hướng tốt hơn"]),
    ],
    "health": [
        ("Tự xử lý theo thông tin hiện có", 5, 0.35, ["Cải thiện", "Không cải thiện", "Tác dụng phụ/rủi ro"]),
        ("Hỏi bác sĩ/dược sĩ trước", 10, 0.9, ["Có lựa chọn an toàn", "Chậm xử lý", "Cần khám thêm"]),
        ("Theo dõi thêm với ngưỡng đi khám", 6, 0.75, ["Tự hết", "Cần tư vấn", "Nặng lên"]),
    ],
    "legal": [
        ("Tự xử lý", 8, 0.35, ["Tiết kiệm chi phí", "Thiếu căn cứ", "Rủi ro pháp lý"]),
        ("Hỏi luật sư/chuyên gia", 25, 0.85, ["Giảm rủi ro", "Tốn phí", "Có chiến lược rõ"]),
        ("Thu thập thêm tài liệu trước", 12, 0.8, ["Củng cố hồ sơ", "Trễ deadline", "Phát hiện điểm yếu"]),
    ],
    "life": [
        ("Hành động ngay theo phương án chính", 12, 0.55, ["Kết quả tốt", "Kết quả trung bình", "Sai hướng"]),
        ("Thử nhỏ trước", 8, 0.85, ["Có tín hiệu tốt", "Tín hiệu mơ hồ", "Dừng sớm"]),
        ("Trì hoãn để thu thêm dữ liệu", 5, 0.9, ["Quyết định chắc hơn", "Mất cơ hội", "Giảm áp lực"]),
    ],
}


def stable_id(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def generate_decision_model(payload: dict[str, Any]) -> dict[str, Any]:
    domain = str(payload.get("domain", "life")).lower()
    templates = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS.get("life", []))
    options = []
    probabilities = [0.42, 0.36, 0.22]
    utilities = [78, 48, 16]
    if domain in {"health", "legal", "safety"}:
        utilities = [72, 38, -35]
    for name, cost, reversibility, scenarios in templates:
        options.append(
            {
                "id": stable_id(f"{domain}:{name}"),
                "name": name,
                "cost": cost,
                "reversibility": reversibility,
                "scenarios": [
                    {
                        "name": scenario,
                        "probability": probabilities[index],
                        "utility": utilities[index],
                        "evidence": "Auto-generated baseline; người dùng nên hiệu chỉnh theo dữ liệu thật.",
                    }
                    for index, scenario in enumerate(scenarios)
                ],
            }
        )
    return {
        **payload,
        "options": options,
        "context": (payload.get("context") or "") + "\n\nDecision model được auto-generate; cần hiệu chỉnh xác suất/utility sau khi có bằng chứng.",
    }
