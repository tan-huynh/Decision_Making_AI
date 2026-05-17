from __future__ import annotations

from typing import Any


DOMAIN_QUESTIONS = {
    "career": [
        "Mức lương tối thiểu hoặc điều kiện tài chính bạn cần giữ là gì?",
        "Bạn ưu tiên tăng trưởng kỹ năng, thu nhập, ổn định hay tự do thời gian?",
        "Bạn có thể chịu rủi ro trong bao lâu nếu lựa chọn không hiệu quả?",
    ],
    "income": [
        "Bạn muốn tăng thêm bao nhiêu tiền mỗi tháng và trong bao lâu?",
        "Bạn có bao nhiêu giờ mỗi tuần, vốn ban đầu và kỹ năng có thể bán được là gì?",
        "Bạn thích tăng thu nhập qua lương, freelance, kinh doanh nhỏ, content, đầu tư hay sản phẩm số?",
        "Bạn có giới hạn đạo đức, pháp lý hoặc rủi ro tài chính nào không muốn vượt qua?",
    ],
    "education": [
        "Bạn đang có nền tảng kỹ năng nào và còn thiếu kỹ năng gì?",
        "Bạn muốn kết quả trong bao lâu: có việc, tăng lương, hay đổi ngành?",
        "Bạn có bao nhiêu giờ mỗi tuần và ngân sách học tối đa là bao nhiêu?",
    ],
    "finance": [
        "Số tiền đầu tư, thời hạn nắm giữ và mức lỗ tối đa bạn chịu được là bao nhiêu?",
        "Bạn đang so sánh với lựa chọn thay thế nào: tiền mặt, ETF, trái phiếu hay cổ phiếu khác?",
        "Bạn cần thanh khoản trong thời gian tới không?",
    ],
    "business": [
        "Giả thuyết lớn nhất cần kiểm chứng là nhu cầu, giá bán, phân phối hay vận hành?",
        "Chi phí thử nghiệm nhỏ nhất để có dữ liệu thật là bao nhiêu?",
        "Bạn sẽ dừng hoặc pivot theo tín hiệu nào?",
    ],
    "life": [
        "Mục tiêu thật sự quan trọng nhất của bạn trong quyết định này là gì?",
        "Điều gì là ràng buộc không thể vi phạm?",
        "Nếu chọn sai, chi phí đảo ngược là gì?",
    ],
    "wisdom": [
        "Giá trị cá nhân nào bạn không muốn đánh đổi trong quyết định này?",
        "Nếu nhìn lại sau 1 năm, điều gì sẽ khiến bạn thấy quyết định này đúng?",
        "Bạn đang sợ rủi ro thật, hay đang tránh một việc khó nhưng cần làm?",
        "Ai bị ảnh hưởng bởi quyết định này và mức độ ảnh hưởng ra sao?",
    ],
    "relationship": [
        "Kết quả bạn mong muốn là hòa giải, đặt ranh giới, hay chấm dứt tình huống?",
        "Có yếu tố an toàn thể chất/tinh thần nào cần ưu tiên không?",
        "Bạn đã thử trao đổi trực tiếp và kết quả ra sao?",
    ],
    "health": [
        "Tên thuốc, hoạt chất, liều dùng, tần suất và mục đích dùng là gì?",
        "Bạn đang dùng thuốc/thực phẩm bổ sung nào khác, có uống rượu hoặc caffeine không?",
        "Bạn có dị ứng thuốc, bệnh gan/thận/tim mạch, đang mang thai hoặc bệnh nền nào không?",
        "Triệu chứng, thời gian kéo dài và mức độ nghiêm trọng hiện tại là gì?",
        "Bạn đã được chuyên gia y tế đánh giá chưa?",
    ],
    "legal": [
        "Bạn đang ở khu vực pháp lý nào?",
        "Deadline hoặc rủi ro pháp lý cụ thể là gì?",
        "Bạn đã có tài liệu/hợp đồng/quyết định chính thức nào chưa?",
    ],
}


def build_clarifying_questions(payload: dict[str, Any], quantitative: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    context = str(payload.get("context", "")).strip()
    objective = str(payload.get("objective", "")).strip()
    domain = str(payload.get("domain", "life")).strip().lower()

    if len(context) < 80:
        questions.append("Bạn có thể mô tả thêm bối cảnh, ràng buộc và điều bạn đã biết chắc không?")
    if len(objective) < 40:
        questions.append("Tiêu chí thành công chính xác là gì và đo bằng chỉ số nào?")

    if quantitative.get("value_of_information", {}).get("score", 0) > 60:
        unknowns = quantitative["value_of_information"].get("most_valuable_unknowns", [])[:2]
        if unknowns:
            questions.append(f"Bạn có dữ liệu hoặc bằng chứng nào cho các điểm bất định này không: {', '.join(unknowns)}?")

    low_confidence = [item for item in quantitative.get("option_results", []) if item.get("confidence", 1) < 0.55]
    if low_confidence:
        questions.append("Bạn có thể ước lượng lại xác suất/utility cho lựa chọn có confidence thấp không?")

    for item in DOMAIN_QUESTIONS.get(domain, DOMAIN_QUESTIONS["life"]):
        if len(questions) >= 5:
            break
        questions.append(item)

    return list(dict.fromkeys(questions))[:5]
