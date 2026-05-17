import type { DecisionInput } from "./types";

export const defaultDecision: DecisionInput = {
  question: "Có nên ra mắt sản phẩm AI decision advisor trong 30 ngày tới không?",
  domain: "business",
  objective: "Tối đa hóa xác suất ra quyết định đúng với rủi ro tài chính được kiểm soát.",
  context:
    "Nguồn lực nhỏ, cần ra quyết định nhanh. Có Ollama local để phân tích định tính, và muốn ghi nhận kết quả thực tế để hiệu chỉnh xác suất sau này.",
  riskTolerance: 0.55,
  timeHorizon: "30-90 ngày",
  model: "llama3.1",
  webRealtime: true,
  options: [
    {
      id: "launch-mvp",
      name: "Ra mắt MVP có giới hạn",
      cost: 18,
      reversibility: 0.75,
      scenarios: [
        { name: "Người dùng phản hồi tốt", probability: 0.42, utility: 86, evidence: "Nhu cầu rõ, scope hẹp" },
        { name: "Adoption chậm", probability: 0.38, utility: 48, evidence: "Cần giáo dục người dùng" },
        { name: "Lỗi chất lượng phân tích", probability: 0.2, utility: 18, evidence: "Phụ thuộc prompt và dữ liệu vào" }
      ]
    },
    {
      id: "research-more",
      name: "Nghiên cứu thêm trước khi ra mắt",
      cost: 10,
      reversibility: 0.9,
      scenarios: [
        { name: "Mô hình chính xác hơn", probability: 0.5, utility: 68, evidence: "Có thêm benchmark" },
        { name: "Mất momentum", probability: 0.3, utility: 35, evidence: "Không có feedback thật" },
        { name: "Phát hiện rủi ro lớn", probability: 0.2, utility: 58, evidence: "Tránh build sai hướng" }
      ]
    },
    {
      id: "manual-consulting",
      name: "Làm dịch vụ tư vấn thủ công trước",
      cost: 24,
      reversibility: 0.55,
      scenarios: [
        { name: "Hiểu sâu workflow khách hàng", probability: 0.45, utility: 78, evidence: "Tương tác trực tiếp" },
        { name: "Không scale được", probability: 0.35, utility: 36, evidence: "Tốn thời gian founder" },
        { name: "Có doanh thu sớm", probability: 0.2, utility: 82, evidence: "Bán được gói tư vấn" }
      ]
    }
  ]
};
