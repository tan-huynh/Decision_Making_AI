"""Bài 1 — Xếp hàng hóa máy bay (Cargo Loading LP)

12 biến: x_ij = tấn hàng loại i xếp vào khoang j
i = 1..4 (loại hàng), j = 1..3 (Trước, Trung tâm, Sau)
"""
import sys
sys.path.insert(0, "backend")
from scipy.optimize import linprog
import numpy as np

# ═══════════════════════════════════════════════════════
# 1. BIẾN QUYẾT ĐỊNH
# ═══════════════════════════════════════════════════════
# x[0]=x11  x[1]=x12  x[2]=x13   → Hàng loại 1 ở khoang Trước, TT, Sau
# x[3]=x21  x[4]=x22  x[5]=x23   → Hàng loại 2
# x[6]=x31  x[7]=x32  x[8]=x33   → Hàng loại 3
# x[9]=x41  x[10]=x42 x[11]=x43  → Hàng loại 4

names = [
    "x11", "x12", "x13",  # loại 1: Trước, TT, Sau
    "x21", "x22", "x23",  # loại 2
    "x31", "x32", "x33",  # loại 3
    "x41", "x42", "x43",  # loại 4
]

# Dữ liệu
weight_limit = [9, 15, 8]          # Tấn: Trước, TT, Sau
volume_limit = [6000, 8000, 5000]   # m³: Trước, TT, Sau
available    = [12, 14, 21, 10]     # Tấn hàng sẵn có
volume_rate  = [460, 680, 570, 380] # m³/tấn
profit_rate  = [510, 680, 550, 585] # $/tấn

# ═══════════════════════════════════════════════════════
# 2. HÀM MỤC TIÊU: Maximize Z = Σ profit_i × Σ_j x_ij
# ═══════════════════════════════════════════════════════
# linprog minimizes, so negate for maximize
c = []
for i in range(4):
    for j in range(3):
        c.append(-profit_rate[i])  # negative for maximize

# ═══════════════════════════════════════════════════════
# 3. RÀNG BUỘC BẤT ĐẲNG THỨC (A_ub × x ≤ b_ub)
# ═══════════════════════════════════════════════════════
A_ub = []
b_ub = []

# (a) Trọng lượng mỗi khoang: Σ_i x_ij ≤ W_j
for j in range(3):
    row = [0.0] * 12
    for i in range(4):
        row[i * 3 + j] = 1.0
    A_ub.append(row)
    b_ub.append(weight_limit[j])

# (b) Thể tích mỗi khoang: Σ_i v_i × x_ij ≤ V_j
for j in range(3):
    row = [0.0] * 12
    for i in range(4):
        row[i * 3 + j] = volume_rate[i]
    A_ub.append(row)
    b_ub.append(volume_limit[j])

# (c) Tổng hàng loại i: Σ_j x_ij ≤ available_i
for i in range(4):
    row = [0.0] * 12
    for j in range(3):
        row[i * 3 + j] = 1.0
    A_ub.append(row)
    b_ub.append(available[i])

# ═══════════════════════════════════════════════════════
# 4. RÀNG BUỘC ĐẲNG THỨC: Cân bằng tỉ lệ
# ═══════════════════════════════════════════════════════
# Σ_i x_i1 / 9 = Σ_i x_i2 / 15  →  15·Σx_i1 - 9·Σx_i2 = 0
# Σ_i x_i2 / 15 = Σ_i x_i3 / 8  →  8·Σx_i2 - 15·Σx_i3 = 0
A_eq = []
b_eq = []

# Balance 1: 15·(khoang Trước) = 9·(khoang TT)
row1 = [0.0] * 12
for i in range(4):
    row1[i * 3 + 0] = 15.0   # khoang Trước (j=0)
    row1[i * 3 + 1] = -9.0   # khoang TT (j=1)
A_eq.append(row1)
b_eq.append(0.0)

# Balance 2: 8·(khoang TT) = 15·(khoang Sau)
row2 = [0.0] * 12
for i in range(4):
    row2[i * 3 + 1] = 8.0    # khoang TT (j=1)
    row2[i * 3 + 2] = -15.0  # khoang Sau (j=2)
A_eq.append(row2)
b_eq.append(0.0)

# ═══════════════════════════════════════════════════════
# 5. GIỚI HẠN BIẾN: x_ij ≥ 0
# ═══════════════════════════════════════════════════════
bounds = [(0, None)] * 12

# ═══════════════════════════════════════════════════════
# 6. GIẢI
# ═══════════════════════════════════════════════════════
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

print("=" * 60)
print("BÀI 1: XẾP HÀNG HÓA MÁY BAY — NGHIỆM TỐI ƯU")
print("=" * 60)
print(f"\nStatus: {result.message}")
print(f"Lợi nhuận tối đa Z* = ${-result.fun:,.2f}")

print("\n── BIẾN QUYẾT ĐỊNH ──")
print(f"{'Biến':>6} {'Loại':>6} {'Khoang':>12} {'Giá trị (tấn)':>15}")
print("-" * 45)
khoang_names = ["Trước", "Trung tâm", "Sau"]
x = result.x
for i in range(4):
    for j in range(3):
        idx = i * 3 + j
        if x[idx] > 1e-6:
            print(f"  x{i+1}{j+1}   {i+1:>4}   {khoang_names[j]:>12}   {x[idx]:>13.4f}")

print("\n── TỔNG HỢP THEO KHOANG ──")
for j, name in enumerate(khoang_names):
    total_w = sum(x[i * 3 + j] for i in range(4))
    total_v = sum(volume_rate[i] * x[i * 3 + j] for i in range(4))
    ratio = total_w / weight_limit[j] if weight_limit[j] > 0 else 0
    print(f"  {name:>12}: {total_w:8.4f} tấn / {weight_limit[j]} tấn (tỉ lệ {ratio:.4f}), "
          f"thể tích {total_v:8.1f} / {volume_limit[j]} m³")

print("\n── TỔNG HỢP THEO LOẠI HÀNG ──")
for i in range(4):
    total = sum(x[i * 3 + j] for j in range(3))
    print(f"  Loại {i+1}: {total:8.4f} / {available[i]} tấn, lợi nhuận = ${total * profit_rate[i]:,.2f}")

print("\n── SHADOW PRICES (Giá trị cận biên) ──")
if hasattr(result, "ineq") and result.ineq is not None:
    try:
        marginals = result.ineq.marginals
        constraint_names = (
            [f"Trọng lượng {n}" for n in khoang_names] +
            [f"Thể tích {n}" for n in khoang_names] +
            [f"Hàng loại {i+1}" for i in range(4)]
        )
        for name, val in zip(constraint_names, marginals):
            if abs(val) > 1e-6:
                print(f"  {name:>20}: {-val:>10.4f} $/tấn")
    except Exception:
        pass

print("\n── KIỂM TRA CÂN BẰNG ──")
ratios = []
for j in range(3):
    total = sum(x[i * 3 + j] for i in range(4))
    ratios.append(total / weight_limit[j])
    print(f"  {khoang_names[j]:>12}: {total:.4f} / {weight_limit[j]} = {total/weight_limit[j]:.6f}")
print(f"  → Các tỉ lệ bằng nhau: {'✓ Đúng' if abs(max(ratios) - min(ratios)) < 1e-6 else '✗ Sai'}")
