import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Hệ thống đánh giá rủi ro tín dụng",
    layout="wide"
)

# =============================
# STYLE
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8f5f0, #f1ede6);
    color: #1f2937;
}
.main-title {
    font-size: 30px;
    font-weight: 700;
    color: #1e3a8a;
}
.subtitle {
    color: #6b7280;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}
.low { color: #16a34a; }
.high { color: #dc2626; }
.stButton>button {
    background: #1e3a8a;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    with open('xgboost_dashboard_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# =============================
# HEADER
# =============================
st.markdown('<div class="main-title">HỆ THỐNG ĐÁNH GIÁ RỦI RO TÍN DỤNG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dự đoán khả năng vỡ nợ của khách hàng</div>', unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Thông tin khách hàng")

fico_score = st.sidebar.slider("Điểm FICO", 300, 850, 700)

annual_income = st.sidebar.number_input(
    "Thu nhập ($/năm)",
    1000, 1_000_000, 50000, step=1000
)

loan_amount = st.sidebar.slider("Khoản vay ($)", 500, 40000, 10000)

term = st.sidebar.selectbox("Kỳ hạn (tháng)", [36, 60])

dti = st.sidebar.slider("DTI (%)", 0.0, 40.0, 15.0)

int_rate = st.sidebar.slider("Lãi suất (%)", 5.0, 30.0, 12.0)

grade = st.sidebar.selectbox("Xếp hạng tín dụng", ["A","B","C","D","E","F","G"])

# =============================
# ENCODE FEATURE
# =============================
grade_map = {
    "A":1, "B":2, "C":3,
    "D":4, "E":5, "F":6, "G":7
}

grade_encoded = grade_map[grade]

# Đảm bảo đúng thứ tự feature khi train
features = np.array([[
    int_rate,
    loan_amount,
    annual_income,
    term,
    dti,
    fico_score,
    grade_encoded
]])

# =============================
# LAYOUT
# =============================
col1, col2 = st.columns([1, 1.2])

# ========= LEFT =========
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Dữ liệu đầu vào")

    st.write(f"FICO: {fico_score}")
    st.write(f"Thu nhập: ${annual_income:,}")
    st.write(f"Khoản vay: ${loan_amount:,}")
    st.write(f"Kỳ hạn: {term} tháng")
    st.write(f"DTI: {dti}%")
    st.write(f"Lãi suất: {int_rate}%")
    st.write(f"Grade: {grade}")

    predict = st.button("Phân tích")

    st.markdown('</div>', unsafe_allow_html=True)

# ========= RIGHT =========
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Kết quả")

    if predict:
        prob = model.predict_proba(features)[0][1]

        # ===== DECISION LOGIC =====
        threshold = 0.45

        risk_label = "Cao" if prob >= threshold else "Thấp"
        decision = "Từ chối" if prob >= threshold else "Chấp nhận"
        color = "high" if prob >= threshold else "low"

        c1, c2, c3 = st.columns(3)

        c1.metric("Xác suất vỡ nợ", f"{prob:.2%}")
        c2.markdown(f"<div class='metric-value {color}'>{risk_label}</div><div>Mức rủi ro</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-value'>{decision}</div><div>Quyết định</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ===== GAUGE =====
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1e3a8a"},
                'steps': [
                    {'range': [0, 30], 'color': "#dcfce7"},
                    {'range': [30, 50], 'color': "#fef3c7"},
                    {'range': [50, 100], 'color': "#fee2e2"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ===== INSIGHT =====
        st.markdown("### Nhận định")

        if prob < 0.3:
            st.success("Khách hàng rất an toàn, khả năng trả nợ cao.")
        elif prob < 0.5:
            st.warning("Rủi ro trung bình, nên xem xét thêm.")
        else:
            st.error("Rủi ro cao, khả năng vỡ nợ lớn.")

    else:
        st.info("Nhập thông tin và nhấn 'Phân tích'")

    st.markdown('</div>', unsafe_allow_html=True)
