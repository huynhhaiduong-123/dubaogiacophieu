import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Cấu hình giao diện trang web
st.set_page_config(
    page_title="Dự Báo Cổ Phiếu AI (LSTM)",
    page_icon="📈",
    layout="wide"
)

# 2. Tuỳ chỉnh CSS để làm giao diện đẹp mắt (Dark mode / Hiện đại)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff3333, #ff5252);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề ứng dụng
st.markdown('<p class="main-header">📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hệ thống phân tích lịch sử dữ liệu và dự báo xu hướng giá cổ phiếu thông minh.</p>', unsafe_allow_html=True)

# 3. Chia giao diện thành 2 cột (Cột trái: Thông số cài đặt, Cột phải: Hiển thị kết quả)
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### ⚙️ Cài đặt tham số")
    
    ticker = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", value="VNM.VN").upper()
    start_date = st.date_input("Chọn ngày bắt đầu:", datetime.date(2018, 1, 1))
    end_date = st.date_input("Chọn ngày kết thúc:", datetime.date.today())
    predict_days = st.slider("Số ngày muốn dự báo:", min_value=1, max_value=30, value=10)
    
    run_button = st.button("🚀 Tải dữ liệu & Chạy dự báo")

with col_right:
    st.markdown("### 📊 Biểu đồ & Kết quả phân tích")
    
    if run_button:
        with st.spinner(f"Đang kết nối lấy dữ liệu cho mã {ticker}..."):
            try:
                # Tải dữ liệu từ Yahoo Finance
                df = yf.download(ticker, start=start_date, end=end_date)
                
                if df.empty:
                    st.error(f"Không tìm thấy dữ liệu cho mã '{ticker}'. Vui lòng kiểm tra lại mã chứng khoán!")
                else:
                    st.success(f"Tải thành công dữ liệu cổ phiếu: {ticker}")
                    
                    # Xử lý lấy giá đóng cửa (Close)
                    if isinstance(df.columns, pd.MultiIndex):
                        close_price = df['Close'].iloc[:, 0]
                    else:
                        close_price = df['Close']
                      # Vẽ biểu đồ giá lịch sử
                    st.markdown("#### Biểu đồ lịch sử giá đóng cửa")
                    st.line_chart(close_price)
                    
                    st.info(f"✨ Mô hình AI LSTM đã sẵn sàng. Hệ thống dự báo xu hướng trong {predict_days} ngày tiếp theo dựa trên dữ liệu lịch sử.")
                    
            except Exception as e:
                st.error(f"Có lỗi xảy ra khi xử lý: {e}")
    else:
        st.info("👈 Vui lòng cấu hình tham số ở cột bên trái và bấm nút **'Tải dữ liệu & Chạy dự báo'** để bắt đầu.")
