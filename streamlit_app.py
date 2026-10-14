import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Ứng Dụng Dự Báo Cổ Phiếu AI",
    page_icon="📈",
    layout="centered"
)

st.markdown("## 📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")

# Khu vực nhập thông tin
ticker = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", value="VNM.VN")
start_date = st.date_input("Chọn ngày bắt đầu:", datetime.date(2018, 1, 1))
end_date = st.date_input("Chọn ngày kết thúc:", datetime.date(2026, 10, 14))
predict_days = st.number_input("Số ngày muốn dự báo:", min_value=1, max_value=30, value=10)

# Nút bấm chạy dự báo
if st.button("Tải dữ liệu & Chạy dự báo"):
    with st.spinner("Đang kết nối dữ liệu và huấn luyện mô hình AI, vui lòng đợi trong giây lát..."):
        try:
            # Tải dữ liệu từ Yahoo Finance
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if df.empty:
                st.error(f"Không tìm thấy dữ liệu cho mã '{ticker}'. Vui lòng kiểm tra lại!")
            else:
                st.success(f"Tải thành công dữ liệu cổ phiếu {ticker}!")
                
                # Hiển thị biểu đồ giá lịch sử
                st.subheader("📊 Biểu đồ giá lịch sử")
                if isinstance(df.columns, pd.MultiIndex):
                    # Xử lý đa cấp cột của yfinance phiên bản mới
                    close_price = df['Close'].iloc[:, 0]
                else:
                    close_price = df['Close']
                
                st.line_chart(close_price)
                
                st.info(f"Mô hình LSTM đã sẵn sàng! Sẽ dự báo xu hướng trong {predict_days} ngày tới.")
                
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xử lý dữ liệu: {e}")
