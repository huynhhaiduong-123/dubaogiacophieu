import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import random

import tensorflow as tf



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Input

from tensorflow.keras.callbacks import EarlyStopping



# =========================================================

# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN

# =========================================================

st.set_page_config(

    page_title="StockVision AI",

    page_icon="📈",

    layout="centered"

)



st.title("📈 StockVision AI")

st.subheader("Ứng dụng dự báo giá cổ phiếu bằng trí tuệ nhân tạo (LSTM)")

st.write(

    "Nhập mã cổ phiếu, chọn khoảng thời gian và số phiên cần dự báo. "

    "Ứng dụng sẽ tải dữ liệu lịch sử, huấn luyện mô hình và hiển thị kết quả."

)

st.divider()



# =========================================================

# 2. KHU VỰC NHẬP THÔNG TIN

# =========================================================

with st.form("stock_prediction_form"):

    ticker = st.text_input(

        "Mã cổ phiếu",

        value="VNM.VN",

        placeholder="Ví dụ: VNM.VN, FPT.VN, AAPL"

    ).strip().upper()



    col1, col2 = st.columns(2)

    with col1:

        start_date = st.date_input(

            "Ngày bắt đầu",

            value=pd.Timestamp("2018-01-01").date(),

            max_value=pd.Timestamp.today().date()

        )

    with col2:

        end_date = st.date_input(

            "Ngày kết thúc",

            value=pd.Timestamp.today().date(),

            max_value=pd.Timestamp.today().date()

        )



    forecast_days = st.number_input(

        "Số phiên giao dịch muốn dự báo",

        min_value=1,

        max_value=60,

        value=10,

        step=1

    )



    submitted = st.form_submit_button(

        "📥 Tải dữ liệu & Chạy dự báo",

        use_container_width=True

    )



# =========================================================

# 3. HÀM CHUẨN BỊ DỮ LIỆU

# =========================================================

def create_sequences(values, window_size=60):

    X, y = [], []

    for i in range(window_size, len(values)):

        X.append(values[i - window_size:i])

        y.append(values[i])

    return np.asarray(X), np.asarray(y)





# =========================================================

# 4. TẢI DỮ LIỆU, HUẤN LUYỆN VÀ DỰ BÁO

# =========================================================

if submitted:

    if not ticker:

        st.error("Vui lòng nhập mã cổ phiếu.")

        st.stop()



    if start_date >= end_date:

        st.error("Ngày bắt đầu phải trước ngày kết thúc.")

        st.stop()



    try:
        with st.spinner("Đang tải dữ liệu và huấn luyện mô hình LSTM..."):

            # Cố định seed để tăng khả năng tái lập kết quả

            seed = 42

            np.random.seed(seed)

            random.seed(seed)

            tf.random.set_seed(seed)



            df = yf.download(

                ticker,

                start=start_date.strftime("%Y-%m-%d"),

                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),

                progress=False,

                auto_adjust=False

            )



            if df.empty:

                st.error(

                    "Không tải được dữ liệu. Hãy kiểm tra mã cổ phiếu "

                    "và khoảng thời gian đã chọn."

                )

                st.stop()



            # Lấy cột giá đóng cửa; hỗ trợ cả định dạng cột MultiIndex

            close = df["Close"]

            if isinstance(close, pd.DataFrame):

                close = close.iloc[:, 0]



            close = close.dropna()

            data = close.to_numpy(dtype=np.float32).reshape(-1, 1)

            dates = pd.to_datetime(close.index)



            window_size = 60

            if len(data) < window_size + 30:

                st.error(

                    "Khoảng thời gian này có quá ít dữ liệu. "

                    "Vui lòng chọn khoảng ngày dài hơn."

                )

                st.stop()



            # Chia dữ liệu theo thứ tự thời gian

            train_size = int(0.8 * len(data))

            train_data = data[:train_size]

            test_data = data[train_size:]



            scaler = MinMaxScaler()

            train_scaled = scaler.fit_transform(train_data)

            test_scaled = scaler.transform(test_data)



            X_train, y_train = create_sequences(train_scaled, window_size)



            # Tạo tập test có thêm 60 phiên trước tập test để làm đầu vào

            test_combined = np.vstack([train_scaled[-window_size:], test_scaled])

            X_test, y_test = create_sequences(test_combined, window_size)



            if len(X_train) < 20 or len(X_test) == 0:

                st.error("Không đủ dữ liệu để huấn luyện hoặc đánh giá mô hình.")

                st.stop()



            # Tách phần cuối của tập train để kiểm định

            val_size = max(1, int(0.1 * len(X_train)))

            X_val, y_val = X_train[-val_size:], y_train[-val_size:]

            X_train_final, y_train_final = X_train[:-val_size], y_train[:-val_size]



            tf.keras.backend.clear_session()

            model = Sequential([

                Input(shape=(window_size, 1)),

                LSTM(50, return_sequences=True),

                LSTM(50),

                Dense(1)

            ])

            model.compile(optimizer="adam", loss="mean_squared_error")

            early_stop = EarlyStopping(

                monitor="val_loss",

                patience=5,

                restore_best_weights=True

            )



            model.fit(

                X_train_final,

                y_train_final,

                validation_data=(X_val, y_val),

                epochs=10,

                batch_size=32,

                shuffle=False,

                callbacks=[early_stop],

                verbose=0

            )



            # Dự báo trên tập kiểm tra để tính chỉ số đánh giá

            test_pred_scaled = model.predict(X_test, verbose=0)

            test_pred = scaler.inverse_transform(test_pred_scaled).ravel()

            y_test_real = scaler.inverse_transform(y_test).ravel()



            rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))

            mae = float(mean_absolute_error(y_test_real, test_pred))

            r2 = (

                float(r2_score(y_test_real, test_pred))

                if len(y_test_real) > 1 else float("nan")

            )



            # Dự báo nối tiếp cho các phiên tiếp theo

            last_window = scaler.transform(data[-window_size:]).reshape(

                1, window_size, 1

            )

            future_scaled = []



            for _ in range(int(forecast_days)):

                next_value = model.predict(last_window, verbose=0)

                future_scaled.append(float(next_value[0, 0]))

                last_window = np.concatenate(

                    [last_window[:, 1:, :], next_value.reshape(1, 1, 1)],

                    axis=1

                )



            future_prices = scaler.inverse_transform(

                np.asarray(future_scaled).reshape(-1, 1)

            ).ravel()



            # Ngày dự báo là ngày làm việc dự kiến, không gồm cuối tuần

            future_dates = pd.bdate_range(

                start=pd.Timestamp(dates[-1]) + pd.offsets.BDay(1),

                periods=int(forecast_days)

            )



        # =====================================================

        # 5. HIỂN THỊ KẾT QUẢ (KHÔNG VẼ BIỂU ĐỒ)

        # =====================================================

        st.success("Dự báo hoàn tất!")



        st.header(f"📌 Kết quả dự báo: {ticker}")

        st.caption(

            f"Dữ liệu từ {dates[0].date()} đến {dates[-1].date()} "

            f"({len(data):,} phiên giao dịch có dữ liệu)."

        )



        last_price = float(data[-1, 0])

        next_price = float(future_prices[0])

        percent_change = (

            (next_price - last_price) / last_price * 100

            if last_price != 0 else 0

        )



        col1, col2 = st.columns(2)

        with col1:

            st.metric("Giá đóng cửa gần nhất", f"{last_price:,.2f}")

        with col2:

            st.metric(

                "Giá dự báo phiên kế tiếp", f"{next_price:,.2f}",

                f"{percent_change:+.2f}%"

            )



        st.subheader("Đánh giá mô hình trên tập kiểm tra")

        col1, col2, col3 = st.columns(3)

        col1.metric("RMSE", f"{rmse:,.3f}")

        col2.metric("MAE", f"{mae:,.3f}")

        col3.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")



        st.subheader(f"🔮 Bảng dự báo {int(forecast_days)} phiên tiếp theo")

        result = pd.DataFrame({

            "Ngày dự báo": future_dates.strftime("%d/%m/%Y"),

            "Phiên": [f"T+{i}" for i in range(1, int(forecast_days) + 1)],

            "Giá dự báo": np.round(future_prices, 2)

        })

        st.dataframe(result, use_container_width=True, hide_index=True)



        st.download_button(

            "⬇️ Tải bảng dự báo CSV",

            data=result.to_csv(index=False).encode("utf-8-sig"),

            file_name=f"stockvision_{ticker.replace('.', '_')}.csv",

            mime="text/csv",

            use_container_width=True

        )



        st.info(

            "Lưu ý: Kết quả là ước lượng từ dữ liệu giá lịch sử, không bảo đảm "

            "giá thực tế sẽ diễn biến như dự báo và không phải khuyến nghị đầu tư."

        )



    except Exception as e:

        st.error(f"Đã xảy ra lỗi khi chạy dự báo: {e}")

        st.write(

            "Bạn hãy kiểm tra kết nối Internet, mã cổ phiếu và các thư viện "

            "đã được cài đặt trong requirements.txt."

        )

else:

    st.captio
