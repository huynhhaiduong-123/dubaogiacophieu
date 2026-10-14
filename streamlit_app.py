import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 2. CỐ ĐỊNH SEED (Đảm bảo kết quả nhất quán giữa các lần chạy)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 3. TẢI DỮ LIỆU
TICKER = "VNM.VN"
START_DATE = "2018-01-01"
END_DATE = "2026-10-14"

print(f"Đang tải dữ liệu cổ phiếu {TICKER}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
data = df[['Close']].values

# 4. CHIA TẬP DỮ LIỆU (Train 70% - Validation 10% - Test 20%)
train_size = int(0.7 * len(data))
val_size = int(0.1 * len(data))

train_raw = data[:train_size]
val_raw = data[train_size : train_size + val_size]
test_raw = data[train_size + val_size:]

# 5. CHUẨN HÓA DỮ LIỆU (Tránh rò rỉ dữ liệu - Fit chỉ trên Train)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
val_scaled = scaler.transform(val_raw)
test_scaled = scaler.transform(test_raw)

# 6. TẠO TẬP CHUỖI THỜI GIAN (LOOKBACK WINDOW = 60)
WINDOW_SIZE = 60

def create_sequences(data_array, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(window_size, len(data_array)):
        X.append(data_array[i-window_size:i])
        y.append(data_array[i])
    return np.array(X), np.array(y)

# Ghép 60 ngày cuối của tập trước vào tập sau để giữ đủ chiều dài cửa sổ
val_combined = np.vstack((train_scaled[-WINDOW_SIZE:], val_scaled))
test_combined = np.vstack((val_scaled[-WINDOW_SIZE:], test_scaled))

X_train, y_train = create_sequences(train_scaled)
X_val, y_val = create_sequences(val_combined)
X_test, y_test = create_sequences(test_combined)

# Reshape dạng (samples, time_steps, features) cho LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Hình dạng X_train: {X_train.shape}")
print(f"Hình dạng X_val:   {X_val.shape}")
print(f"Hình dạng X_test:  {X_test.shape}")

# 7. XÂY DỰNG MÔ HÌNH LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 8. HUẤN LUYỆN MÔ HÌNH
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    shuffle=False,  # Chuỗi thời gian không shuffle
    callbacks=[early_stop],
    verbose=1
)

# 9. DỰ BÁO TẬP TEST VÀ ĐÁNH GIÁ
test_pred = model.predict(X_test)

# Đưa dữ liệu về đơn vị giá thực tế (Inverse Transform)
test_pred_real = scaler.inverse_transform(test_pred)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# Tính toán chỉ số đo lường
rmse = np.sqrt(mean_squared_error(y_test_real, test_pred_real))
mae = mean_absolute_error(y_test_real, test_pred_real)
r2 = r2_score(y_test_real, test_pred_real)

print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R2   : {r2:.4f}")

# 10. DỰ BÁO 10 NGÀY TƯƠNG LAI (Cuốn chiếu - Auto-regressive)
all_scaled_data = np.vstack((train_scaled, val_scaled, test_scaled))
last_60_days = all_scaled_data[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)

future_predictions = []
N_FUTURE_DAYS = 10

for _ in range(N_FUTURE_DAYS):
    next_pred = model.predict(last_60_days, verbose=0)
    future_predictions.append(next_pred[0, 0])
    
    # Cập nhật cửa sổ 60 ngày: Bỏ ngày cũ nhất, thêm giá trị vừa dự báo vào
    last_60_days = np.append(
        last_60_days[:, 1:, :],
        next_pred.reshape(1, 1, 1),
        axis=1
    )

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

print(f"\n=== DỰ BÁO GIÁ {N_FUTURE_DAYS} NGÀY TỚI ===")
for i, price in enumerate(future_predictions, 1):
    print(f"Ngày +{i}: {price[0]:,.2f} VND")

# 11. TRỰC QUAN HÓA KẾT QUẢ
plt.figure(figsize=(14, 6))

# Đồ thị 1: Train vs Val Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Hàm mất mát (Loss) qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Đồ thị 2: Giá thực tế vs Giá dự báo + 10 ngày tương lai
plt.subplot(1, 2, 2)
plt.plot(y_test_real, label='Giá thực tế', color='blue')
plt.plot(test_pred_real, label='Giá dự báo Test', color='orange')

future_x = range(len(y_test_real), len(y_test_real) + N_FUTURE_DAYS)
plt.plot(future_x, future_predictions, label=f'Dự báo {N_FUTURE_DAYS} ngày tới', color='green', marker='o')

plt.title(f'Dự báo giá cổ phiếu {TICKER}')
plt.xlabel('Thời gian (Ngày trong tập Test)')
plt.ylabel('Giá đóng cửa (VND)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
