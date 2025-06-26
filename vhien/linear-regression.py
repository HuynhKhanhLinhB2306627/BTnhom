
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("energydata_complete.csv")

# 2. Tiền xử lý dữ liệu
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df = df.drop(columns=['date', 'rv1', 'rv2'])

# 3. Tách dữ liệu đầu vào (features) và đầu ra (target)
X = df.drop(columns=['Appliances'])
y = df['Appliances']

# 4. Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Chuẩn hóa dữ liệu với StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Huấn luyện mô hình Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7. Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test_scaled)

# 8. Tính các chỉ số đánh giá
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 9. In kết quả
print("==== ĐÁNH GIÁ MÔ HÌNH LINEAR REGRESSION ====")
print(f"MAE  (Mean Absolute Error): {mae:.2f} Wh")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} Wh")
print(f"R² Score (Coefficient of Determination): {r2:.2f}")
