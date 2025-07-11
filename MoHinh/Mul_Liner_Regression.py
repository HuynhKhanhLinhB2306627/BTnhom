import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

# ==== 1. Đọc & xử lý dữ liệu ====
df = pd.read_csv("energydata_complete.csv")
df['date']    = pd.to_datetime(df['date'])
df['hour']    = df['date'].dt.hour
df['day']     = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

X = df.drop(columns=['date', 'Appliances'])
y = df['Appliances']

# Xử lý giá trị thiếu
X = X.fillna(X.median())

# ==== 2. Tách train/test ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 3. Chuẩn hóa ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==== 4. Hồi quy tuyến tính đa biến (KHÔNG PCA) ====
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ==== 5. Đánh giá mô hình theo style cây quyết định ====
print("\n🎯 HỒI QUY ĐA BIẾN")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE:", np.mean(np.abs((y_test - y_pred) / y_test)) * 100, "%")
print("R2 Score:", r2_score(y_test, y_pred))
print("Explained Variance Score:", explained_variance_score(y_test, y_pred))

# ==== 6. In phương trình hồi quy ====
intercept = model.intercept_
coeffs = model.coef_
feature_names = X.columns

terms = [f"{intercept:.3f}"]
for feat, beta in zip(feature_names, coeffs):
    sign = "+" if beta >= 0 else "-"
    terms.append(f" {sign} {abs(beta):.3f}*{feat}")
equation = " ".join(terms)

print("\nPhương trình hồi quy:")
print("Appliances_hat =", equation)


import matplotlib.pyplot as plt

# ==== 7. Vẽ biểu đồ so sánh thực tế và dự đoán ====
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Thực tế", linewidth=2)
plt.plot(y_pred[:100], label="Hồi quy đa biến dự đoán", alpha=0.7)
plt.title("Hồi quy đa biến: So sánh dự đoán vs thực tế (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
