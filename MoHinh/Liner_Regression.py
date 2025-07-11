import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")

# Xử lý thời gian
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['hour'] = df['date'].dt.hour
df['week'] = df['date'].dt.isocalendar().week
df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1

# Đặc trưng và mục tiêu
X = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])
y = df['Appliances']

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Hồi quy tuyến tính KHÔNG PCA ===
model1 = LinearRegression()
model1.fit(X_train_scaled, y_train)
y_pred1 = model1.predict(X_test_scaled)

# === Hồi quy tuyến tính CÓ PCA ===
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model2 = LinearRegression()
model2.fit(X_train_pca, y_train)
y_pred2 = model2.predict(X_test_pca)

# === Hàm đánh giá ===
def get_metrics(y_test, y_pred):
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE (%)": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        "R² Score": r2_score(y_test, y_pred),
        "Explained Variance": explained_variance_score(y_test, y_pred)
    }

# Tạo bảng đánh giá theo chiều dọc
metrics1 = get_metrics(y_test, y_pred1)
metrics2 = get_metrics(y_test, y_pred2)


print("\n🎯 Linear Regression (KHÔNG PCA)")
print("MAE:", round(metrics1["MAE"], 3))
print("MSE:", round(metrics1["MSE"], 3))
print("RMSE:", round(metrics1["RMSE"], 3))
print("MAPE:", round(metrics1["MAPE (%)"], 2), "%")
print("R2 Score:", round(metrics1["R² Score"], 3)) 
print("Explained Variance Score:", round(metrics1["Explained Variance"], 3))

print("\n🎯 Linear Regression (CÓ PCA)")
print("MAE:", round(metrics2["MAE"], 3))
print("MSE:", round(metrics2["MSE"], 3))
print("RMSE:", round(metrics2["RMSE"], 3))
print("MAPE:", round(metrics2["MAPE (%)"], 2), "%")
print("R2 Score:", round(metrics2["R² Score"], 3)) 
print("Explained Variance Score:", round(metrics2["Explained Variance"], 3))


# Vẽ biểu đồ so sánh
# === Vẽ biểu đồ: KHÔNG PCA ===
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Thực tế", linewidth=2)
plt.plot(y_pred1[:100], label="Dự đoán (không PCA)", alpha=0.7)
plt.title("Linear Regression - Không PCA (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Vẽ biểu đồ: CÓ PCA ===
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Thực tế", linewidth=2)
plt.plot(y_pred2[:100], label="Dự đoán (PCA)", alpha=0.7)
plt.title("Linear Regression - Có PCA (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

