import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df = df.drop(columns=['date', 'rv1', 'rv2'])

# Tách dữ liệu đầu vào (tất cả cột sau tiền xử lý trừ Appliances), đầu ra (cột Appliances)
X = df.drop(columns=['Appliances'])
y = df['Appliances']

#Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hàm đánh giá và trực quan hóa
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"=== {name} ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Trực quan hóa kết quả dự đoán
    plt.figure(figsize=(8, 4))
    plt.plot(y_test.values[:100], label='Actual', marker='o')
    plt.plot(y_pred[:100], label='Predicted', marker='x')
    plt.title(f"{name} - Actual vs Predicted (100 samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Energy (Wh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ', '_')}_result.png")
    plt.close()

# Linear Regression
evaluate_model("Linear Regression", LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test)

# Random Forest (không cần scaled)
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test)

# XGBoost
evaluate_model("XGBoost", XGBRegressor(n_estimators=100, random_state=42, verbosity=0), X_train, X_test, y_train, y_test)

print("Đã lưu hình ảnh kết quả của từng mô hình dưới dạng PNG.")
