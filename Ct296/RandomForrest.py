import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==== 1. Đọc & xử lý dữ liệu ====
df = pd.read_csv("energydata_complete.csv")
df['date']    = pd.to_datetime(df['date'])
df['hour']    = df['date'].dt.hour
df['day']     = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

X = df.drop(columns=['date', 'Appliances'])
y = df['Appliances']
X = X.fillna(X.median())

# ==== 2. Tách & chuẩn hóa ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==== 3. Random Forest - KHÔNG PCA ====
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("🔹 RANDOM FOREST (KHÔNG PCA)")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
print("R² :",  round(r2_score(y_test, y_pred), 3))

# ==== 4. VẼ BIỂU ĐỒ - 100 điểm đầu tiên ====
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế", marker='o')
plt.plot(y_pred[:100], label="Dự đoán", marker='x', alpha=0.7)
plt.title(" Random Forest - So sánh 100 điểm đầu tiên (Không PCA)")
plt.xlabel("Chỉ số điểm dữ liệu")
plt.ylabel("Năng lượng tiêu thụ (Wh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== 5. Random Forest - CÓ PCA ====
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

rf_pca = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)

print("\n🔸 RANDOM FOREST (CÓ PCA)")
print("MAE:", round(mean_absolute_error(y_test, y_pred_pca), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred_pca), 3))
print("R² :",  round(r2_score(y_test, y_pred_pca), 3))

# ==== 6. VẼ BIỂU ĐỒ - Có PCA ====
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế", marker='o')
plt.plot(y_pred_pca[:100], label="Dự đoán (PCA)", marker='x', alpha=0.7)
plt.title(" Random Forest - So sánh 100 điểm đầu tiên (Có PCA)")
plt.xlabel("Chỉ số điểm dữ liệu")
plt.ylabel("Năng lượng tiêu thụ (Wh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
