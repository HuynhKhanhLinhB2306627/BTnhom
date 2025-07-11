import pandas as pd
from sklearn.linear_model import LinearRegression
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

# ==== 2. Chia dữ liệu ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 3. Chuẩn hóa ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==== 4. Hồi quy tuyến tính - KHÔNG dùng PCA ====
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mae_no_pca = mean_absolute_error(y_test, y_pred)
mse_no_pca = mean_squared_error(y_test, y_pred)
r2_no_pca  = r2_score(y_test, y_pred)

print("🔹 HỒI QUY TUYẾN TÍNH (KHÔNG PCA)")
print("MAE:", round(mae_no_pca, 3))
print("MSE:", round(mse_no_pca, 3))
print("R² :",  round(r2_no_pca, 3))

# ==== 5. Hồi quy tuyến tính - CÓ PCA ====
pca = PCA(n_components=0.95)  # giữ lại 95% phương sai
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)

mae_pca = mean_absolute_error(y_test, y_pred_pca)
mse_pca = mean_squared_error(y_test, y_pred_pca)
r2_pca  = r2_score(y_test, y_pred_pca)

print("\n🔸 HỒI QUY TUYẾN TÍNH (CÓ PCA)")
print("MAE:", round(mae_pca, 3))
print("MSE:", round(mse_pca, 3))
print("R² :",  round(r2_pca, 3))
