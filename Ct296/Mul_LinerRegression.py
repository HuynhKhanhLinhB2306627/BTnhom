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

# ==== 2. Tách train/test ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 3. Chuẩn hóa ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==== 4. Hồi quy đa biến KHÔNG dùng PCA ====
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("🔹 HỒI QUY ĐA BIẾN (KHÔNG PCA)")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
print("R² :",  round(r2_score(y_test, y_pred), 3))

# In phương trình hồi quy với đặc trưng gốc
intercept = model.intercept_
coeffs = model.coef_
feature_names = X.columns

terms = [f"{intercept:.3f}"]
for feat, beta in zip(feature_names, coeffs):
    sign = "+" if beta >= 0 else "-"
    terms.append(f" {sign} {abs(beta):.3f}*{feat}")
equation = " ".join(terms)

print("\nPhương trình hồi quy KHÔNG PCA:")
print("Appliances_hat =", equation)

# ==== 5. Áp dụng PCA ====
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)

print("\n🔸 HỒI QUY ĐA BIẾN (CÓ PCA)")
print("MAE:", round(mean_absolute_error(y_test, y_pred_pca), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred_pca), 3))
print("R² :",  round(r2_score(y_test, y_pred_pca), 3))

# In phương trình hồi quy sau PCA
intercept_pca = model_pca.intercept_
coeffs_pca = model_pca.coef_
pc_names = [f"PC{i}" for i in range(len(coeffs_pca))]

terms_pca = [f"{intercept_pca:.3f}"]
for pc, beta in zip(pc_names, coeffs_pca):
    sign = "+" if beta >= 0 else "-"
    terms_pca.append(f" {sign} {abs(beta):.3f}*{pc}")
equation_pca = " ".join(terms_pca)

print("\nPhương trình hồi quy CÓ PCA:")
print("Appliances_hat =", equation_pca)
