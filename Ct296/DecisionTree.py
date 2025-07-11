import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

feature_names = X.columns.tolist()

# ==== 2. Tách dữ liệu ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 3. Chuẩn hóa ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==== 4. Decision Tree - KHÔNG PCA ====
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train_scaled, y_train)
y_pred_tree = tree_model.predict(X_test_scaled)

print("🔹 DECISION TREE (KHÔNG PCA)")
print("MAE:", round(mean_absolute_error(y_test, y_pred_tree), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred_tree), 3))
print("R² :",  round(r2_score(y_test, y_pred_tree), 3))

# ==== 5. VẼ SƠ ĐỒ CÂY (nếu muốn) ====
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=feature_names, filled=True, rounded=True, fontsize=8)
plt.title("Cây quyết định - không dùng PCA")
plt.tight_layout()
plt.show()
"""
Điều kiện chia nhánh tại nút này. Nếu đúng → đi nhánh trái, nếu sai → đi nhánh phải.
squared_error = ... (mặc định)	Tổng sai số bình phương (variance) tại nút đó – càng nhỏ tức là các mẫu càng “giống nhau” hơn.
samples = 1200	Số lượng mẫu dữ liệu tại nút này. Giúp biết có bao nhiêu điểm dữ liệu được phân loại qua nút đó.
value = 72.4	Giá trị đầu ra trung bình (mean) của các mẫu trong nút đó. Đây chính là giá trị mà mô hình sẽ dự đoán nếu dừng ở nút này.
"""


# ==== 6. Decision Tree - CÓ PCA ====
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

tree_model_pca = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model_pca.fit(X_train_pca, y_train)
y_pred_tree_pca = tree_model_pca.predict(X_test_pca)

print("\n🔸 DECISION TREE (CÓ PCA)")
print("MAE:", round(mean_absolute_error(y_test, y_pred_tree_pca), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred_tree_pca), 3))
print("R² :",  round(r2_score(y_test, y_pred_tree_pca), 3))

"""
❗ PCA không thể vẽ cây một cách có ý nghĩa vì:

Khi bạn áp dụng PCA, các đặc trưng ban đầu như T1, RH_1, hour... bị biến đổi 
thành các trục mới: PC0, PC1, ..., là tổ hợp tuyến tính phức tạp của nhiều đặc trưng gốc.

➡️ Khi bạn vẽ cây sau PCA, thì:

- Các nút quyết định của cây sẽ là các trục PCA (PC0, PC1, ...) 
  → không thể giải thích trực quan như “nhiệt độ cao → tăng điện năng”.

- Việc gán label cho các feature khi vẽ bằng plot_tree() sẽ không rõ ràng 
  hoặc bị sai (nếu bạn đưa feature_names=feature_names thì tên sẽ không khớp).
"""
