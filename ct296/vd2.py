import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv('energydata_complete.csv')

# Xử lý cột thời gian
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

# Chọn đặc trưng và mục tiêu
features = df.drop(columns=['date', 'Appliances'])
target = df['Appliances']

# Tách tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Pipeline KNN với chuẩn hóa dữ liệu
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5))
])

# Huấn luyện mô hình
knn_pipeline.fit(X_train, y_train)

# Dự đoán
knn_preds = knn_pipeline.predict(X_test)

# Hàm đánh giá
def evaluate(name, y_true, y_pred):
    print(f"\n🎯 {name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("R2 Score:", r2_score(y_true, y_pred))

evaluate("KNN Regressor", y_test, knn_preds)

# Vẽ biểu đồ dự đoán vs thực tế
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế")
plt.plot(knn_preds[:100], label="KNN Dự đoán", alpha=0.7)
plt.title("So sánh dự đoán KNN vs thực tế (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.show()
