import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đảm bảo bạn đã tải file 'energydata_complete.csv' lên
df = pd.read_csv("energydata_complete.csv")

# Xử lý dữ liệu
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
features = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])
target = df['Appliances']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Huấn luyện mô hình cây quyết định (cây đầy đủ)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Lấy độ quan trọng của các đặc trưng từ mô hình ĐẦY ĐỦ
importances = model.feature_importances_
feature_names = features.columns

# Tạo DataFrame để dễ xem và vẽ
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# In ra các đặc trưng quan trọng nhất
print("Top 10 đặc trưng quan trọng nhất của mô hình (cây đầy đủ):")
print(feature_importance_df.head(10))

# Vẽ biểu đồ
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Độ quan trọng")
plt.ylabel("Đặc trưng")
plt.title("Độ quan trọng của các đặc trưng trong mô hình Decision Tree")
plt.gca().invert_yaxis()  # Đưa đặc trưng quan trọng nhất lên trên cùng
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout() # Đảm bảo các nhãn không bị cắt
plt.savefig("feature_importance.png")
print("\nĐã lưu biểu đồ độ quan trọng vào file: feature_importance.png")