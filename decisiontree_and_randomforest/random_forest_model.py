import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
# Dữ liệu gồm nhiều thông số đo lường tiêu thụ năng lượng trong gia đình

df = pd.read_csv("energydata_complete.csv")

# Tiền xử lý dữ liệu
# Chuyển cột 'date' sang kiểu datetime để dễ trích xuất thông tin thời gian
# Tạo thêm các đặc trưng về thời gian: giờ, ngày, thứ trong tuần
# Loại bỏ các cột không cần thiết khỏi tập đặc trưng (features)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['hour'] = df['date'].dt.hour
df['week'] = df['date'].dt.isocalendar().week
df['day_of_week'] = df['date'].dt.day_name()
df['week_of_month'] = (df['date'].dt.day-1) // 7 + 1
features = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])  # Tập đặc trưng đầu vào

# Biến mục tiêu là lượng tiêu thụ thiết bị điện ('Appliances')
target = df['Appliances']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình rừng ngẫu nhiên (Random Forest)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
preds = model.predict(X_test)

# Đánh giá chất lượng mô hình bằng các chỉ số MAE, MSE, R2
print("\n🎯 Random Forest")
print("MAE:", mean_absolute_error(y_test, preds))  # Sai số tuyệt đối trung bình
print("MSE:", mean_squared_error(y_test, preds))  # Sai số bình phương trung bình
print("R2 Score:", r2_score(y_test, preds))       # Hệ số xác định (độ phù hợp)

# Trực quan hóa: So sánh giá trị thực tế và dự đoán trên 100 điểm dữ liệu đầu tiên
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế")
plt.plot(preds[:100], label="Random Forest Dự đoán", alpha=0.7)
plt.title("Random Forest: So sánh dự đoán vs thực tế (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.show()
