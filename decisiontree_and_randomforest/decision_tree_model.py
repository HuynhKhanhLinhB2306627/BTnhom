import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
# Dữ liệu gồm nhiều thông số đo lường tiêu thụ năng lượng trong gia đình

df = pd.read_csv("energydata_complete.csv")

# Tiền xử lý dữ liệu
# Chuyển cột 'date' sang kiểu datetime để dễ trích xuất thông tin thời gian
# Tạo thêm các đặc trưng về thời gian: tháng, thứ trong tuần, giờ, tuần trong năm, tuần trong tháng
# Loại bỏ các cột không cần thiết khỏi tập đặc trưng (features)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month                # Tháng trong năm
df['weekday'] = df['date'].dt.weekday            # Thứ trong tuần (0=Thứ 2, 6=Chủ nhật)
df['hour'] = df['date'].dt.hour                  # Giờ trong ngày
# Tuần trong năm (ISO week)
df['week'] = df['date'].dt.isocalendar().week    # Tuần trong năm
# Tuần trong tháng (tính từ ngày 1)
df['week_of_month'] = (df['date'].dt.day-1) // 7 + 1
features = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])  # Tập đặc trưng đầu vào

# Biến mục tiêu là lượng tiêu thụ thiết bị điện ('Appliances')
target = df['Appliances']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình cây quyết định (Decision Tree)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
preds = model.predict(X_test)

# Đánh giá chất lượng mô hình bằng các chỉ số MAE, MSE, R2
print("\n🎯 Decision Tree")
print("MAE:", mean_absolute_error(y_test, preds))  # Sai số tuyệt đối trung bình
print("MSE:", mean_squared_error(y_test, preds))  # Sai số bình phương trung bình
print("R2 Score:", r2_score(y_test, preds))       # Hệ số xác định (độ phù hợp)

# Trực quan hóa: So sánh giá trị thực tế và dự đoán trên 100 điểm dữ liệu đầu tiên
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế")
plt.plot(preds[:100], label="Decision Tree Dự đoán", alpha=0.7)
plt.title("Decision Tree: So sánh dự đoán vs thực tế (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.show()