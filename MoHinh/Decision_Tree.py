import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
# File 'energydata_complete.csv' cần nằm cùng thư mục với file code này
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
df['week'] = df['date'].dt.isocalendar().week    # Tuần trong năm (ISO week)
df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1  # Tuần trong tháng
features = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])  # Tập đặc trưng đầu vào

# Biến mục tiêu là lượng tiêu thụ thiết bị điện ('Appliances')
target = df['Appliances']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#----------------------------------------------------------------------------------------------------------#



# Khởi tạo và huấn luyện mô hình cây quyết định (Decision Tree)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
preds = model.predict(X_test)

# Đánh giá chất lượng mô hình bằng các chỉ số MAE, MSE, RMSE, MAPE, R2, Explained Variance
print("\n🎯 Decision Tree")
print("MAE:", mean_absolute_error(y_test, preds))  # Sai số tuyệt đối trung bình
print("MSE:", mean_squared_error(y_test, preds))  # Sai số bình phương trung bình
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))  # Căn bậc hai của MSE
print("MAPE:", np.mean(np.abs((y_test - preds) / y_test)) * 100)  # Sai số phần trăm tuyệt đối trung bình
print("R2 Score:", r2_score(y_test, preds))       # Hệ số xác định (độ phù hợp)
print("Explained Variance Score:", explained_variance_score(y_test, preds))  # Phần phương sai được giải thích

# Trực quan hóa kết quả dự đoán:
# Vẽ biểu đồ đường so sánh giữa giá trị thực tế và giá trị dự đoán của mô hình trên 100 điểm dữ liệu đầu tiên của tập kiểm tra.
# Đường màu xanh là giá trị thực tế, đường màu cam là giá trị dự đoán.
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế")
plt.plot(preds[:100], label="Decision Tree Dự đoán", alpha=0.7)
plt.title("Decision Tree: So sánh dự đoán vs thực tế (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")  # Trục hoành: chỉ số điểm dữ liệu
plt.ylabel("Tiêu thụ năng lượng (Wh)")  # Trục tung: giá trị tiêu thụ năng lượng
plt.legend()  # Hiển thị chú thích các đường
plt.grid(True)  # Hiển thị lưới cho biểu đồ dễ quan sát
plt.show()
