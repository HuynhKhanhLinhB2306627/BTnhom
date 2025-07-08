#Thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")

#Tiền xử lý dữ liệu:  Xử lý cột thời gian
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
print(df.isnull().sum())

#Chọn đặc trưng đầu vào
features = df.drop(columns=['date', 'Appliances'])  # bỏ cột date và target
target = df['Appliances'] # nhãn của tập dữ liệu

#Tách tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Cây quyết định
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

#Rừng ngẫu nhiên
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

#Hàm đánh giá
def evaluate(name, y_true, y_pred):
    print(f"\n🎯 {name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("R2 Score:", r2_score(y_true, y_pred))

evaluate("Decision Tree", y_test, dt_preds)
evaluate("Random Forest", y_test, rf_preds)


#Vẽ biểu đồ giữ thực tế và dự đoán
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế")
plt.plot(rf_preds[:100], label="Random Forest Dự đoán", alpha=0.7)
plt.title("So sánh dự đoán vs thực tế (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.show()
