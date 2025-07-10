import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # Thêm GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- Phần đọc và xử lý dữ liệu của bạn giữ nguyên ---
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['hour'] = df['date'].dt.hour
features = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])
target = df['Appliances']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# ---------------------------------------------------------

# 1. Định nghĩa "lưới" các siêu tham số bạn muốn thử
param_grid = {
    'n_estimators': [100, 150],       # Thử với 100 và 150 cây
    'max_depth': [10, 20, None],       # Thử các độ sâu khác nhau
    'max_features': ['sqrt', 'log2']   # Thử các cách chọn feature
}

# 2. Khởi tạo GridSearchCV
# cv=5 nghĩa là dùng 5-fold cross-validation
# n_jobs=-1 để dùng tất cả CPU, giúp chạy nhanh hơn
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2,
                           scoring='r2')

# 3. Huấn luyện (quá trình này sẽ tìm bộ tham số tốt nhất)
print("Bắt đầu tinh chỉnh siêu tham số...")
grid_search.fit(X_train, y_train)

# 4. In ra bộ tham số tốt nhất đã tìm được
print(f"Bộ tham số tốt nhất: {grid_search.best_params_}")

# 5. Đánh giá mô hình tốt nhất trên tập test
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
print(f"R2 Score trên tập test với mô hình tốt nhất: {r2_score(y_test, predictions)}")