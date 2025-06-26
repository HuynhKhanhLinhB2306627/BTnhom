
# ========= 1. Thông số tuỳ chỉnh =========
CSV_FILE      = "energydata_complete.csv"  # file
TEST_SIZE     = 0.3       # 30 % test
N_NEIGHBORS   = 5         # k của K-NN
IMPUTE_STRAT  = "median"  # chọn cách lấp (impute) giá trị NaN – ở đây lấy median (trung vị) của mỗi cột.
KFOLD         = 5         # chỉ số K_Fold
RAND          = 42        #Chia ngẫu nhiên (seed = 42)

# ========= 2. Thư viện =========
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.metrics         import (mean_absolute_error,
                                     mean_squared_error,
                                     r2_score,
                                     explained_variance_score)

# ========= 3. Đọc & tiền xử lý =========
df = pd.read_csv(CSV_FILE)
df['date']    = pd.to_datetime(df['date'])
df['hour']    = df['date'].dt.hour
df['day']     = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

#Chọn đặc trưng đầu vào
X = df.drop(columns=['date', 'Appliances'])
y = df['Appliances']

# ========= 4. Train-test split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND )    

# ========= 5. Pipeline =========
pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=IMPUTE_STRAT)),      # điền thiếu theo trung vị.
    ('scaler',  StandardScaler()),                          # chuẩn hoá mỗi đặc trưng (mean = 0, var = 1)
    ('knn',     KNeighborsRegressor(n_neighbors=N_NEIGHBORS)) # KNN Regressor: mô hình lõi.
])
pipe.fit(X_train, y_train)


# ========= 6. Dự đoán & metric =========
y_pred = pipe.predict(X_test)
residuals = y_test - y_pred

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

r2          = r2_score(y_test, y_pred)
n_test, p   = X_test.shape
metrics = {
    'MAE'          : mean_absolute_error(y_test, y_pred),
    'RMSE'         : np.sqrt(mean_squared_error(y_test, y_pred)),  
    'R²'           : r2,
    'Adj R²'       : adjusted_r2(r2, n_test, p),
    'ExplainedVar' : explained_variance_score(y_test, y_pred)
}

# R²_pred (PRESS) 5-fold CV
y_pred_cv          = cross_val_predict(pipe, X, y, cv=KFOLD)
metrics['R²_pred'] = r2_score(y, y_pred_cv)


print("==== Metric trên tập test ====")
for k, v in metrics.items():
    print(f"{k:15}: {v: .4f}")

# ========= 7. So sánh 7 dòng đầu =========
print("\n>>> Thực tế vs Dự đoán (7 điểm đầu tập test)")
for i, (true, pred) in enumerate(zip(y_test.iloc[:7], y_pred[:7]), 1):
    print(f"{i:2}: {true:7.1f} Wh | {pred:7.1f} Wh")

# ========= 9. Biểu đồ =========

# 9.1 Line plot Actual vs Predicted
plt.figure(figsize=(11, 4))
plt.plot(y_test.values[:200], label="Thực tế")
plt.plot(y_pred[:200], label="Dự đoán", alpha=0.7)
plt.title("Actual vs Predicted (200 điểm đầu)")
plt.xlabel("Điểm dữ liệu"); plt.ylabel("Wh")
plt.legend(); plt.grid(True); plt.show()

# 9.2 Scatter Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], linestyle='--')
plt.title("Scatter: Predicted vs Actual")
plt.xlabel("Thực tế (Wh)"); plt.ylabel("Dự đoán (Wh)")
plt.grid(True); plt.show()

# 9.3 Residuals vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Dự đoán (Wh)"); plt.ylabel("Residual (Wh)")
plt.grid(True); plt.show()

# 9.4 Histogram Residuals
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=50)
plt.title("Histogram Residuals")
plt.xlabel("Residual (Wh)"); plt.ylabel("Frequency")
plt.grid(True); plt.show()

# 9.5 Bar chart Metrics
plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values())
plt.title("Regression Metrics")
plt.ylabel("Value")
plt.xticks(rotation=45); plt.grid(True, axis='y'); plt.tight_layout()
plt.show()

# ========= 10. (Tuỳ chọn) Lưu mô hình =========
# import joblib
# joblib.dump(pipe, "knn_energy_model.pkl")
