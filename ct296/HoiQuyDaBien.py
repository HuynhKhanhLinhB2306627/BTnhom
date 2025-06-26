import pandas as pd
from sklearn.linear_model import LinearRegression

# ==== 1. Đọc & sinh feature ==== 
df = pd.read_csv("energydata_complete.csv")
df['date']    = pd.to_datetime(df['date'])
df['hour']    = df['date'].dt.hour
df['day']     = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

X = df.drop(columns=['date', 'Appliances'])
y = df['Appliances']
X = X.fillna(X.median())     
# ==== 2. Mô hình LinearRegression ====
model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coeffs    = dict(zip(X.columns, model.coef_))

# ==== 3. In hệ số & phương trình ====
print("Intercept:", f"{intercept:.3f}")
for feat, beta in coeffs.items():
    print(f"{feat:10}: {beta: .3f}")

# --- Tạo chuỗi phương trình gọn ---
terms = [f"{intercept:.3f}"]
for feat, beta in coeffs.items():
    sign = "+" if beta >= 0 else "-"
    terms.append(f" {sign} {abs(beta):.3f}*{feat}")

equation = " ".join(terms)

print("\nPhương trình hồi quy đa biến:")
print("Appliances_hat =", equation)
