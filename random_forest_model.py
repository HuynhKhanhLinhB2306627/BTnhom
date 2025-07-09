import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
features = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])
target = df['Appliances']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluation
print("\n🎯 Random Forest")
print("MAE:", mean_absolute_error(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))

# Visualization
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Thực tế")
plt.plot(preds[:100], label="Random Forest Dự đoán", alpha=0.7)
plt.title("Random Forest: So sánh dự đoán vs thực tế (100 điểm đầu)")
plt.xlabel("Điểm dữ liệu")
plt.ylabel("Tiêu thụ năng lượng (Wh)")
plt.legend()
plt.grid(True)
plt.show()
