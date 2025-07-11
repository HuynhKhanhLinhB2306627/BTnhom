# plot_section_d.py

import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])

# --- Biểu đồ 1: Mean by weekday ---
df['weekday'] = df['date'].dt.weekday
mean_by_weekday = df.groupby('weekday')['Appliances'].mean()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(8, 5))
plt.bar([days[i] for i in mean_by_weekday.index], mean_by_weekday.values, color='skyblue')
plt.title("Mean Energy Consumption by Day of Week")
plt.ylabel("Mean Energy Consumption (Wh)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("d1_mean_by_weekday.png")
plt.show()

# --- Biểu đồ 2: Mean by hour ---
df['hour'] = df['date'].dt.hour
mean_by_hour = df.groupby('hour')['Appliances'].mean()

plt.figure(figsize=(8, 5))
plt.plot(mean_by_hour.index, mean_by_hour.values, marker='o', color='steelblue')
plt.title("Mean Energy Consumption by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Energy Consumption (Wh)")
plt.grid(True)
plt.tight_layout()
plt.savefig("d2_mean_by_hour.png")
plt.show()
