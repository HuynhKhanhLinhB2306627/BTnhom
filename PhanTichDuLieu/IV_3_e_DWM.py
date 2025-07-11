# plot_section_e.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])

# Tính tuần trong tháng và thứ trong tuần
df['weekday'] = df['date'].dt.weekday
df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1

# Tạo pivot table
pivot_table = df.pivot_table(values='Appliances', index='weekday', columns='week_of_month', aggfunc='mean')

# Vẽ heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Mean Energy Consumption by Weekday and Week of Month")
plt.xlabel("Week of Month")
plt.ylabel("Weekday (0=Monday, 6=Sunday)")
plt.tight_layout()
plt.savefig("e1_weekday_vs_weekofmonth.png")
plt.show()
