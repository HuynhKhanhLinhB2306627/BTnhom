import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")

# Tiền xử lý thời gian
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['hour'] = df['date'].dt.hour
df['week'] = df['date'].dt.isocalendar().week
df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1

# Tập đặc trưng và mục tiêu
features = df.drop(columns=['date', 'Appliances', 'rv1', 'rv2'])
target = df['Appliances']

# Tạo figure với 1 dòng 2 biểu đồ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(target, kde=True, ax=axes[0])
axes[0].set_title("Histogram of Appliances")
axes[0].set_xlabel("Appliances")
axes[0].set_ylabel("Count")

# Boxplot
sns.boxplot(x=target, ax=axes[1])
axes[1].set_title("Boxplot of Appliances")
axes[1].set_xlabel("Appliances")

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
