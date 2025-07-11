import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")

# Tạo figure có 2 biểu đồ cạnh nhau
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Appliances'], kde=True, bins=100)
plt.title("Phân phối biến Appliances")
plt.xlabel("Appliances")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Appliances'])
plt.title("Biểu đồ hộp biến Appliances")
plt.xlabel("Appliances")

plt.tight_layout()
plt.show()


