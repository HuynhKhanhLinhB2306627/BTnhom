import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])

# Lấy 2 cột để phân tích
data = df[['T1', 'Appliances']].dropna()

# Chuẩn hóa
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# DBSCAN
db = DBSCAN(eps=0.5, min_samples=10)
labels = db.fit_predict(data_scaled)
data['Cluster'] = labels

# Vẽ
plt.figure(figsize=(12, 6))

"""
Biểu đồ tán xạ (Scatter plot with DBSCAN): Trục X là T1 (nhiệt độ phòng 1), trục Y là Appliances (tiêu thụ năng lượng).
Mỗi điểm là một quan sát (1 hàng dữ liệu).
Màu sắc phản ánh cụm DBSCAN:
    Màu tím (Core Points): các điểm nằm trong vùng mật độ cao → thuộc cụm rõ ràng.
    Màu vàng (Noise): các điểm bị coi là ngoại lai (outliers).
Mục tiêu: phát hiện vùng bất thường có thể là sai lệch hoặc điện áp tăng cao.
"""
plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='T1', y='Appliances', hue='Cluster', palette='viridis', s=15, alpha=0.7)
plt.title("Phân cụm DBSCAN: T1 vs Appliances")
plt.legend(title="Nhóm", bbox_to_anchor=(1.05, 1), loc='upper left')

"""
Biểu đồ hộp đứng (Box plot - vertical)
    Dựng theo biến Appliances, đứng dọc.
    Thể hiện sự lệch phải và nhiều giá trị ngoại lai (dots).
    Hỗ trợ xác nhận trực quan rằng dữ liệu bị lệch và có outlier.
"""
plt.subplot(1, 2, 2)
sns.boxplot(y=data['Appliances'], orient='v', color='mediumpurple')
plt.title("Biểu đồ hộp: Appliances")

plt.tight_layout()
plt.show()
