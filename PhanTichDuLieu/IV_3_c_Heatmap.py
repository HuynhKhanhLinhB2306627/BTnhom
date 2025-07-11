import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")

# Lấy subset các biến đầu vào
input_features = df.drop(columns=['date', 'Appliances'])  # hoặc chọn cột T1, RH_1, T_out, etc.

# Tính hệ số tương quan Pearson
corr_matrix = input_features.corr(method='pearson')

"""
Biểu đồ tương quan (Correlation Heatmap) hoặc Ma trận tương quan (Correlation Matrix)
Ý nghĩa:
    Hiển thị mức độ tương quan tuyến tính giữa các biến đầu vào trong tập dữ liệu
    Mỗi ô thể hiện hệ số tương quan Pearson giữa 2 biến:
    Giá trị từ -1 đến +1:
        +1 → tương quan dương hoàn hảo
        -1 → tương quan âm hoàn hảo
        0 → không có tương quan
Mục tiêu:
    Phát hiện cặp biến tương quan mạnh → có thể loại bớt 1 trong 2 để tránh dư thừa
    Tránh hiện tượng đa cộng tuyến (multicollinearity) trong mô hình học máy
"""
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f", square=True, linewidths=0.5)
plt.title("Biểu đồ tương quan giữa các biến đầu vào (Correlation Heatmap)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
