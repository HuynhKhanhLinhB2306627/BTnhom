import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# 1. Đọc dữ liệu và chọn biến đầu vào
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])

X = df[['T1', 'RH_1', 'T_out', 'Windspeed']]  # ví dụ vài biến đầu vào
X = X.dropna()

# 2. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Áp dụng LOF
lof = LocalOutlierFactor(n_neighbors=20)
outlier_flags = lof.fit_predict(X_scaled)

# 4. Gán nhãn (1: outlier, 0: inlier)
df_lof = X.copy()
df_lof['outlier'] = (outlier_flags == -1).astype(int)
df_lof['Appliances'] = df.loc[X.index, 'Appliances']  # Lấy lại cột Appliances

"""
Tên biểu đồ: Biểu đồ hộp phân nhóm theo outlier - Grouped Boxplot by LOF Outlier Flag
Ý nghĩa biểu đồ:
    Trục Y: giá trị tiêu thụ năng lượng (Appliances)
    Trục X: nhãn outlier, gồm:
    0: điểm bình thường (regular)
    1: điểm bị phát hiện là outlier theo LOF
Kết luận từ biểu đồ:
    Phân phối giữa nhóm thường và nhóm outlier gần như giống nhau
    Trung vị của cả hai nhóm đều quanh 60 Wh
→ Điều này cho thấy: giá trị ngoại lai được LOF phát hiện không thật sự ảnh hưởng đến biến mục tiêu, 
    nên có thể cân nhắc không loại bỏ chúng
"""
plt.figure(figsize=(10, 5))
sns.boxplot(x='outlier', y='Appliances', data=df_lof, palette='coolwarm')
plt.title("Energy usage for regular data points and the data points with outliers")
plt.xlabel("Outlier (1 = có ngoại lai)")
plt.ylabel("Appliances")
plt.grid(True)
plt.tight_layout()
plt.show()
