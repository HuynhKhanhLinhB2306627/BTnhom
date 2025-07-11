import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")

# In số mẫu và số thuộc tính
print("Số mẫu (số dòng):", df.shape[0])
print("Số thuộc tính (số cột):", df.shape[1])

# In tên các thuộc tính
print("\nTên các thuộc tính:")
print(df.columns.tolist())

# Kiểm tra null
print("\nSố lượng phần tử bị khuyết trng bảng dữ liệu:")
print(df.isnull().sum().to_dict())


