#Thư viện cần thiết
import pandas as pd


#Đọc dữ liệu
df = pd.read_csv("energydata_complete.csv")

#Xử lý thời gian
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

print(df.isnull().sum())