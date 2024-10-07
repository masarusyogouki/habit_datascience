# データの読み込みと可視化
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 頭2行はスキップして、csvファイルを読み込む
dir = 'habit'
df = pd.read_csv(f'{dir}/習慣化.csv', skiprows=2)

# 週をdatatime型に変換
df['week'] = pd.to_datetime(df['週'])
df['relative value'] = df['習慣化: (日本)']

# 週をインデックスに設定
df.set_index('week', inplace=True)

# matplotlibによる可視化
plt.figure(figsize=(10,6))
plt.plot(df.index, df['relative value'], label = 'relative value')

# 線形回帰モデルを使用してトレンドラインを追加
x = np.arange(len(df.index)).reshape(-1, 1)
y = df['relative value'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)
trend_line = model.predict(x)

# トレンドラインをプロット
plt.plot(df.index, trend_line, color = 'red', linestyle = '-', label = 'Trend Line')

plt.title('Search Trends Over 5 Years')
plt.xlabel('Week')
plt.ylabel('Relative Count')
plt.legend()

# グラフをpng形式で保存
plt.savefig(f'{dir}/習慣化の可視化.png')