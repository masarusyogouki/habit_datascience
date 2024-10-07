import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# データの読み込み
dir = 'habit'
df = pd.read_csv(f'{dir}/習慣化.csv', skiprows=2)

# 週をdatetime型に変換し、インデックスに設定
df['week'] = pd.to_datetime(df['週'])
df['relative value'] = df['習慣化: (日本)']
df.set_index('week', inplace=True)

# 時系列データの季節分解 (multiplicativeかadditiveを選べます)
result = seasonal_decompose(df['relative value'], model='additive', period=52)

# 分解結果をプロット
plt.figure(figsize=(10,8))

# トレンド成分
plt.subplot(3, 1, 1)
plt.plot(result.trend)
plt.title('Trend')

# 季節成分
plt.subplot(3, 1, 2)
plt.plot(result.seasonal)
plt.title('Seasonality')

# 残差成分
plt.subplot(3, 1, 3)
plt.plot(result.resid)
plt.title('Residual')

# グラフをpng形式で保存
plt.tight_layout()
plt.savefig(f'{dir}/季節性分解の可視化.png')