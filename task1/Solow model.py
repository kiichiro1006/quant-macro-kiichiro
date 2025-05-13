import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web

# 期間設定
start_date = '1990-01-01'
end_date = '2022-01-01'

# データ取得（FREDから）
gdp = web.DataReader('GDPC1', 'fred', start_date, end_date)         # 実質GDP
investment = web.DataReader('PNFI', 'fred', start_date, end_date)   # 固定資本形成（資本 proxy）
employment = web.DataReader('CE16OV', 'fred', start_date, end_date) # 雇用者数（労働）

# データを結合
data = pd.concat([gdp, investment, employment], axis=1)
data.columns = ['GDP', 'Capital_Proxy', 'Labor']
data = data.dropna()

# 年次平均に変換（四半期→年次）
data = data.resample('A').mean()

# 対数変換と成長率（Δlog）
log_data = np.log(data)
growth = log_data.diff().dropna()

# 資本の分配率（仮にα = 0.33 とする）
alpha = 0.33

# Solow残差（TFPの成長率）を推定
growth['TFP'] = (
    growth['GDP']
    - alpha * growth['Capital_Proxy']
    - (1 - alpha) * growth['Labor']
)

# 成分別寄与度をプロット
growth[['TFP', 'Capital_Proxy', 'Labor']].plot(
    title='Economic Growth Decomposition (USA)',
    figsize=(10, 6)
)
plt.ylabel('Growth Rate (Δlog)')
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web

# 分析期間の設定
start_date = '1994-01-01'
end_date = '2023-12-31'

# ======================================
# 日本のデータ読み込み（Excelファイルから）
# ======================================

# ファイル読み込み
japan_comp = pd.read_excel('japan composition.xlsx')  # 列: Year, GDP, Capital, Employee

# 年をインデックスに設定
japan_comp.index = pd.to_datetime(japan_comp['Year'].astype(str), format='%Y')
japan_comp = japan_comp.drop(columns='Year')

# 単位変換（1ドル=100円想定）
japan_comp['GDP'] = japan_comp['GDP'] * 1e9 / 100         # 10億円 → 円 → ドル
japan_comp['Capital'] = japan_comp['Capital'] * 1e9 / 100 # 10億円 → 円 → ドル
japan_comp['Employee'] = japan_comp['Employee'] * 1e4     # 万人 → 人

# ログ変換と成長率
japan_log = np.log(japan_comp)
japan_growth = japan_log.diff().dropna()

# αの設定（資本分配率）
alpha = 0.33

# Solow残差（TFP成長率）の推計
japan_growth['TFP'] = japan_growth['GDP'] - alpha * japan_growth['Capital'] - (1 - alpha) * japan_growth['Employee']


# ======================================
# アメリカのデータ取得（FRED）
# ======================================

us_gdp = web.DataReader('GDPC1', 'fred', start_date, end_date)       # 実質GDP
us_capital = web.DataReader('PNFI', 'fred', start_date, end_date)    # 資本
us_labor = web.DataReader('CE16OV', 'fred', start_date, end_date)    # 雇用者数

# 年次でリサンプリング（平均）
us_data = pd.concat([us_gdp, us_capital, us_labor], axis=1)
us_data.columns = ['GDP', 'Capital', 'Labor']
us_data = us_data.dropna().resample('YE').mean()  # 'A' は非推奨なので 'YE' に変更

# ログ変換と成長率
us_log = np.log(us_data)
us_growth = us_log.diff().dropna()

# Solow残差（TFP成長率）の推計
us_growth['TFP'] = us_growth['GDP'] - alpha * us_growth['Capital'] - (1 - alpha) * us_growth['Labor']


# ======================================
# TFP成長率の比較プロット
# ======================================

plt.figure(figsize=(12, 6))
plt.plot(japan_growth['TFP'], label='Japan TFP Growth', marker='o')
plt.plot(us_growth['TFP'], label='USA TFP Growth', marker='x')
plt.title('TFP Growth Comparison: Japan vs USA (Solow Residual)')
plt.ylabel('TFP Growth Rate (Δlog)')
plt.xlabel('Year')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# α（資本の所得分配率）
alpha = 0.33

# -------------------------------
# 日本の成長率寄与分解（既にある japan_growth を使用）
# -------------------------------
japan_growth['Capital_Contribution'] = alpha * japan_growth['Capital']
japan_growth['Labor_Contribution'] = (1 - alpha) * japan_growth['Employee']
japan_growth['TFP_Contribution'] = japan_growth['TFP']

# -------------------------------
# アメリカの成長率寄与分解（既にある us_growth を使用）
# -------------------------------
us_growth['Capital_Contribution'] = alpha * us_growth['Capital']
us_growth['Labor_Contribution'] = (1 - alpha) * us_growth['Labor']
us_growth['TFP_Contribution'] = us_growth['TFP']

# -------------------------------
# 積み上げ棒グラフ：日本
# -------------------------------
plt.figure(figsize=(14, 5))
japan_growth_plot = japan_growth[['Capital_Contribution', 'Labor_Contribution', 'TFP_Contribution']]
japan_growth_plot.plot(kind='bar', stacked=True, figsize=(14,5), title='japan composition Solow Model')
plt.ylabel('growth rate Δlog')
plt.xlabel('year')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()

# -------------------------------
# 積み上げ棒グラフ：アメリカ
# -------------------------------
plt.figure(figsize=(14, 5))
us_growth_plot = us_growth[['Capital_Contribution', 'Labor_Contribution', 'TFP_Contribution']]
us_growth_plot.plot(kind='bar', stacked=True, figsize=(14,5), title='US composition Solow Model')
plt.ylabel('growth rate Δlog')
plt.xlabel('year')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()
