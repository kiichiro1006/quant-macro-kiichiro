import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# 1. データ取得期間
start_date = '1994-01-01'
end_date = '2024-01-01'

# 2. 実質GDPを取得し、対数変換
gdp_us = np.log(web.DataReader('GDPC1', 'fred', start_date, end_date))
gdp_jp = np.log(web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date))

# 3. λのリスト
lambdas = [10, 100, 1600]

# 4. 結果格納用の辞書
std_devs = {}
correlations = {}

# 5. 各λについて計算
for lam in lambdas:
    cycle_us, trend_us = sm.tsa.filters.hpfilter(gdp_us, lamb=lam)
    cycle_jp, trend_jp = sm.tsa.filters.hpfilter(gdp_jp, lamb=lam)

    # 標準偏差
    std_us = cycle_us.std()
    std_jp = cycle_jp.std()
    std_devs[lam] = {'US': std_us, 'JP': std_jp}

    # 相関係数
    corr = np.corrcoef(cycle_us.values.T, cycle_jp.values.T)[0, 1]
    correlations[lam] = corr

    # グラフ描画
    plt.figure(figsize=(12, 6))
    plt.plot(cycle_us, label='United States', color='red')
    plt.plot(cycle_jp, label='Japan', color='blue')
    plt.title(f'Cyclical Component of Real GDP (HP Filter λ={lam})')
    plt.xlabel('Year')
    plt.ylabel('Cyclical Component')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

lines = ["Standard deviation of λ"]
for lam in lambdas:
    line = f"λ={lam}: the US: {std_devs[lam]['US']:.4f} | Japan: {std_devs[lam]['JP']:.4f}"
    lines.append(line)

lines.append("Correlation of λ between the US and Japan")
for lam in lambdas:
    line = f"λ={lam}: Correlation = {correlations[lam]:.4f}"
    lines.append(line)

# プロットとして描画
fig, ax = plt.subplots(figsize=(8, 4 + len(lines) * 0.3))
ax.axis('off')  # 軸非表示

# テキスト描画
text = "\n".join(lines)
ax.text(0, 1, text, fontsize=12, va='top', ha='left', family='monospace')

# 保存
plt.tight_layout()
plt.show()