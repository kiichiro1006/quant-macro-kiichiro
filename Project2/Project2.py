import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

oecd_countries = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland',
    'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Japan',
    'Netherlands', 'New Zealand', 'Norway', 'Portugal', 'Spain', 'Sweden',
    'Switzerland', 'United Kingdom', 'United States'
]
data = pwt90[
    pwt90['country'].isin(oecd_countries) &
    pwt90['year'].between(1990, 2019)
]

relevant_cols = ['countrycode', 'country', 'year', 'rgdpna', 'rkna', 'pop', 'emp', 'avh', 'labsh', 'rtfpna','hc']
data = data[relevant_cols].dropna()

# Calculate additional variables
data['alpha'] = 1 - data['labsh']
data['hours'] = data['emp'] * data['avh']  # L
data['y'] = data['rgdpna'] / data['hours']  # y = Y/L
data['tfp_term'] = data['rtfpna'] #A
data['k'] = (data['rkna'] +data['hc'])/ data['hours'] # k = K/L
data['cap_term'] = (data['k']) ** (data['alpha'] )  # k^α
data = data.sort_values('year').groupby('countrycode').apply(lambda x: x.assign(
    alpha=1 - x['labsh'],
    y_n_shifted=100 * x['y'] / x['y'].iloc[0],
    tfp_term_shifted=100 * x['tfp_term'] / x['tfp_term'].iloc[0],
    cap_term_shifted=100 * x['cap_term'] / x['cap_term'].iloc[0],
)).reset_index(drop=True).dropna()


def calculate_growth_rates(country_data):

    start_year_actual = country_data['year'].min()
    end_year_actual = country_data['year'].max()

    start_data = country_data[country_data['year'] == start_year_actual].iloc[0]
    end_data = country_data[country_data['year'] == end_year_actual].iloc[0]

    years = end_data['year'] - start_data['year']

    g_y = ((end_data['y'] / start_data['y']) ** (1/years) - 1) * 100

    g_k = ((end_data['cap_term'] / start_data['cap_term']) ** (1/years) - 1) * 100

    g_a = ((end_data['tfp_term'] / start_data['tfp_term']) ** (1/years) - 1) * 100

    alpha_avg = (start_data['alpha'] + end_data['alpha']) / 2.0
    capital_deepening_contrib = alpha_avg * g_k
    tfp_growth_calculated = g_a

    tfp_share = (tfp_growth_calculated / g_y)
    cap_share = (capital_deepening_contrib / g_y)

    return {
        'Country': start_data['country'],
        'Growth Rate': round(g_y, 2),
        'TFP Growth': round(tfp_growth_calculated, 2),
        'Capital Deepening': round(capital_deepening_contrib, 2),
        'TFP Share': round(tfp_share, 2),
        'Capital Share': round(cap_share, 2)
    }


results_list = data.groupby('country').apply(calculate_growth_rates).dropna().tolist()
results_df = pd.DataFrame(results_list)

avg_row_data = {
    'Country': 'Average',
    'Growth Rate': round(results_df['Growth Rate'].mean(), 2),
    'TFP Growth': round(results_df['TFP Growth'].mean(), 2),
    'Capital Deepening': round(results_df['Capital Deepening'].mean(), 2),
    'TFP Share': round(results_df['TFP Share'].mean(), 2),
    'Capital Share': round(results_df['Capital Share'].mean(), 2)
}
results_df = pd.concat([results_df, pd.DataFrame([avg_row_data])], ignore_index=True)

print("\nGrowth Accounting in OECD Countries: 1990-2019 period")
print("="*85)
print(results_df.to_string(index=False))

# 表示する列を選択（すべての列を含める）
columns_to_display = [
    'Country',
    'Growth Rate',
    'TFP Growth',
    'Capital Deepening',
    'TFP Share',
    'Capital Share'
]
df_to_plot = results_df[columns_to_display]

# プロットのセットアップ
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# テーブルの描画
col_widths = [0.19] + [0.21] * (len(columns_to_display) - 1)

table = ax.table(cellText=df_to_plot.values,
                 colLabels=df_to_plot.columns,
                 loc='center',
                 cellLoc='center',
                 colWidths=col_widths)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# タイトル
fig.text(0.02, 0.95, 'Growth Accounting in OECD Countries: 1990-2019',
         fontsize=14, fontweight='bold', ha='left', va='top')

# ヘッダー太字
for j in range(len(columns_to_display)):
    header_cell = table[0, j]
    header_cell._text.set_weight('bold')

# 'Average' 行の強調
avg_row_index = len(df_to_plot)
for j in range(len(columns_to_display)):
    cell = table[(avg_row_index, j)]
    cell._text.set_weight('bold')

# すべてのセルの境界線を非表示（まず初期化）
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_linewidth(0)

# ヘッダーに下線のみ
for j in range(len(columns_to_display)):
    cell = table[(0, j)]
    cell.visible_edges = 'T'  # 下だけ線を表示
    cell.set_edgecolor('black')
    cell.set_linewidth(1.5)
    cell.set_fontsize(11)
    cell.PAD = 0.1

# 'Australia' 行（インデックス1）の上に横線
australia_row_index = 1
for j in range(len(columns_to_display)):
    cell = table[(australia_row_index, j)]
    cell.visible_edges = 'T'
    cell.set_edgecolor('black')
    cell.set_linewidth(1.5)


# 'Average'行に上下線のみ
for j in range(len(columns_to_display)):
    cell = table[(avg_row_index, j)]
    cell.visible_edges = 'TB'  # 上線だけ表示（必要に応じて 'TB' で上下両方）
    cell.set_edgecolor('black')
    cell.set_linewidth(1.5)
    cell.set_fontsize(11)
    cell.PAD = 0.1

# 画像として保存
plt.savefig('growth_accounting_final_styled_table.png', bbox_inches='tight', dpi=300)
plt.show()
