import pandas as pd
import matplotlib.pyplot as plt

# ファイルの読み込み
# 1行目はメタデータのためスキップし、ヘッダーなしとして読み込みます
file_path = 'Ir(PIQ)2(dmbpy)PF6-limonene.CSV'
df = pd.read_csv(file_path, skiprows=1, header=None, names=['X', 'Y'])

# データのプロット
plt.figure(figsize=(10, 6))
plt.plot(df['X'], df['Y'], color='blue', linewidth=1)

# タイトルとラベル（仮）
plt.title(f'Plot of {file_path}')
plt.xlabel('X Axis (Column 1)')
plt.ylabel('Y Axis (Column 2)')
plt.grid(True, linestyle='--', alpha=0.6)

# グラフを表示
plt.show()