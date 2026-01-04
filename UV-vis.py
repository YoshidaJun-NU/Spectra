import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# --- (1) 動作確認用ダミーデータ作成関数（データがあれば不要です） ---
def create_dummy_data():
    """動作確認用にUV1.csv～UV7.csvを作成する"""
    x = np.linspace(200, 800, 100)
    for i in range(1, 8):
        # 少しずつピークをずらしてデータを作成
        y = np.exp(-((x - (300 + i * 50))**2) / 2000) + np.random.normal(0, 0.005, len(x))
        df = pd.DataFrame({'Wavelength': x, 'Abs': y})
        filename = f'UV{i}.csv'
        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
            print(f"作成しました: {filename}")

# --- (2) メインのプロット関数 ---
def plot_spectra(
    file_list=None,          # 読み込むファイル名のリスト
    file_type='csv',         # 'csv' または 'txt'
    cmap_name='viridis',     # カラーマップ名 (例: 'viridis', 'jet', 'coolwarm')
    manual_colors=None,      # 手動で色を指定する場合のリスト ['red', 'blue'...]
    legend_loc='outside',    # 'outside' または 'inside'
    x_label="Wavelength (nm)",
    y_label="Abs.",
    x_lim=None,              # 例: (200, 800)
    y_lim=None,              # 例: (0, 1.5)
    header_row=0             # ヘッダーの行番号
):
    
    # 1. デフォルトのファイルリスト設定 (UV1.csv ~ UV7.csv)
    if file_list is None:
        ext = 'csv' if file_type == 'csv' else 'txt'
        file_list = [f'UV{i}.{ext}' for i in range(1, 8)]

    # 存在しないファイルを除外
    valid_files = [f for f in file_list if os.path.exists(f)]
    if not valid_files:
        print("エラー: 読み込めるファイルが見つかりません。")
        return

    num_files = len(valid_files)
    
    # 2. 色の設定 (カラーマップ vs 手動設定)
    if manual_colors:
        # 手動設定が有効な場合
        colors = manual_colors
        # ファイル数より色が少ない場合は繰り返す
        if len(colors) < num_files:
            colors = colors * (num_files // len(colors) + 1)
    else:
        # カラーマップから連続的な色を生成
        # 0.0(始点)から1.0(終点)までの間をファイル数分だけ分割して色を取得
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i) for i in np.linspace(0, 1, num_files)]

    # 3. グラフの準備
    fig, ax = plt.subplots(figsize=(8, 6))

    # 4. データ読み込みとプロット
    print(f"--- プロット開始 ({len(valid_files)} files) ---")
    
    for i, file_path in enumerate(valid_files):
        try:
            # 区切り文字の自動判定
            delimiter = ',' if file_type == 'csv' else '\t'
            
            # データ読み込み
            df = pd.read_csv(file_path, delimiter=delimiter, header=header_row)
            
            # 列データの取得 (1列目をX, 2列目をYと仮定)
            # 列名がわかっている場合は df['Wavelength'] など指定推奨
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            
            # ファイル名をラベルにする（拡張子なし）
            label_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # プロット実行
            ax.plot(x, y, label=label_name, color=colors[i], linewidth=1.5, alpha=0.9)
            
        except Exception as e:
            print(f"読込エラー: {file_path} -> {e}")

    # 5. スタイル設定
    
    # 目盛を外向きにする ('in', 'out', 'inout')
    ax.tick_params(direction='out', which='both', length=6, width=1)
    # 上と右の枠線にも目盛をつけるかどうか（お好みでTrue/False）
    ax.tick_params(top=False, right=False)

    # 軸ラベル
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # 軸範囲設定
    if x_lim: ax.set_xlim(x_lim)
    if y_lim: ax.set_ylim(y_lim)

    # 凡例の位置設定
    if legend_loc == 'outside':
        # 枠の外側（右）に配置
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
        plt.subplots_adjust(right=0.75) # 凡例のために右側に余白を作る
    else:
        # 枠の内側（最適な位置）
        ax.legend(loc='best', fontsize=10)

    # グリッド（お好みで）
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.title(f"Spectra Overlay (Colormap: {cmap_name if not manual_colors else 'Manual'})")
    
    # 保存と表示
    plt.savefig("spectra_result.png", dpi=300, bbox_inches='tight')
    print("グラフを 'spectra_result.png' に保存しました。")
    plt.show()

# --- (3) 実行ブロック ---
if __name__ == "__main__":
    # 1. 動作確認用にダミーデータを生成（初回のみ必要）
    create_dummy_data()

    # --- 設定例 A: デフォルト設定（UV1-UV7, カラーマップ: viridis, 凡例外側）---
    print("\n--- パターンA: デフォルト (Viridis) ---")
    plot_spectra()

    # --- 設定例 B: カラーマップ変更(jet) ＆ 凡例内側 ＆ 範囲指定 ---
    # print("\n--- パターンB: Jet, 凡例内側, 範囲指定 ---")
    # plot_spectra(
    #     cmap_name='jet',
    #     legend_loc='inside',
    #     x_lim=(200, 700),
    #     y_lim=(-0.1, 1.2)
    # )

    # --- 設定例 C: 手動色設定 ＆ txtファイル読み込み想定 ---
    # print("\n--- パターンC: 手動色設定 ---")
    # plot_spectra(
    #     file_list=['UV1.csv', 'UV3.csv', 'UV5.csv'], # 飛び飛びで指定
    #     manual_colors=['black', 'red', 'blue'],      # 色を直接指定
    #     legend_loc='inside'
    # )