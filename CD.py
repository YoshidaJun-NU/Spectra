import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ---------------------------------------------------------
# 関数定義: CD用ダミーデータの生成 (正負のあるデータ)
# ---------------------------------------------------------
def generate_cd_dummy_data():
    """動作確認用にCDスペクトル（正負あり）を作成する"""
    data_list = []
    x = np.linspace(200, 350, 300) # CDでよく見る波長範囲
    
    # パターン1: 正のコットン効果 (Type A)
    y1 = 20 * np.exp(-((x - 280)**2) / (2 * 10**2)) - 10 * np.exp(-((x - 220)**2) / (2 * 15**2))
    y1 += np.random.normal(0, 0.2, len(x)) # ノイズ
    
    # パターン2: 負のコットン効果 (Type B: 鏡像に近い形)
    y2 = -15 * np.exp(-((x - 280)**2) / (2 * 10**2)) + 12 * np.exp(-((x - 225)**2) / (2 * 15**2))
    y2 += np.random.normal(0, 0.2, len(x))

    # データセット作成
    data_list.append({'label': 'Type_A_Protein', 'x': x, 'y': y1})
    data_list.append({'label': 'Type_B_Mutant', 'x': x, 'y': y2})
    
    return data_list

# ---------------------------------------------------------
# 関数定義: ファイル読み込み
# ---------------------------------------------------------
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    for uploaded_file in uploaded_files:
        try:
            sep_char = ',' if separator == 'comma' else '\t'
            header_setting = 0 if has_header else None
            
            df = pd.read_csv(uploaded_file, sep=sep_char, skiprows=skip_rows, header=header_setting)
            
            # データ抽出 (1列目X, 2列目Y)
            # エラー防止のため、数値変換可能な行だけ残す処理を入れるのが理想ですが、今回はシンプルに
            x = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
            y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
            
            # NaN除去 (数値変換できなかった行を削除)
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            
            label = uploaded_file.name.rsplit('.', 1)[0]
            data_list.append({'label': label, 'x': x, 'y': y})
            
        except Exception as e:
            st.error(f"エラー: {uploaded_file.name} を読み込めませんでした。\n{e}")
    return data_list

# ---------------------------------------------------------
# 関数定義: Gnuplotデータ作成
# ---------------------------------------------------------
def create_gnuplot_data(data_list):
    if not data_list: return None
    df_merged = pd.DataFrame({'Wavelength': data_list[0]['x'], data_list[0]['label']: data_list[0]['y']})
    for item in data_list[1:]:
        df_temp = pd.DataFrame({'Wavelength': item['x'], item['label']: item['y']})
        df_merged = pd.merge(df_merged, df_temp, on='Wavelength', how='outer')
    return df_merged.sort_values('Wavelength').to_csv(sep='\t', index=False, float_format='%.4f')

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="CD Spectra Plotter", layout="wide")
    st.title("CD Spectra Plotter (Circular Dichroism)")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー ---
    st.sidebar.header("1. データ読み込み")
    
    # ダミーデータ (2種類生成)
    if st.sidebar.button("サンプルデータをロード (Type A & B)"):
        st.session_state['data_list'] = generate_cd_dummy_data()
        st.sidebar.success("CDスペクトル（正・負）を生成しました")

    st.sidebar.markdown("---")
    
    # ファイル設定
    separator = st.sidebar.radio("区切り文字", ('comma', 'tab'), format_func=lambda x: "カンマ (CSV)" if x=='comma' else "タブ (TXT)")
    skip_rows = st.sidebar.number_input("スキップ行数", value=0, min_value=0)
    has_header = st.sidebar.checkbox("ヘッダーあり", value=True)
    
    uploaded_files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True)
    if uploaded_files:
        st.session_state['data_list'] = load_data(uploaded_files, separator, skip_rows, has_header)

    # --- グラフ設定 ---
    st.sidebar.header("2. グラフ設定")
    
    # 2色プロットのための色設定
    color_mode = st.sidebar.selectbox("配色モード", ["Auto (Distinct)", "Manual", "CoolWarm (Gradation)"])
    
    manual_colors = []
    if color_mode == "Manual":
        st.sidebar.markdown("各データの色を指定:")
        # データ数分だけカラーピッカーを出す（最大5つまで表示などの制限も可能だが、ここではループで表示）
        # データがまだない時はデフォルト2色を表示
        current_data_count = len(st.session_state['data_list']) if st.session_state['data_list'] else 2
        default_cols = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd'] # 青, 赤, 緑...
        
        for i in range(current_data_count):
            col_val = default_cols[i % len(default_cols)]
            c = st.sidebar.color_picker(f"データ {i+1} の色", col_val)
            manual_colors.append(c)

    # 軸設定
    x_label = st.sidebar.text_input("X軸ラベル", "Wavelength (nm)")
    y_label = st.sidebar.text_input("Y軸ラベル", "Ellipticity (mdeg)")
    
    # 範囲
    use_manual_range = st.sidebar.checkbox("軸範囲を手動設定")
    x_min, x_max, y_min, y_max = None, None, None, None
    if use_manual_range:
        c1, c2 = st.sidebar.columns(2)
        x_min = c1.number_input("X Min", value=200.0)
        x_max = c2.number_input("X Max", value=350.0)
        y_min = c1.number_input("Y Min", value=-20.0)
        y_max = c2.number_input("Y Max", value=20.0)

    # --- プロット処理 ---
    data_list = st.session_state['data_list']
    if data_list:
        st.subheader("CD Spectra Overlay")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # ゼロ線の描画 (CDスペクトルで重要)
        ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)

        # 色の決定
        num_files = len(data_list)
        colors = []
        if color_mode == "Manual":
            colors = manual_colors
        elif color_mode == "Auto (Distinct)":
            # 視認性の良いタブローカラーを使用 (青、オレンジ、緑、赤...)
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        else:
            # グラデーション
            colors = [plt.cm.coolwarm(i) for i in np.linspace(0, 1, num_files)]

        # ループ描画
        for i, item in enumerate(data_list):
            c = colors[i % len(colors)] # 色が足りない場合はループ
            ax.plot(item['x'], item['y'], label=item['label'], color=c, linewidth=2.0, alpha=0.8)

        # 装飾
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(direction='out', top=False, right=False)
        
        if use_manual_range:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        plt.tight_layout()

        st.pyplot(fig)

        # --- ダウンロード ---
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        # PNG
        img_png = io.BytesIO()
        plt.savefig(img_png, format='png', dpi=300)
        img_png.seek(0)
        col1.download_button("画像 (PNG)", img_png, "cd_spectra.png", "image/png")
        
        # TIFF
        img_tiff = io.BytesIO()
        plt.savefig(img_tiff, format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        img_tiff.seek(0)
        col2.download_button("画像 (TIFF)", img_tiff, "cd_spectra.tiff", "image/tiff")
        
        # Gnuplot Data
        gnu_data = create_gnuplot_data(data_list)
        if gnu_data:
            col3.download_button("Gnuplotデータ (.dat)", gnu_data, "cd_data.dat", "text/plain")
            
    else:
        st.info("👈 左側の「サンプルデータをロード」を押すか、ファイルをアップロードしてください。")

if __name__ == "__main__":
    main()