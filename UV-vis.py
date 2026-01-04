import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ---------------------------------------------------------
# 関数定義: ダミーデータの生成
# ---------------------------------------------------------
def generate_dummy_data():
    """動作確認用に7つのガウス分布データを作成する"""
    data_list = []
    x = np.linspace(200, 800, 300) # 200nm ~ 800nm
    
    for i in range(1, 8):
        center = 300 + (i * 40)
        height = 0.5 + (i * 0.1)
        width = 40
        y = height * np.exp(-((x - center)**2) / (2 * width**2))
        y += np.random.normal(0, 0.002, len(x))
        
        df = pd.DataFrame({'Wavelength': x, 'Abs': y})
        data_list.append({
            'label': f'Dummy_Sample_{i}',
            'x': x,
            'y': y,
            'df_raw': df
        })
    return data_list

# ---------------------------------------------------------
# 関数定義: ファイルデータの読み込み (更新)
# ---------------------------------------------------------
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    
    for uploaded_file in uploaded_files:
        try:
            # 区切り文字の設定
            sep_char = ',' if separator == 'comma' else '\t'
            
            # ヘッダー設定: チェックがあれば0行目(スキップ後)をヘッダーにする、なければNone
            header_setting = 0 if has_header else None
            
            # データの読み込み
            # skiprows: 指定した行数分だけ先頭から無視する
            df = pd.read_csv(
                uploaded_file, 
                sep=sep_char, 
                skiprows=skip_rows, 
                header=header_setting
            )
            
            # データの抽出 (エラー回避のため、強制的に数値型に変換できるかチェックするとより堅牢ですが、ここでは簡易的にilocを使用)
            # 1列目をX, 2列目をYとする
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
            # ファイル名を取得
            label = uploaded_file.name.rsplit('.', 1)[0]
            
            data_list.append({
                'label': label,
                'x': x,
                'y': y,
                'df_raw': df
            })
            
        except Exception as e:
            st.error(f"エラー: {uploaded_file.name} を読み込めませんでした。\n(詳細: {e})")
            
    return data_list

# ---------------------------------------------------------
# 関数定義: Gnuplot用データの作成
# ---------------------------------------------------------
def create_gnuplot_data(data_list):
    if not data_list:
        return None
    # 結合用のベースデータ
    df_merged = pd.DataFrame({'Wavelength': data_list[0]['x'], data_list[0]['label']: data_list[0]['y']})
    
    for item in data_list[1:]:
        df_temp = pd.DataFrame({'Wavelength': item['x'], item['label']: item['y']})
        # 外部結合でマージ
        df_merged = pd.merge(df_merged, df_temp, on='Wavelength', how='outer')
    
    df_merged = df_merged.sort_values('Wavelength')
    return df_merged.to_csv(sep='\t', index=False, float_format='%.4f')

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Plotter", layout="wide")
    st.title("Spectra Viewer (UV, 蛍光，IRなど)")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：データ設定 ---
    st.sidebar.header("1. データ読み込み設定")
    
    # 1-1. ファイルフォーマット設定
    st.sidebar.subheader("フォーマット指定")
    separator = st.sidebar.radio("区切り文字", ('comma', 'tab'), format_func=lambda x: "カンマ (CSV)" if x=='comma' else "タブ (TXT/DAT)")
    
    # 【変更点】スキップ行数とヘッダー有無の設定
    skip_rows = st.sidebar.number_input("スキップする行数 (メタデータなど)", value=0, min_value=0, help="ファイルの先頭から無視する行数を指定します。")
    has_header = st.sidebar.checkbox("ヘッダー(列名)がある", value=True, help="チェックを外すと、スキップ後の1行目からデータとして読み込みます。")

    st.sidebar.markdown("---")

    # 1-2. データソース
    if st.sidebar.button("ダミーデータをロード (Sample 1-7)"):
        st.session_state['data_list'] = generate_dummy_data()
        st.sidebar.success("ダミーデータを生成しました")

    uploaded_files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True, type=['csv', 'txt', 'dat'])
    
    if uploaded_files:
        # アップロード時に設定に基づいて読み込み直す
        st.session_state['data_list'] = load_data(uploaded_files, separator, skip_rows, has_header)

    # --- サイドバー：グラフ設定 ---
    st.sidebar.header("2. グラフ設定")
    cmap_options = ['viridis', 'jet', 'coolwarm', 'rainbow', 'plasma', 'Manual']
    cmap_name = st.sidebar.selectbox("カラーマップ", cmap_options, index=0)
    legend_loc = st.sidebar.radio("凡例の位置", ('Outside', 'Inside'))
    x_label = st.sidebar.text_input("X軸ラベル", "Wavelength (nm)")
    y_label = st.sidebar.text_input("Y軸ラベル", "Abs.")
    
    use_manual_range = st.sidebar.checkbox("軸範囲を手動設定")
    x_min, x_max, y_min, y_max = None, None, None, None
    if use_manual_range:
        c1, c2 = st.sidebar.columns(2)
        x_min = c1.number_input("X Min", value=200.0)
        x_max = c2.number_input("X Max", value=800.0)
        y_min = c1.number_input("Y Min", value=-0.1)
        y_max = c2.number_input("Y Max", value=1.5)

    # --- メインエリア ---
    data_list = st.session_state['data_list']

    if data_list:
        st.subheader(f"プロットプレビュー ({len(data_list)} samples)")
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_files = len(data_list)
        if cmap_name == 'Manual':
            base_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
            colors = base_colors * (num_files // len(base_colors) + 1)
        else:
            cmap = plt.get_cmap(cmap_name)
            colors = [cmap(i) for i in np.linspace(0, 1, num_files)]

        for i, item in enumerate(data_list):
            ax.plot(item['x'], item['y'], label=item['label'], color=colors[i], linewidth=1.5, alpha=0.8)

        # 装飾
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(direction='out', length=6, width=1)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        if use_manual_range:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        if legend_loc == 'Outside':
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        else:
            ax.legend(loc='best')

        st.pyplot(fig)

        # --- ダウンロードエリア ---
        st.markdown("---")
        st.subheader("📥 ダウンロード")
        
        col1, col2, col3 = st.columns(3)

        # PNG
        img_png = io.BytesIO()
        plt.savefig(img_png, format='png', bbox_inches='tight', dpi=300)
        img_png.seek(0)
        col1.download_button("画像 (PNG)", data=img_png, file_name="plot.png", mime="image/png")

        # TIFF
        img_tiff = io.BytesIO()
        plt.savefig(img_tiff, format='tiff', bbox_inches='tight', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        img_tiff.seek(0)
        col2.download_button("画像 (TIFF)", data=img_tiff, file_name="plot.tiff", mime="image/tiff")

        # Gnuplot
        gnu_data = create_gnuplot_data(data_list)
        if gnu_data:
            col3.download_button("Gnuplotデータ (.dat)", data=gnu_data, file_name="data.dat", mime="text/plain")
            
    else:
        st.info("👈 左側のサイドバーからファイルをアップロードしてください。")

if __name__ == "__main__":
    main()