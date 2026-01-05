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
# 関数定義: ファイルデータの読み込み (自動検出機能付き)
# ---------------------------------------------------------
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    
    for uploaded_file in uploaded_files:
        try:
            # --- 初期設定 (サイドバーの値を使用) ---
            use_sep = ',' if separator == 'comma' else '\t'
            use_skip = skip_rows
            use_header = 0 if has_header else None
            
            # --- ファイル形式の自動判別ロジック ---
            # ファイルポインタを先頭に戻す
            uploaded_file.seek(0)
            
            # 先頭の数キロバイトを読み込んで中身をチェック
            # (大きなファイルでも最初だけ読めば形式判別できるため)
            preview_bytes = uploaded_file.read(4096)
            uploaded_file.seek(0) # 読み込み後に必ずポインタを戻す

            # 文字コードの推定 (utf-8 でダメなら shift_jis)
            try:
                preview_text = preview_bytes.decode('utf-8')
            except UnicodeDecodeError:
                preview_text = preview_bytes.decode('shift_jis', errors='replace')

            # 【追加機能】JASCO形式などの "XYDATA" キーワード検出
            if 'XYDATA' in preview_text:
                lines = preview_text.splitlines()
                for i, line in enumerate(lines):
                    if 'XYDATA' in line:
                        # XYDATAのある行の"次の行"からデータが始まるとみなす
                        use_skip = i + 1
                        # この形式は通常ヘッダー行を持たないのでNoneにする
                        use_header = None 
                        # JASCO形式は通常タブ区切り
                        use_sep = '\t'
                        break
            
            # --- データの読み込み ---
            df = pd.read_csv(
                uploaded_file, 
                sep=use_sep, 
                skiprows=use_skip, 
                header=use_header,
                engine='python' # 柔軟なパースのためpythonエンジンを指定
            )
            
            # データの抽出 (1列目をX, 2列目をYとする)
            # 型変換を試みて、数値でないデータが含まれている場合のエラーを防ぐ
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] < 2:
                st.warning(f"警告: {uploaded_file.name} から十分な列(2列以上)を読み込めませんでした。区切り文字設定などを確認してください。")
                continue

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
        # 外部結合でマージ (波長が完全に一致しない場合も考慮)
        df_merged = pd.merge(df_merged, df_temp, on='Wavelength', how='outer')
    
    df_merged = df_merged.sort_values('Wavelength')
    return df_merged.to_csv(sep='\t', index=False, float_format='%.4f')

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Plotter", layout="wide")
    st.title("Spectra Viewer")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：データ設定 ---
    st.sidebar.header("1. データ読み込み設定")
    
    # 1-1. ファイルフォーマット設定
    st.sidebar.subheader("フォーマット指定")
    st.sidebar.caption("※ 'XYDATA' を含むファイルは自動認識されます。")
    separator = st.sidebar.radio("区切り文字", ('comma', 'tab'), format_func=lambda x: "カンマ (CSV)" if x=='comma' else "タブ (TXT/DAT)")
    
    skip_rows = st.sidebar.number_input("スキップする行数", value=0, min_value=0, help="ファイルの先頭から無視する行数を指定します（自動認識時は無視されます）。")
    has_header = st.sidebar.checkbox("ヘッダー(列名)がある", value=True, help="チェックを外すと、スキップ後の1行目からデータとして読み込みます。")

    st.sidebar.markdown("---")

    # 1-2. データソース
    if st.sidebar.button("ダミーデータをロード (Sample 1-7)"):
        st.session_state['data_list'] = generate_dummy_data()
        st.sidebar.success("ダミーデータを生成しました")

    uploaded_files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True, type=['csv', 'txt', 'dat'])
    
    if uploaded_files:
        st.session_state['data_list'] = load_data(uploaded_files, separator, skip_rows, has_header)

    # --- サイドバー：グラフ設定 ---
    st.sidebar.header("2. グラフ設定")
    
    # --- 前処理設定 ---
    st.sidebar.subheader("前処理")
    do_normalize = st.sidebar.checkbox("正規化 (Min-Max Normalization)", help="各データの最小値を0、最大値を1にスケーリングして表示・保存します。")
    # -----------------------

    cmap_options = ['viridis', 'jet', 'coolwarm', 'rainbow', 'plasma', 'Manual']
    cmap_name = st.sidebar.selectbox("カラーマップ", cmap_options, index=0)
    legend_loc = st.sidebar.radio("凡例の位置", ('Outside', 'Inside'))
    x_label = st.sidebar.text_input("X軸ラベル", "Wavelength (nm)")
    y_label = st.sidebar.text_input("Y軸ラベル", "Norm. Abs." if do_normalize else "Abs.") 
    
    use_manual_range = st.sidebar.checkbox("軸範囲を手動設定")
    x_min, x_max, y_min, y_max = None, None, None, None
    if use_manual_range:
        c1, c2 = st.sidebar.columns(2)
        x_min = c1.number_input("X Min", value=200.0)
        x_max = c2.number_input("X Max", value=800.0)
        # 正規化時はデフォルト範囲を変更
        default_ymin = -0.1 if not do_normalize else -0.05
        default_ymax = 1.5 if not do_normalize else 1.1
        y_min = c1.number_input("Y Min", value=default_ymin)
        y_max = c2.number_input("Y Max", value=default_ymax)

    # --- メインエリア ---
    raw_data_list = st.session_state['data_list']

    if raw_data_list:
        # --- 表示用データの構築（正規化処理） ---
        display_data_list = []
        for item in raw_data_list:
            x_vals = item['x']
            y_vals = item['y'].copy() # 元データを壊さないようにコピー
            
            if do_normalize:
                min_y = np.min(y_vals)
                max_y = np.max(y_vals)
                # ゼロ除算回避
                if max_y - min_y != 0:
                    y_vals = (y_vals - min_y) / (max_y - min_y)
                else:
                    y_vals = y_vals - min_y 

            display_data_list.append({
                'label': item['label'],
                'x': x_vals,
                'y': y_vals
            })
        # ---------------------------------------------

        st.subheader(f"プロットプレビュー ({len(display_data_list)} samples)")
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_files = len(display_data_list)
        if cmap_name == 'Manual':
            base_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
            colors = base_colors * (num_files // len(base_colors) + 1)
        else:
            cmap = plt.get_cmap(cmap_name)
            colors = [cmap(i) for i in np.linspace(0, 1, num_files)]

        # 表示用リストを使ってプロット
        for i, item in enumerate(display_data_list):
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
        gnu_data = create_gnuplot_data(display_data_list)
        if gnu_data:
            fname = "data_normalized.dat" if do_normalize else "data.dat"
            col3.download_button(f"データファイル ({fname})", data=gnu_data, file_name=fname, mime="text/plain")
            
    else:
        st.info("👈 左側のサイドバーからファイルをアップロードしてください。")

if __name__ == "__main__":
    main()