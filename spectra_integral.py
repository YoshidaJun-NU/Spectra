import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ---------------------------------------------------------
# 定数定義
# ---------------------------------------------------------
# デフォルトのカラーパレット (Matplotlib tab10 hex codes)
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# 線種の表示名とMatplotlib記号の対応
LINE_STYLES = {
    'Solid (実線)': '-',
    'Dashed (破線)': '--',
    'Dash-dot (一点鎖線)': '-.',
    'Dotted (点線)': ':'
}

# ---------------------------------------------------------
# 関数定義: スタイルの初期化
# ---------------------------------------------------------
def init_styles(data_list):
    """データごとにデフォルトのスタイル情報をsession_stateに保存する"""
    if 'styles' not in st.session_state:
        st.session_state['styles'] = {}
    
    for i, item in enumerate(data_list):
        label = item['label']
        if label not in st.session_state['styles']:
            default_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            st.session_state['styles'][label] = {
                'color': default_color,
                'linewidth': 1.5,
                'linestyle': 'Solid (実線)'
            }

# ---------------------------------------------------------
# 関数定義: ダミーデータの生成
# ---------------------------------------------------------
def generate_dummy_data():
    """動作確認用に7つのガウス分布データを作成する"""
    data_list = []
    x = np.linspace(200, 800, 300) 
    
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
# 関数定義: ファイルデータの読み込み
# ---------------------------------------------------------
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    
    for uploaded_file in uploaded_files:
        try:
            # --- 1. 文字コードの自動判定 ---
            uploaded_file.seek(0)
            content_bytes = uploaded_file.read()
            uploaded_file.seek(0)

            encoding = 'utf-8'
            decoded_text = ""
            
            try:
                decoded_text = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    encoding = 'cp932'
                    decoded_text = content_bytes.decode('cp932')
                except UnicodeDecodeError:
                    encoding = 'latin1'
                    decoded_text = content_bytes.decode('latin1', errors='replace')

            # --- 2. 初期設定 ---
            use_sep = ',' if separator == 'comma' else '\t'
            use_skip = skip_rows
            use_header = 0 if has_header else None
            
            # --- 3. ファイル構造の解析 (XYDATA検出) ---
            if 'XYDATA' in decoded_text:
                lines = decoded_text.splitlines()
                for i, line in enumerate(lines):
                    if 'XYDATA' in line:
                        use_skip = i + 1
                        use_header = None 
                        use_sep = '\t'
                        break
            
            # --- 4. データの読み込み ---
            df = pd.read_csv(
                uploaded_file, 
                sep=use_sep, 
                skiprows=use_skip, 
                header=use_header,
                engine='python', 
                encoding=encoding
            )
            
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] < 2:
                st.warning(f"警告: {uploaded_file.name} から十分な列(2列以上)を読み込めませんでした。")
                continue

            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
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
    df_merged = pd.DataFrame({'Wavelength': data_list[0]['x'], data_list[0]['label']: data_list[0]['y']})
    
    for item in data_list[1:]:
        df_temp = pd.DataFrame({'Wavelength': item['x'], item['label']: item['y']})
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

    # --- サイドバー：1. データ読み込み設定 ---
    st.sidebar.header("1. データ読み込み設定")
    
    uploaded_files = st.sidebar.file_uploader(
        "ファイルをアップロード", 
        accept_multiple_files=True, 
        type=['txt', 'csv', 'dat', 'spz']
    )

    st.sidebar.subheader("フォーマット指定")
    st.sidebar.caption("※ 'XYDATA' を含むファイルは自動認識されます。")
    separator = st.sidebar.radio("区切り文字", ('comma', 'tab'), index=1, format_func=lambda x: "カンマ (CSV)" if x=='comma' else "タブ (TXT/DAT/SPZ)")
    skip_rows = st.sidebar.number_input("スキップする行数", value=19, min_value=0, help="デフォルトは19行です。")
    has_header = st.sidebar.checkbox("ヘッダー(列名)がある", value=True)

    if uploaded_files:
        st.session_state['data_list'] = load_data(uploaded_files, separator, skip_rows, has_header)
        init_styles(st.session_state['data_list'])

    st.sidebar.markdown("---")

    # --- サイドバー：2. 表示データの選択 ---
    st.sidebar.header("2. 表示データの選択")
    
    selected_labels = []
    if st.session_state['data_list']:
        all_labels = [d['label'] for d in st.session_state['data_list']]
        selected_labels = st.sidebar.multiselect(
            "プロットするファイルを選択",
            options=all_labels,
            default=all_labels
        )
    else:
        st.sidebar.info("データを読み込むとここにリストが表示されます。")

    st.sidebar.markdown("---")

    # --- サイドバー：3. グラフ設定 ---
    st.sidebar.header("3. グラフ設定")
    
    st.sidebar.subheader("前処理")
    do_normalize = st.sidebar.checkbox("正規化 (Min-Max Normalization)")

    st.sidebar.subheader("軸・凡例")
    x_label = st.sidebar.text_input("X軸ラベル", "Wavelength (nm)")
    y_label = st.sidebar.text_input("Y軸ラベル", "Norm. Abs." if do_normalize else "Abs.") 
    legend_loc = st.sidebar.radio("凡例の位置", ('Outside', 'Inside'))

    st.sidebar.subheader("プロット線スタイル")
    use_custom_style = st.sidebar.checkbox("個別スタイルを適用する", value=False)
    
    cmap_name = 'viridis' 
    if not use_custom_style:
        cmap_options = ['viridis', 'jet', 'coolwarm', 'rainbow', 'plasma', 'Manual']
        cmap_name = st.sidebar.selectbox("カラーマップ (自動割り当て)", cmap_options, index=0)
    else:
        st.sidebar.markdown("##### 各プロットの詳細設定")
        if not selected_labels:
            st.sidebar.warning("ファイルが選択されていません。")
        else:
            for label in selected_labels:
                if label not in st.session_state['styles']:
                    st.session_state['styles'][label] = {'color': '#000000', 'linewidth': 1.5, 'linestyle': 'Solid (実線)'}
                
                with st.sidebar.expander(f"🖊 {label}", expanded=False):
                    c1, c2 = st.columns(2)
                    st.session_state['styles'][label]['color'] = c1.color_picker("色", st.session_state['styles'][label]['color'], key=f"c_{label}")
                    st.session_state['styles'][label]['linewidth'] = c2.number_input("太さ", 0.5, 10.0, st.session_state['styles'][label]['linewidth'], step=0.5, key=f"w_{label}")
                    st.session_state['styles'][label]['linestyle'] = st.selectbox("線種", list(LINE_STYLES.keys()), index=list(LINE_STYLES.keys()).index(st.session_state['styles'][label]['linestyle']), key=f"s_{label}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("グリッド (目盛線) 設定")
    show_grid = st.sidebar.checkbox("グリッド線を表示", value=True)
    grid_params = {'color': '#b0b0b0', 'linewidth': 0.8, 'linestyle': ':'}
    
    if show_grid:
        c1, c2, c3 = st.sidebar.columns([1, 1, 2])
        grid_params['color'] = c1.color_picker("グリッド色", "#b0b0b0")
        grid_params['linewidth'] = c2.number_input("グリッド太さ", 0.1, 5.0, 0.8, 0.1)
        grid_ls_key = c3.selectbox("グリッド線種", list(LINE_STYLES.keys()), index=3)
        grid_params['linestyle'] = LINE_STYLES[grid_ls_key]

    st.sidebar.markdown("---")
    st.sidebar.subheader("軸範囲")
    use_manual_range = st.sidebar.checkbox("軸範囲を手動設定")
    x_min, x_max, y_min, y_max = None, None, None, None
    if use_manual_range:
        c1, c2 = st.sidebar.columns(2)
        x_min = c1.number_input("X Min", value=200.0)
        x_max = c2.number_input("X Max", value=800.0)
        default_ymin = -0.1 if not do_normalize else -0.05
        default_ymax = 1.5 if not do_normalize else 1.1
        y_min = c1.number_input("Y Min", value=default_ymin)
        y_max = c2.number_input("Y Max", value=default_ymax)

    # --- サイドバー：4. 解析 ---
    st.sidebar.header("4. 解析")
    do_calc_area = st.sidebar.checkbox("面積(積分)を計算", help="指定した波長範囲の曲線下の面積を計算します（台形積分）。")
    calc_start = 0.0
    calc_end = 0.0
    
    if do_calc_area:
        c1, c2 = st.sidebar.columns(2)
        calc_start = c1.number_input("開始波長 (nm)", value=300.0)
        calc_end = c2.number_input("終了波長 (nm)", value=500.0)
        if calc_start > calc_end:
            st.sidebar.warning("開始波長が終了波長より大きいため、入れ替えて計算します。")
            calc_start, calc_end = calc_end, calc_start

    # --- ダミーデータ生成コマンド ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### その他")
    if st.sidebar.button("ダミーデータをロード (Sample 1-7)"):
        st.session_state['data_list'] = generate_dummy_data()
        init_styles(st.session_state['data_list'])
        st.sidebar.success("ダミーデータを生成しました")
        st.rerun()

    # --- メインエリア ---
    full_data_list = st.session_state['data_list']
    target_data_list = [d for d in full_data_list if d['label'] in selected_labels]

    if target_data_list:
        # 表示用データの構築
        display_data_list = []
        for item in target_data_list:
            x_vals = item['x']
            y_vals = item['y'].copy()
            
            if do_normalize:
                min_y = np.min(y_vals)
                max_y = np.max(y_vals)
                if max_y - min_y != 0:
                    y_vals = (y_vals - min_y) / (max_y - min_y)
                else:
                    y_vals = y_vals - min_y 

            display_data_list.append({
                'label': item['label'],
                'x': x_vals,
                'y': y_vals
            })

        st.subheader(f"プロットプレビュー ({len(display_data_list)} samples)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # プロット処理
        num_files = len(display_data_list)
        colors_list = []
        
        if not use_custom_style:
            # 一括モード
            if cmap_name == 'Manual':
                base_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
                colors_list = base_colors * (num_files // len(base_colors) + 1)
            else:
                cmap = plt.get_cmap(cmap_name)
                colors_list = [cmap(i) for i in np.linspace(0, 1, num_files)]
            
            for i, item in enumerate(display_data_list):
                current_color = colors_list[i]
                ax.plot(item['x'], item['y'], label=item['label'], color=current_color, linewidth=1.5, alpha=0.8)
                
                # --- 積分エリアのシェーディング ---
                if do_calc_area:
                    mask = (item['x'] >= calc_start) & (item['x'] <= calc_end)
                    ax.fill_between(item['x'], item['y'], where=mask, color=current_color, alpha=0.2)
                # ------------------------------

        else:
            # 個別モード
            for item in display_data_list:
                style = st.session_state['styles'].get(item['label'], {'color':'black', 'linewidth':1.5, 'linestyle':'Solid (実線)'})
                ls_code = LINE_STYLES.get(style['linestyle'], '-')
                ax.plot(
                    item['x'], 
                    item['y'], 
                    label=item['label'], 
                    color=style['color'], 
                    linewidth=style['linewidth'], 
                    linestyle=ls_code,
                    alpha=0.9
                )
                
                # --- 積分エリアのシェーディング ---
                if do_calc_area:
                    mask = (item['x'] >= calc_start) & (item['x'] <= calc_end)
                    ax.fill_between(item['x'], item['y'], where=mask, color=style['color'], alpha=0.2)
                # ------------------------------

        # 積分範囲の縦線表示
        if do_calc_area:
            ax.axvline(calc_start, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(calc_end, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # 装飾
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(direction='out', length=6, width=1)
        
        if show_grid:
            ax.grid(True, color=grid_params['color'], linewidth=grid_params['linewidth'], linestyle=grid_params['linestyle'], alpha=0.5)
        else:
            ax.grid(False)

        if use_manual_range:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        if legend_loc == 'Outside':
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        else:
            ax.legend(loc='best')

        st.pyplot(fig)

        # --- 面積計算結果の表示 (修正版) ---
        if do_calc_area:
            st.markdown("---")
            st.subheader("📊 面積計算結果")
            st.caption(f"波長範囲: {calc_start} nm 〜 {calc_end} nm (台形積分)")
            
            area_results = []
            for item in display_data_list:
                # 範囲内のデータを抽出
                mask = (item['x'] >= calc_start) & (item['x'] <= calc_end)
                x_sub = item['x'][mask]
                y_sub = item['y'][mask]
                
                # データが存在する場合のみ積分
                if len(x_sub) > 1:
                    sort_idx = np.argsort(x_sub)
                    
                    # --- NumPy 2.0対応の変更箇所 ---
                    if hasattr(np, 'trapezoid'):
                         area = np.trapezoid(y_sub[sort_idx], x_sub[sort_idx])
                    else:
                         area = np.trapz(y_sub[sort_idx], x_sub[sort_idx])
                    # ----------------------------

                    area_results.append({'ファイル名': item['label'], '面積': area})
                else:
                    area_results.append({'ファイル名': item['label'], '面積': 0.0})
            
            if area_results:
                df_area = pd.DataFrame(area_results)
                st.dataframe(df_area, use_container_width=True)


        # --- ダウンロード ---
        st.markdown("---")
        st.subheader("📥 ダウンロード (表示中のデータのみ)")
        
        col1, col2, col3 = st.columns(3)

        img_png = io.BytesIO()
        plt.savefig(img_png, format='png', bbox_inches='tight', dpi=300)
        img_png.seek(0)
        col1.download_button("画像 (PNG)", data=img_png, file_name="plot.png", mime="image/png")

        img_tiff = io.BytesIO()
        plt.savefig(img_tiff, format='tiff', bbox_inches='tight', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        img_tiff.seek(0)
        col2.download_button("画像 (TIFF)", data=img_tiff, file_name="plot.tiff", mime="image/tiff")

        gnu_data = create_gnuplot_data(display_data_list)
        if gnu_data:
            fname = "data_normalized.dat" if do_normalize else "data.dat"
            col3.download_button(f"データファイル ({fname})", data=gnu_data, file_name=fname, mime="text/plain")
            
    else:
        if full_data_list:
             st.warning("👈 サイドバーで表示するファイルを選択してください。")
        else:
             st.info("👈 左側のサイドバーからファイルをアップロードしてください。")

if __name__ == "__main__":
    main()