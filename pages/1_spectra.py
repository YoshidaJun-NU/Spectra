import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks

# ---------------------------------------------------------
# 定数定義
# ---------------------------------------------------------
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

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
    data_list = []
    x = np.linspace(200, 800, 300) 
    
    for i in range(1, 8):
        center = 300 + (i * 40)
        height = 0.5 + (i * 0.1)
        width = 40
        y = height * np.exp(-((x - center)**2) / (2 * width**2))
        y += np.random.normal(0, 0.002, len(x))
        y += 0.05 * np.exp(-((x - (center - 50))**2) / (2 * 5**2))
        
        df = pd.DataFrame({'Wavelength': x, 'Intensity': y})
        data_list.append({
            'label': f'Dummy_Sample_{i}',
            'x': x,
            'y': y,
            'df_raw': df
        })
    return data_list

# ---------------------------------------------------------
# 関数定義: ファイルデータの読み込み (JASCO CSV/TXT 対応強化版)
# ---------------------------------------------------------
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    
    for uploaded_file in uploaded_files:
        try:
            # ファイルのバイナリ読み込み
            content_bytes = uploaded_file.getvalue()
            
            # エンコーディングの試行
            decoded_text = ""
            encoding_found = None
            for enc in ['utf-8', 'cp932', 'shift_jis', 'latin1']:
                try:
                    decoded_text = content_bytes.decode(enc)
                    encoding_found = enc
                    break
                except UnicodeDecodeError:
                    continue

            if not encoding_found:
                st.error(f"エラー: {uploaded_file.name} の文字コードを判別できませんでした。")
                continue

            # デフォルト設定
            use_sep = ',' if separator == 'comma' else '\t'
            use_skip = skip_rows
            use_header = 0 if has_header else None
            
            # --- JASCO形式 (XYDATA) の自動検知ロジック ---
            if 'XYDATA' in decoded_text:
                lines = decoded_text.splitlines()
                for i, line in enumerate(lines):
                    if 'XYDATA' in line:
                        use_skip = i + 1
                        use_header = None
                        # XYDATA以降の区切り文字を拡張子で自動判別
                        if uploaded_file.name.lower().endswith('.csv'):
                            use_sep = ','
                        else:
                            use_sep = None # Pandasに推論させる (タブ・スペース混在対応)
                        break
            
            # Pandasで読み込み
            df = pd.read_csv(
                io.StringIO(decoded_text), 
                sep=use_sep, 
                skiprows=use_skip, 
                header=use_header,
                engine='python'
            )
            
            # 数値以外の行を除去し、欠損値を削除
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] < 2:
                st.warning(f"警告: {uploaded_file.name} に数値データが見つかりませんでした。設定を確認してください。")
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
            st.error(f"エラー: {uploaded_file.name} の解析中に問題が発生しました。\n({e})")
            
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
    st.title("Spectral Viewer 📈")

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
    st.sidebar.caption("※ 'XYDATA' を含むJASCO形式などは自動認識されます。")
    separator = st.sidebar.radio("区切り文字 (通常時)", ('comma', 'tab'), index=1, format_func=lambda x: "カンマ (CSV)" if x=='comma' else "タブ (TXT/DAT/SPZ)")
    skip_rows = st.sidebar.number_input("スキップする行数 (通常時)", value=19, min_value=0)
    has_header = st.sidebar.checkbox("ヘッダー(列名)がある", value=True)

    if uploaded_files:
        # アップロードのたびにロード
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

    st.sidebar.subheader("プロットスタイル")
    use_custom_style = st.sidebar.checkbox("個別スタイルを適用する", value=False)
    
    cmap_name = 'viridis' 
    if not use_custom_style:
        cmap_options = ['viridis', 'jet', 'coolwarm', 'rainbow', 'plasma', 'Manual']
        cmap_name = st.sidebar.selectbox("カラーマップ", cmap_options, index=0)
    else:
        if selected_labels:
            for label in selected_labels:
                if label not in st.session_state['styles']:
                    st.session_state['styles'][label] = {'color': '#000000', 'linewidth': 1.5, 'linestyle': 'Solid (実線)'}
                with st.sidebar.expander(f"🖊 {label}", expanded=False):
                    c1, c2 = st.columns(2)
                    st.session_state['styles'][label]['color'] = c1.color_picker("色", st.session_state['styles'][label]['color'], key=f"c_{label}")
                    st.session_state['styles'][label]['linewidth'] = c2.number_input("太さ", 0.5, 10.0, st.session_state['styles'][label]['linewidth'], step=0.5, key=f"w_{label}")
                    st.session_state['styles'][label]['linestyle'] = st.selectbox("線種", list(LINE_STYLES.keys()), index=list(LINE_STYLES.keys()).index(st.session_state['styles'][label]['linestyle']), key=f"s_{label}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("グリッド設定")
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
        y_min = c1.number_input("Y Min", value=-0.1 if not do_normalize else -0.05)
        y_max = c2.number_input("Y Max", value=1.5 if not do_normalize else 1.1)

    # --- サイドバー：4. 解析 ---
    st.sidebar.header("4. 解析")
    do_calc_area = st.sidebar.checkbox("面積(積分)を計算")
    if do_calc_area:
        c1, c2 = st.sidebar.columns(2)
        calc_start = c1.number_input("開始波長 (nm)", value=300.0)
        calc_end = c2.number_input("終了波長 (nm)", value=500.0)
        if calc_start > calc_end: calc_start, calc_end = calc_end, calc_start
            
    st.sidebar.markdown("---")
    do_peak_search = st.sidebar.checkbox("ピーク検出を行う")
    if do_peak_search:
        peak_prominence = st.sidebar.number_input("プロミネンス (感度)", value=0.01, format="%.4f", step=0.001)
        peak_min_height = st.sidebar.number_input("最小高さ", value=0.0, format="%.2f")
        peak_distance = st.sidebar.number_input("最小距離 (点数)", value=10, min_value=1)

    # --- ダミーデータボタン ---
    st.sidebar.markdown("---")
    if st.sidebar.button("ダミーデータをロード"):
        st.session_state['data_list'] = generate_dummy_data()
        init_styles(st.session_state['data_list'])
        st.rerun()

    # --- メインエリア描画 ---
    full_data_list = st.session_state['data_list']
    target_data_list = [d for d in full_data_list if d['label'] in selected_labels]

    if target_data_list:
        display_data_list = []
        for item in target_data_list:
            x_vals = item['x']
            y_vals = item['y'].copy()
            if do_normalize:
                min_y, max_y = np.min(y_vals), np.max(y_vals)
                if max_y - min_y != 0: y_vals = (y_vals - min_y) / (max_y - min_y)
            display_data_list.append({'label': item['label'], 'x': x_vals, 'y': y_vals})

        st.subheader(f"プロットプレビュー ({len(display_data_list)} samples)")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # カラー設定
        num_files = len(display_data_list)
        if not use_custom_style:
            if cmap_name == 'Manual':
                colors_list = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown'] * (num_files//7 + 1)
            else:
                cmap = plt.get_cmap(cmap_name)
                colors_list = [cmap(i) for i in np.linspace(0, 1, num_files)]
        
        peak_results = []
        for i, item in enumerate(display_data_list):
            if not use_custom_style:
                color, lw, ls = colors_list[i], 1.5, '-'
            else:
                style = st.session_state['styles'].get(item['label'], {'color':'black', 'linewidth':1.5, 'linestyle':'Solid (実線)'})
                color, lw, ls = style['color'], style['linewidth'], LINE_STYLES.get(style['linestyle'], '-')

            ax.plot(item['x'], item['y'], label=item['label'], color=color, linewidth=lw, linestyle=ls, alpha=0.8)
            
            if do_calc_area:
                mask = (item['x'] >= calc_start) & (item['x'] <= calc_end)
                ax.fill_between(item['x'], item['y'], where=mask, color=color, alpha=0.1)

            if do_peak_search:
                peaks, _ = find_peaks(item['y'], height=peak_min_height, prominence=peak_prominence, distance=peak_distance)
                if len(peaks) > 0:
                    ax.plot(item['x'][peaks], item['y'][peaks], "v", color=color, markersize=8, markeredgecolor='black')
                    for p_idx in peaks:
                        peak_results.append({'ファイル名': item['label'], '波長 (nm)': item['x'][p_idx], '値': item['y'][p_idx]})

        # 装飾
        ax.set_xlabel(x_label, fontsize=12); ax.set_ylabel(y_label, fontsize=12)
        if show_grid: ax.grid(True, **grid_params, alpha=0.5)
        if use_manual_range: ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        if legend_loc == 'Outside': ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else: ax.legend(loc='best')

        st.pyplot(fig)

        # 解析結果
        if do_calc_area:
            st.markdown("---"); st.subheader("📊 面積計算結果")
            area_results = []
            for item in display_data_list:
                mask = (item['x'] >= calc_start) & (item['x'] <= calc_end)
                x_s, y_s = item['x'][mask], item['y'][mask]
                if len(x_s) > 1:
                    idx = np.argsort(x_s)
                    area = np.trapezoid(y_s[idx], x_s[idx]) if hasattr(np, 'trapezoid') else np.trapz(y_s[idx], x_s[idx])
                    area_results.append({'ファイル名': item['label'], '面積': area})
            st.dataframe(pd.DataFrame(area_results), use_container_width=True)

        if do_peak_search:
            st.markdown("---"); st.subheader("🏔 ピーク検出結果")
            if peak_results:
                st.dataframe(pd.DataFrame(peak_results).sort_values(['ファイル名', '波長 (nm)']), use_container_width=True)
            else:
                st.info("ピークは見つかりませんでした。")

        # ダウンロード
        st.markdown("---"); st.subheader("📥 ダウンロード")
        c1, c2, c3 = st.columns(3)
        img_png = io.BytesIO(); plt.savefig(img_png, format='png', bbox_inches='tight', dpi=300); c1.download_button("PNG保存", img_png, "plot.png")
        img_tiff = io.BytesIO(); plt.savefig(img_tiff, format='tiff', bbox_inches='tight', dpi=300); c2.download_button("TIFF保存", img_tiff, "plot.tiff")
        gnu_data = create_gnuplot_data(display_data_list)
        if gnu_data: c3.download_button("DATファイル保存", gnu_data, "data.dat")
            
    else:
        st.info("👈 ファイルをアップロードして、表示するデータを選択してください。")

if __name__ == "__main__":
    main()