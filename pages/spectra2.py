import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
import plotly.graph_objects as go # Plotlyの追加

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
            content_bytes = uploaded_file.getvalue()
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

            use_sep = ',' if separator == 'comma' else '\t'
            use_skip = skip_rows
            use_header = 0 if has_header else None
            
            if 'XYDATA' in decoded_text:
                lines = decoded_text.splitlines()
                for i, line in enumerate(lines):
                    if 'XYDATA' in line:
                        use_skip = i + 1
                        use_header = None
                        if uploaded_file.name.lower().endswith('.csv'):
                            use_sep = ','
                        else:
                            use_sep = None
                        break
            
            df = pd.read_csv(
                io.StringIO(decoded_text), 
                sep=use_sep, 
                skiprows=use_skip, 
                header=use_header,
                engine='python'
            )
            
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

    # --- サイドバー：データ読み込み設定 ---
    st.sidebar.header("1. データ読み込み設定")
    uploaded_files = st.sidebar.file_uploader(
        "ファイルをアップロード", 
        accept_multiple_files=True, 
        type=['txt', 'csv', 'dat', 'spz']
    )
    st.sidebar.caption("※ 'XYDATA' を含むJASCO形式などは自動認識されます。")
    separator = st.sidebar.radio("区切り文字 (通常時)", ('comma', 'tab'), index=1, format_func=lambda x: "カンマ (CSV)" if x=='comma' else "タブ (TXT/DAT/SPZ)")
    skip_rows = st.sidebar.number_input("スキップ行数", value=19, min_value=0)
    has_header = st.sidebar.checkbox("ヘッダーあり", value=True)

    if uploaded_files:
        st.session_state['data_list'] = load_data(uploaded_files, separator, skip_rows, has_header)
        init_styles(st.session_state['data_list'])
    
    # ダミーデータボタン
    if st.sidebar.button("ダミーデータをロード"):
        st.session_state['data_list'] = generate_dummy_data()
        init_styles(st.session_state['data_list'])
        st.rerun()

    st.sidebar.markdown("---")

    # --- サイドバー：表示データの選択 ---
    st.sidebar.header("2. 表示データの選択")
    selected_labels = []
    if st.session_state['data_list']:
        all_labels = [d['label'] for d in st.session_state['data_list']]
        selected_labels = st.sidebar.multiselect("プロット対象", options=all_labels, default=all_labels)
    else:
        st.sidebar.info("データを読み込んでください。")

    st.sidebar.markdown("---")

    # --- サイドバー：共通グラフ設定 ---
    st.sidebar.header("3. グラフ共通設定")
    do_normalize = st.sidebar.checkbox("正規化 (Min-Max)", value=False)
    
    # --- データ前処理 ---
    full_data_list = st.session_state['data_list']
    target_data_list = [d for d in full_data_list if d['label'] in selected_labels]
    
    display_data_list = []
    for item in target_data_list:
        x_vals = item['x']
        y_vals = item['y'].copy()
        if do_normalize:
            min_y, max_y = np.min(y_vals), np.max(y_vals)
            if max_y - min_y != 0: y_vals = (y_vals - min_y) / (max_y - min_y)
        display_data_list.append({'label': item['label'], 'x': x_vals, 'y': y_vals})

    # =========================================================
    # タブによるモード切替
    # =========================================================
    tab_display, tab_analysis = st.tabs(["📊 表示モード (印刷・保存用)", "🔍 解析モード (積分・ピーク検出)"])

    # ---------------------------------------------------------
    # タブ1: 表示モード (Matplotlib)
    # ---------------------------------------------------------
    with tab_display:
        st.caption("Matplotlibを使用した静的な高解像度プロットを作成します。論文やレポート用の画像出力に適しています。")
        
        # 表示モード専用のサイドバー設定（のようなものをExpanderで配置）
        with st.expander("グラフのスタイル設定", expanded=False):
            c1, c2, c3 = st.columns(3)
            x_label = c1.text_input("X軸ラベル", "Wavelength (nm)")
            y_label = c2.text_input("Y軸ラベル", "Norm. Abs." if do_normalize else "Abs.")
            legend_loc = c3.radio("凡例位置", ('Outside', 'Inside'))
            
            c1, c2 = st.columns(2)
            show_grid = c1.checkbox("グリッド表示", value=True)
            use_manual_range = c2.checkbox("軸範囲を手動設定", value=False)
            
            x_min, x_max, y_min, y_max = None, None, None, None
            if use_manual_range:
                cc1, cc2, cc3, cc4 = st.columns(4)
                x_min = cc1.number_input("X Min", value=200.0)
                x_max = cc2.number_input("X Max", value=800.0)
                y_min = cc3.number_input("Y Min", value=-0.1)
                y_max = cc4.number_input("Y Max", value=1.2)

            # カラーマップ設定
            use_custom_style = st.checkbox("個別スタイルを適用", value=False)
            cmap_name = 'viridis'
            if not use_custom_style:
                cmap_name = st.selectbox("カラーマップ", ['viridis', 'jet', 'coolwarm', 'rainbow', 'Manual'], index=0)
            else:
                st.info("サイドバーで個別スタイルを設定できません。コード内のスタイル辞書が適用されます。")
                # ここでは簡易化のため、Matplotlibモードでの詳細な個別設定UIは省略し、
                # session_state['styles'] を参照する形にします

        if display_data_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # カラー設定
            num_files = len(display_data_list)
            if not use_custom_style:
                if cmap_name == 'Manual':
                    colors_list = DEFAULT_COLORS * (num_files//len(DEFAULT_COLORS) + 1)
                else:
                    cmap = plt.get_cmap(cmap_name)
                    colors_list = [cmap(i) for i in np.linspace(0, 1, num_files)]
            
            for i, item in enumerate(display_data_list):
                label = item['label']
                # スタイル決定
                if use_custom_style and label in st.session_state['styles']:
                    s = st.session_state['styles'][label]
                    color, lw, ls = s['color'], s['linewidth'], LINE_STYLES.get(s['linestyle'], '-')
                else:
                    color = colors_list[i] if not use_custom_style else 'black'
                    lw, ls = 1.5, '-'

                ax.plot(item['x'], item['y'], label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.8)

            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            if show_grid: ax.grid(True, linestyle=':', alpha=0.6)
            if use_manual_range: ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            
            if legend_loc == 'Outside': ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else: ax.legend(loc='best')

            st.pyplot(fig)

            # ダウンロードボタン
            st.markdown("#### 📥 画像保存")
            col1, col2, col3 = st.columns(3)
            img_png = io.BytesIO()
            plt.savefig(img_png, format='png', bbox_inches='tight', dpi=300)
            col1.download_button("PNGで保存", img_png, "plot.png")
            
            img_tiff = io.BytesIO()
            plt.savefig(img_tiff, format='tiff', bbox_inches='tight', dpi=300)
            col2.download_button("TIFFで保存", img_tiff, "plot.tiff")

            gnu_data = create_gnuplot_data(display_data_list)
            if gnu_data: col3.download_button("DAT(Gnuplot用)保存", gnu_data, "data.dat")
        else:
            st.info("データを選択してください。")

    # ---------------------------------------------------------
    # タブ2: 解析モード (Plotly)
    # ---------------------------------------------------------
    with tab_analysis:
        st.caption("Plotlyを使用したインタラクティブな解析です。マウスカーソルを合わせると値を読み取れます。")
        
        if not display_data_list:
            st.warning("データが選択されていません。サイドバーでデータを選択してください。")
        else:
            # --- 解析用コントロールパネル ---
            st.subheader("🛠 解析設定")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**1. 面積計算 (積分)**")
                do_calc_area = st.checkbox("面積を計算・表示する", value=True)
                c1, c2 = st.columns(2)
                calc_start = c1.number_input("開始波長 (nm)", value=300.0, step=10.0)
                calc_end = c2.number_input("終了波長 (nm)", value=500.0, step=10.0)
                if calc_start > calc_end: calc_start, calc_end = calc_end, calc_start
            
            with col_b:
                st.markdown("**2. ピーク検出**")
                do_peak_search = st.checkbox("ピークを検出する", value=True)
                c1, c2 = st.columns(2)
                peak_prominence = c1.number_input("感度 (Prominence)", value=0.01, format="%.4f", step=0.005)
                peak_distance = c2.number_input("最小距離 (Points)", value=10, step=1)

            st.markdown("---")

            # --- Plotly 描画 ---
            fig_p = go.Figure()
            
            peak_results_all = []
            area_results_all = []

            # 色生成
            colors = DEFAULT_COLORS * (len(display_data_list)//len(DEFAULT_COLORS) + 1)

            for i, item in enumerate(display_data_list):
                color = colors[i]
                label = item['label']
                
                # 1. メインのスペクトル描画
                fig_p.add_trace(go.Scatter(
                    x=item['x'], y=item['y'],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{label}</b><br>Wave: %{{x:.2f}} nm<br>Int: %{{y:.4f}}<extra></extra>"
                ))

                # 2. 面積計算と塗りつぶし
                if do_calc_area:
                    mask = (item['x'] >= calc_start) & (item['x'] <= calc_end)
                    x_sub = item['x'][mask]
                    y_sub = item['y'][mask]
                    
                    if len(x_sub) > 1:
                        # 面積計算 (台形公式)
                        area = np.trapezoid(y_sub, x_sub) if hasattr(np, 'trapezoid') else np.trapz(y_sub, x_sub)
                        area_results_all.append({'ファイル名': label, '面積': area})
                        
                        # 塗りつぶし用トレース（閉じたポリゴンを作る）
                        # x, yの配列の両端に y=0 の点を追加して閉じる
                        x_fill = np.concatenate(([x_sub[0]], x_sub, [x_sub[-1]]))
                        y_fill = np.concatenate(([0], y_sub, [0]))
                        
                        fig_p.add_trace(go.Scatter(
                            x=x_fill, y=y_fill,
                            fill='toself',
                            mode='none', # 線は描かない
                            fillcolor=color,
                            opacity=0.2,
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                # 3. ピーク検出とマーカー表示
                if do_peak_search:
                    peaks, _ = find_peaks(item['y'], prominence=peak_prominence, distance=peak_distance)
                    if len(peaks) > 0:
                        peak_x = item['x'][peaks]
                        peak_y = item['y'][peaks]
                        
                        # テーブル用データ保存
                        for px, py in zip(peak_x, peak_y):
                            peak_results_all.append({'ファイル名': label, '波長 (nm)': px, '強度': py})

                        # マーカー描画
                        fig_p.add_trace(go.Scatter(
                            x=peak_x, y=peak_y,
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=10, color=color, line=dict(color='black', width=1)),
                            name=f"{label} Peaks",
                            showlegend=False, # 凡例がうるさくなるので隠す
                            hovertemplate=f"<b>{label} Peak</b><br>Wave: %{{x:.2f}} nm<br>Int: %{{y:.4f}}<extra></extra>"
                        ))

            # レイアウト調整
            fig_p.update_layout(
                title="Interactive Spectra Analysis",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity",
                template="plotly_white",
                height=600,
                hovermode="x unified", # x軸を揃えてホバー表示
                legend=dict(x=1.01, y=1)
            )

            # 積分範囲の縦線を表示
            if do_calc_area:
                fig_p.add_vline(x=calc_start, line_width=1, line_dash="dash", line_color="gray")
                fig_p.add_vline(x=calc_end, line_width=1, line_dash="dash", line_color="gray")

            st.plotly_chart(fig_p, use_container_width=True)

            # --- 解析結果テーブルの表示 ---
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.subheader("📊 面積計算結果")
                if area_results_all:
                    st.dataframe(pd.DataFrame(area_results_all), use_container_width=True)
                else:
                    st.info("計算対象のデータがありません")
            
            with col_res2:
                st.subheader("🏔 ピーク検出結果")
                if peak_results_all:
                    df_peaks = pd.DataFrame(peak_results_all).sort_values(['ファイル名', '波長 (nm)'])
                    st.dataframe(df_peaks, use_container_width=True)
                else:
                    st.info("ピークが見つかりませんでした。感度設定を調整してください。")

if __name__ == "__main__":
    main()