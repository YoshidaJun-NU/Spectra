import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import savgol_filter

# ---------------------------------------------------------
# 関数定義: CD用ダミーデータの生成 (キャッシュ有効化)
# ---------------------------------------------------------
@st.cache_data
def generate_cd_dummy_data():
    """動作確認用にCDスペクトル（正負あり）を作成する"""
    data_list = []
    x = np.linspace(200, 350, 301)
    
    # パターン1: Type A (Alpha-helix like)
    y1 = 20 * np.exp(-((x - 192)**2) / (2 * 10**2)) - 10 * np.exp(-((x - 222)**2) / (2 * 10**2)) - 10 * np.exp(-((x - 208)**2) / (2 * 10**2))
    y1 = y1 * 2
    y1 += np.random.normal(0, 0.5, len(x))
    
    # パターン2: Type B (Beta-sheet like)
    y2 = 10 * np.exp(-((x - 195)**2) / (2 * 8**2)) - 8 * np.exp(-((x - 218)**2) / (2 * 12**2))
    y2 += np.random.normal(0, 0.5, len(x))

    # パターン3: Type C (Random Coil / Buffer)
    y3 = -15 * np.exp(-((x - 198)**2) / (2 * 15**2)) + 2 * np.sin(x/10)
    y3 += np.random.normal(0, 0.3, len(x))

    data_list.append({'label': 'Alpha_Helix', 'x': x, 'y': y1})
    data_list.append({'label': 'Beta_Sheet', 'x': x, 'y': y2})
    data_list.append({'label': 'Random_Coil', 'x': x, 'y': y3})
    
    return data_list

# ---------------------------------------------------------
# 関数定義: ファイル読み込み (キャッシュ有効化)
# ---------------------------------------------------------
@st.cache_data
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            
            sep_char = '\t' if separator == 'tab' else ','
            header_setting = 0 if has_header else None
            
            # エンコーディング対応が必要な場合は encoding='shift_jis' などを検討
            df = pd.read_csv(uploaded_file, sep=sep_char, skiprows=skip_rows, header=header_setting)
            
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
            if df.shape[1] < 2:
                continue

            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]

            label = uploaded_file.name.rsplit('.', 1)[0]
            data_list.append({'label': label, 'x': x, 'y': y})
        except Exception as e:
            pass
    return data_list

# ---------------------------------------------------------
# 関数定義: データ処理 (平滑化・オフセット)
# ---------------------------------------------------------
def process_data(data_list, smooth_window, use_offset, offset_wl):
    processed = []
    for item in data_list:
        x, y = item['x'], item['y']
        
        # 1. 平滑化 (Savitzky-Golay)
        if smooth_window > 1:
            try:
                wl = int(smooth_window)
                if wl % 2 == 0: wl += 1
                if len(x) > wl:
                    y = savgol_filter(y, window_length=wl, polyorder=3)
            except:
                pass

        # 2. オフセット補正
        if use_offset:
            idx = (np.abs(x - offset_wl)).argmin()
            y = y - y[idx]

        processed.append({'label': item['label'], 'x': x, 'y': y})
    return processed

# ---------------------------------------------------------
# 関数定義: Gnuplotデータ作成
# ---------------------------------------------------------
def create_gnuplot_data(data_list):
    if not data_list: return None
    all_x = np.concatenate([d['x'] for d in data_list])
    unique_x = np.unique(all_x)
    unique_x.sort()
    
    df_merged = pd.DataFrame({'Wavelength': unique_x})
    
    for item in data_list:
        y_interp = np.interp(unique_x, item['x'], item['y'], left=np.nan, right=np.nan)
        df_merged[item['label']] = y_interp
        
    return df_merged.to_csv(sep='\t', index=False, float_format='%.4f')

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="CD Spectra Plotter Pro", layout="wide", page_icon="🧬")
    st.title("🧬 CD Spectra Plotter Pro")

    if 'raw_data_list' not in st.session_state:
        st.session_state['raw_data_list'] = []

    # --- Sidebar: データ入力 ---
    with st.sidebar:
        st.header("1. データ読み込み")
        col_load1, col_load2 = st.columns(2)
        if col_load1.button("サンプルデータ"):
            st.session_state['raw_data_list'] = generate_cd_dummy_data()
            st.success("サンプル読込完了")
        
        if col_load2.button("クリア"):
            st.session_state['raw_data_list'] = []
            st.rerun()

        st.markdown("---")
        with st.expander("CSV/TXT 読み込み設定", expanded=True):
            # ▼ 修正箇所: デフォルトをタブにし、表示を見やすくしました
            separator = st.radio(
                "区切り文字", 
                ('tab', 'comma'), 
                index=0, 
                horizontal=True,
                format_func=lambda x: "Tab (TXT)" if x == 'tab' else "Comma (CSV)"
            )
            
            skip_rows = st.number_input("ヘッダー前のスキップ行", value=19, min_value=0)
            has_header = st.checkbox("ヘッダー行あり", value=True)
            
            uploaded_files = st.file_uploader("ファイルをアップロード", accept_multiple_files=True)
            if uploaded_files:
                loaded = load_data(uploaded_files, separator, skip_rows, has_header)
                if loaded:
                    st.session_state['raw_data_list'] = loaded

    # --- Sidebar: データ処理 ---
    with st.sidebar:
        st.header("2. データ処理")
        
        smooth_window = st.slider(
            "平滑化 (Window Size)", 
            min_value=1, max_value=21, value=1, step=2,
            help="Savitzky-Golayフィルタによるノイズ除去"
        )

        use_offset = st.checkbox("ゼロ点補正 (Offset Correction)")
        offset_wl = 350.0
        if use_offset:
            offset_wl = st.number_input("ゼロ点とする波長 (nm)", value=350.0, step=1.0)

    if st.session_state['raw_data_list']:
        plot_data = process_data(st.session_state['raw_data_list'], smooth_window, use_offset, offset_wl)
    else:
        plot_data = []

    # --- Sidebar: グラフ設定 ---
    with st.sidebar:
        st.header("3. グラフスタイル")
        
        x_label = st.text_input("X軸ラベル", "Wavelength (nm)")
        y_label = st.text_input("Y軸ラベル", "Ellipticity (mdeg)")
        
        with st.expander("軸範囲の手動設定"):
            use_manual_range = st.checkbox("有効にする")
            c1, c2 = st.columns(2)
            x_min = c1.number_input("X Min", value=190.0)
            x_max = c2.number_input("X Max", value=260.0)
            y_min = c1.number_input("Y Min", value=-50.0)
            y_max = c2.number_input("Y Max", value=50.0)

        style_mode = st.selectbox("配色テーマ", ["Auto (Distinct)", "CoolWarm", "Manual"])
        legend_loc = st.radio("凡例位置", ('Best', 'Outside'), horizontal=True)

        plot_settings = []
        if plot_data:
            if style_mode == "Manual":
                st.markdown("##### 個別ライン設定")
                line_style_dict = {'Solid': '-', 'Dash': '--', 'Dot': ':', 'DashDot': '-.'}
                default_cols = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
                
                for i, item in enumerate(plot_data):
                    with st.expander(f"{item['label']}", expanded=False):
                        c1, c2 = st.columns(2)
                        col = c1.color_picker("色", default_cols[i % len(default_cols)], key=f"c_{i}")
                        lw = c2.number_input("太さ", 1.0, 5.0, 2.0, 0.5, key=f"w_{i}")
                        ls_key = st.selectbox("線種", list(line_style_dict.keys()), key=f"s_{i}")
                        plot_settings.append({'color': col, 'ls': line_style_dict[ls_key], 'lw': lw})
            else:
                for i in range(len(plot_data)):
                    if style_mode == "Auto (Distinct)":
                        c = plt.cm.tab10(i % 10)
                    else:
                        c = plt.cm.coolwarm(i / max(len(plot_data)-1, 1))
                    plot_settings.append({'color': c, 'ls': '-', 'lw': 2.0})

    # --- Main: プロット描画 ---
    if plot_data:
        col_main, col_dl = st.columns([4, 1])
        with col_main:
            st.subheader("CD Spectra Analysis")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)

        for i, item in enumerate(plot_data):
            sett = plot_settings[i]
            ax.plot(item['x'], item['y'], label=item['label'], 
                    color=sett['color'], linestyle=sett['ls'], linewidth=sett['lw'], alpha=0.9)

        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.tick_params(direction='in', top=True, right=True)
        
        if use_manual_range:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.autoscale(enable=True, axis='both', tight=True)
            ylim = ax.get_ylim()
            y_range = ylim[1] - ylim[0]
            ax.set_ylim(ylim[0] - y_range*0.05, ylim[1] + y_range*0.05)

        if legend_loc == 'Outside':
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        else:
            ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### 📥 Export")
        c1, c2, c3 = st.columns(3)
        
        img_png = io.BytesIO()
        plt.savefig(img_png, format='png', dpi=300, bbox_inches='tight')
        c1.download_button("High-Res PNG (300dpi)", img_png.getvalue(), "cd_spectra.png", "image/png")
        
        img_tiff = io.BytesIO()
        plt.savefig(img_tiff, format='tiff', dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        c2.download_button("Publication TIFF", img_tiff.getvalue(), "cd_spectra.tiff", "image/tiff")
        
        gnu_data = create_gnuplot_data(plot_data)
        if gnu_data:
            c3.download_button("Processed Data (.txt)", gnu_data, "processed_cd_data.txt", "text/plain")

        plt.close(fig)

    else:
        st.info("👈 左サイドバーから **[サンプルデータ]** を読み込むか、ファイルをアップロードしてください。")
        st.markdown("""
        ### Supported Formats
        * **CSV / TXT**: 1列目が波長(nm)、2列目がCD値(mdeg)
        * **J-815 / J-1500**: JASCOのテキスト出力に対応（デフォルトはTab区切り）
        """)

if __name__ == "__main__":
    main()