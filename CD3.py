import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors  # これを追加
import numpy as np
import io
from scipy.signal import savgol_filter

# GUIなし環境（Ubuntuサーバー等）での動作を安定させる
matplotlib.use('Agg')

# ---------------------------------------------------------
# 1. 関数定義: データ処理・生成
# ---------------------------------------------------------
@st.cache_data
def generate_cd_dummy_data():
    x = np.linspace(190, 260, 150)
    y1 = 30 * np.exp(-((x - 192)**2) / 50) - 15 * np.exp(-((x - 222)**2) / 100) - 10 * np.exp(-((x - 208)**2) / 100)
    y2 = 10 * np.exp(-((x - 195)**2) / 80) - 12 * np.exp(-((x - 218)**2) / 200)
    return [
        {'label': 'Protein_A', 'x': x, 'y': y1 + np.random.normal(0, 0.2, len(x))},
        {'label': 'Protein_B', 'x': x, 'y': y2 + np.random.normal(0, 0.2, len(x))}
    ]

@st.cache_data
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            sep_char = '\t' if separator == 'tab' else ','
            df = pd.read_csv(uploaded_file, sep=sep_char, skiprows=skip_rows, header=0 if has_header else None)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if df.shape[1] >= 2:
                x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
                idx = np.argsort(x)
                data_list.append({'label': uploaded_file.name.split('.')[0], 'x': x[idx], 'y': y[idx]})
        except Exception: continue
    return data_list

def apply_processing(data_list, smooth, use_offset, offset_wl, convert_to_de, params_dict):
    processed = []
    for item in data_list:
        x, y = item['x'].copy(), item['y'].copy()
        if smooth > 1:
            y = savgol_filter(y, window_length=smooth if smooth%2!=0 else smooth+1, polyorder=3)
        if use_offset:
            y -= y[np.abs(x - offset_wl).argmin()]
        if convert_to_de:
            p = params_dict.get(item['label'], {'c': 1e-5, 'l': 0.1})
            y /= (32980 * p['c'] * p['l'])
        processed.append({'label': item['label'], 'x': x, 'y': y})
    return processed

# ---------------------------------------------------------
# 2. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Advanced CD Plotter", layout="wide")
    st.title("🧬 CD Spectra Plotter")

    if 'raw_data' not in st.session_state: st.session_state['raw_data'] = []

    # --- サイドバー 1: データ管理 ---
    with st.sidebar:
        st.header("1. Data Management")
        c1, c2 = st.columns(2)
        if c1.button("Sample Data"): st.session_state['raw_data'] = generate_cd_dummy_data()
        if c2.button("Clear"): st.session_state['raw_data'] = []; st.rerun()
        
        files = st.file_uploader("Upload CSV/TXT", accept_multiple_files=True)
        if files:
            with st.expander("Import Settings"):
                sep = st.radio("Separator", ("tab", "comma"))
                skip = st.number_input("Skip Rows", 0, 100, 19)
                head = st.checkbox("Header exists", True)
            st.session_state['raw_data'] = load_data(files, sep, skip, head)

    if not st.session_state['raw_data']:
        st.info("👈 Please load data to start.")
        return

    # --- サイドバー 2: 選択と単位変換 ---
    all_labels = [d['label'] for d in st.session_state['raw_data']]
    selected = st.sidebar.multiselect("Select Series", all_labels, default=all_labels)
    target_data = [d for d in st.session_state['raw_data'] if d['label'] in selected]

    convert_de = st.sidebar.checkbox("Convert to Δε (M⁻¹cm⁻¹)")
    unit_params = {}
    if convert_de:
        st.sidebar.caption("Define Conc(M) and Path(cm):")
        for d in target_data:
            with st.sidebar.expander(f"Params: {d['label']}"):
                c = st.number_input("Conc (M)", value=1.0e-5, format="%.2e", key=f"c_{d['label']}")
                l = st.number_input("Path (cm)", value=0.1, key=f"l_{d['label']}")
                unit_params[d['label']] = {'c': c, 'l': l}

    # --- サイドバー 3: プロット設定 (ここがメインの追加機能) ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. Plot Customization")
    
    with st.sidebar.expander("Global Axis Style", expanded=True):
        tick_dir = st.radio("Tick Direction", ["in", "out", "inout"], index=0, horizontal=True)
        show_top_right = st.checkbox("Show Top/Right Spines (Box)", value=True)
        grid_on = st.checkbox("Show Grid", value=False)
        x_lab = st.text_input("X Label", "Wavelength (nm)")
        y_lab = st.text_input("Y Label", r"$\Delta\epsilon$" if convert_de else "Ellipticity (mdeg)")

    # 系列ごとの詳細設定
    line_configs = {}
    st.sidebar.subheader("Series Style")
    for i, d in enumerate(target_data):
        with st.sidebar.expander(f"Style: {d['label']}"):
            # mcolors.to_hex() でヘックス形式に変換する
            default_color = mcolors.to_hex(plt.cm.tab10(i % 10))
            col = st.color_picker("Color", default_color, key=f"col_{d['label']}")
            width = st.slider("Line Width", 0.5, 5.0, 2.0, 0.5, key=f"width_{d['label']}")
            style = st.selectbox("Line Style", ["-", "--", ":", "-."], key=f"style_{d['label']}")
            line_configs[d['label']] = {'color': col, 'lw': width, 'ls': style}

    # --- 描画実行 ---
    smooth = st.sidebar.slider("Smoothing", 1, 31, 1, 2)
    processed_data = apply_processing(target_data, smooth, False, 350, convert_de, unit_params)

    # Matplotlib 描画
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # 基本設定の適用
    ax.tick_params(direction=tick_dir, top=show_top_right, right=show_top_right, labelsize=11)
    if not show_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if grid_on:
        ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    
    # プロットごとにループして設定を反映
    for d in processed_data:
        cfg = line_configs[d['label']]
        ax.plot(d['x'], d['y'], label=d['label'], 
                color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['ls'])

    ax.set_xlabel(x_lab, fontsize=13)
    ax.set_ylabel(y_lab, fontsize=13)
    ax.legend(frameon=False)
    
    # メイン画面表示
    st.pyplot(fig)

    # --- エクスポート ---
    st.markdown("### 📥 Export")
    c1, c2, c3 = st.columns(3)
    
    # PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    c1.download_button("Download PNG (300dpi)", buf.getvalue(), "plot.png", "image/png")
    
    # TIFF (論文用)
    tif_buf = io.BytesIO()
    fig.savefig(tif_buf, format="tiff", dpi=300, bbox_inches='tight')
    c2.download_button("Download TIFF", tif_buf.getvalue(), "plot.tiff", "image/tiff")
    
    # CSV (処理済みデータ)
    csv_data = pd.DataFrame({d['label']: pd.Series(d['y'], index=d['x']) for d in processed_data})
    c3.download_button("Download CSV", csv_data.to_csv(), "processed_data.csv", "text/csv")

    plt.close(fig)

if __name__ == "__main__":
    main()