import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. 解析・計算用関数
# ---------------------------------------------------------
def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        amp, cen, sigma = params[i], params[i+1], params[i+2]
        y += amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
    y += params[-1] # offset
    return y

def trans_to_abs(y_trans):
    y_clamped = np.clip(y_trans, 1e-5, 100.0)
    return 2.0 - np.log10(y_clamped)

# ---------------------------------------------------------
# 2. データ読み込み
# ---------------------------------------------------------
def load_data(uploaded_files):
    data_list = []
    for f in uploaded_files:
        try:
            content = f.getvalue()
            for enc in ['utf-8', 'cp932', 'shift_jis', 'latin1']:
                try: text = content.decode(enc); break
                except: continue
            
            lines = text.splitlines()
            x_unit, y_unit = "Wavenumber (cm⁻¹)", "Transmittance (%)"
            use_skip = 0
            
            for i, line in enumerate(lines):
                if 'XUNITS' in line:
                    val = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                    if val: x_unit = val
                if 'YUNITS' in line:
                    val = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                    if val: y_unit = val
                if 'XYDATA' in line:
                    use_skip = i + 1
                    break
            
            sep = ',' if f.name.lower().endswith('.csv') else None
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=use_skip, header=None, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] >= 2:
                data_list.append({
                    'label': f.name.rsplit('.', 1)[0],
                    'x': df.iloc[:, 0].values,
                    'y': df.iloc[:, 1].values,
                    'x_unit': x_unit,
                    'y_unit': y_unit
                })
        except Exception as e:
            st.error(f"{f.name} の読み込み失敗: {e}")
    return data_list

# ---------------------------------------------------------
# 3. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="IR Spectra Pro", layout="wide")
    st.title("IR Spectra Analyzer 🧪")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：1. ロード ---
    st.sidebar.header("1. データ読み込み")
    files = st.sidebar.file_uploader("JASCO CSV/TXTをアップロード", accept_multiple_files=True)
    if files:
        if st.sidebar.button("データを最新化（リセット）"):
            st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("👈 左側のサイドバーからファイルをアップロードしてください。")
        return

    # --- サイドバー：2. 表示・スタイル設定 ---
    st.sidebar.header("2. グラフ・スタイル設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    selected = st.sidebar.multiselect("表示ファイル", all_labels, default=all_labels)
    y_mode = st.sidebar.radio("縦軸モード", ["Transmittance (%)", "Absorbance"], index=0)

    # 軸範囲設定
    with st.sidebar.expander("軸の範囲設定"):
        col_x1, col_x2 = columns = st.columns(2)
        x_max_def = col_x1.number_input("横軸 開始", value=4000.0)
        x_min_def = col_x2.number_input("横軸 終了", value=400.0)
        
        y_min_val, y_max_val = (0.0, 2.0) if y_mode == "Absorbance" else (0.0, 105.0)
        col_y1, col_y2 = st.columns(2)
        y_min_input = col_y1.number_input("縦軸 最小", value=float(y_min_val))
        y_max_input = col_y2.number_input("縦軸 最大", value=float(y_max_val))

    # 個別スタイル設定
    plot_configs = {}
    if selected:
        st.sidebar.subheader("個別プロット設定")
        for label in selected:
            with st.sidebar.expander(f"設定: {label}"):
                c1, c2 = st.columns(2)
                color = c1.color_picker("色", key=f"clr_{label}")
                ls = c2.selectbox("線種", ["-", "--", "-.", ":"], key=f"ls_{label}")
                lw = c1.slider("太さ", 0.5, 5.0, 1.5, key=f"lw_{label}")
                offset = c2.number_input("Yオフセット", value=0.0, step=0.1, key=f"off_{label}")
                plot_configs[label] = {"color": color, "ls": ls, "lw": lw, "offset": offset}

    # --- サイドバー：3. 解析機能 ---
    st.sidebar.header("3. 解析")
    do_fit = st.sidebar.checkbox("マルチガウスフィッティング")
    num_peaks = st.sidebar.number_input("ピーク数", 1, 10, 1)
    fit_target = st.sidebar.selectbox("解析対象ファイル", selected) if selected else None

    # --- グラフ描画 ---
    if selected:
        fig, ax = plt.subplots(figsize=(10, 6))
        display_data = [d for d in st.session_state['data_list'] if d['label'] in selected]

        for item in display_data:
            label = item['label']
            x, y = item['x'], item['y'].copy()
            conf = plot_configs[label]

            # 縦軸変換
            if y_mode == "Absorbance":
                if np.max(y) > 10: 
                    y = trans_to_abs(y)
            
            # オフセット適用
            y_plotted = y + conf['offset']

            # プロット
            ax.plot(x, y_plotted, label=label, 
                    color=conf['color'], linestyle=conf['ls'], linewidth=conf['lw'])

            # フィッティング (オフセットなしの元のyに対して計算し、描画時にオフセットを加える)
            if do_fit and label == fit_target:
                mask = (x >= min(x_min_def, x_max_def)) & (x <= max(x_min_def, x_max_def))
                xf, yf = x[mask], y[mask]
                try:
                    p0 = []
                    found, _ = find_peaks(yf if y_mode=="Absorbance" else -yf, prominence=0.01)
                    idx_peaks = found[:num_peaks] if len(found) >= num_peaks else np.linspace(0, len(xf)-1, num_peaks, dtype=int)
                    for idx in idx_peaks:
                        p0.extend([yf[idx], xf[idx], 5.0])
                    p0.append(np.mean(yf))
                    popt, _ = curve_fit(multi_gaussian, xf, yf, p0=p0)
                    # フィッティング曲線にもオフセットを適用
                    ax.plot(xf, multi_gaussian(xf, *popt) + conf['offset'], 
                            color=conf['color'], linestyle="--", alpha=0.7)
                except:
                    st.sidebar.warning(f"Fitting failed for {label}")

        ax.set_xlabel(display_data[0].get('x_unit', "Wavenumber (cm⁻¹)"))
        ax.set_ylabel(y_mode)
        ax.set_xlim(x_max_def, x_min_def)
        ax.set_ylim(y_min_input, y_max_input)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        st.pyplot(fig)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        st.download_button("グラフ保存 (PNG)", buf.getvalue(), "ir_analysis.png")

if __name__ == "__main__":
    main()