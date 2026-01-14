import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 定数とガウス関数定義
# ---------------------------------------------------------
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
LINE_STYLES = {'Solid (実線)': '-', 'Dashed (破線)': '--', 'Dash-dot (一点鎖線)': '-.', 'Dotted (点線)': ':'}

def gaussian(x, amp, cen, sigma, offset):
    """ガウス関数モデル: A * exp(-(x-mu)^2 / (2*sigma^2)) + offset"""
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2)) + offset

# ---------------------------------------------------------
# データ読み込み・スタイル初期化 (前回機能を継承)
# ---------------------------------------------------------
def init_styles(data_list):
    if 'styles' not in st.session_state:
        st.session_state['styles'] = {}
    for i, item in enumerate(data_list):
        label = item['label']
        if label not in st.session_state['styles']:
            st.session_state['styles'][label] = {
                'color': DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                'linewidth': 1.5, 'linestyle': 'Solid (実線)'
            }

def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    for uploaded_file in uploaded_files:
        try:
            content_bytes = uploaded_file.getvalue()
            decoded_text = ""
            for enc in ['utf-8', 'cp932', 'shift_jis', 'latin1']:
                try:
                    decoded_text = content_bytes.decode(enc)
                    break
                except UnicodeDecodeError: continue

            use_sep = ',' if separator == 'comma' else '\t'
            use_skip, use_header = skip_rows, (0 if has_header else None)
            
            if 'XYDATA' in decoded_text:
                lines = decoded_text.splitlines()
                for i, line in enumerate(lines):
                    if 'XYDATA' in line:
                        use_skip, use_header = i + 1, None
                        use_sep = ',' if uploaded_file.name.lower().endswith('.csv') else None
                        break
            
            df = pd.read_csv(io.StringIO(decoded_text), sep=use_sep, skiprows=use_skip, header=use_header, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if df.shape[1] >= 2:
                data_list.append({'label': uploaded_file.name.rsplit('.', 1)[0], 'x': df.iloc[:, 0].values, 'y': df.iloc[:, 1].values})
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
    return data_list

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Advanced Spectra Analyzer", layout="wide")
    st.title("Spectra Analyzer 🧪")

    if 'data_list' not in st.session_state: st.session_state['data_list'] = []

    # --- サイドバー：1. ロード ---
    st.sidebar.header("1. データ読み込み")
    uploaded_files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True, type=['txt', 'csv', 'dat', 'spz'])
    
    if uploaded_files:
        st.session_state['data_list'] = load_data(uploaded_files, 'tab', 19, True)
        init_styles(st.session_state['data_list'])

    # --- サイドバー：2. 表示選択 ---
    st.sidebar.header("2. 表示データの選択")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    selected_labels = st.sidebar.multiselect("表示するファイル", options=all_labels, default=all_labels)

    # --- サイドバー：3. 前処理とデザイン ---
    st.sidebar.header("3. グラフ・前処理設定")
    
    # ベースライン補正
    st.sidebar.subheader("ベースライン補正")
    bl_mode = st.sidebar.selectbox("補正モード", ["None", "Constant (ゼロ点補正)", "Linear (2点指定直線)"])
    bl_params = {}
    if bl_mode == "Constant (ゼロ点補正)":
        bl_params['wave'] = st.sidebar.number_input("補正基準波長 (nm)", value=700.0)
    elif bl_mode == "Linear (2点指定直線)":
        c1, c2 = st.sidebar.columns(2)
        bl_params['p1'] = c1.number_input("波長1 (nm)", value=650.0)
        bl_params['p2'] = c2.number_input("波長2 (nm)", value=750.0)

    # 軸ラベル設定 (デフォルトをIntensityに変更)
    y_label_text = st.sidebar.text_input("Y軸ラベル", "Intensity (a.u.)")
    show_grid = st.sidebar.checkbox("グリッド表示", value=True)
    
    # 個別スタイル設定
    if st.sidebar.checkbox("個別スタイル設定"):
        for label in selected_labels:
            with st.sidebar.expander(f"🎨 {label}"):
                style = st.session_state['styles'][label]
                style['color'] = st.color_picker("色", style['color'], key=f"c_{label}")
                style['linewidth'] = st.number_input("太さ", 0.5, 5.0, style['linewidth'], key=f"w_{label}")
                style['linestyle'] = st.selectbox("線種", list(LINE_STYLES.keys()), index=0, key=f"s_{label}")

    # --- サイドバー：4. 高度な解析 ---
    st.sidebar.header("4. 高度な解析")
    
    # ガウスフィッティング設定
    do_fit = st.sidebar.checkbox("ガウス関数フィッティング")
    fit_params = {}
    if do_fit:
        st.sidebar.caption("フィッティングする範囲を選択してください")
        f_col1, f_col2 = st.sidebar.columns(2)
        fit_params['start'] = f_col1.number_input("開始(nm)", value=350.0)
        fit_params['end'] = f_col2.number_input("終了(nm)", value=450.0)
        fit_params['target'] = st.sidebar.selectbox("対象ファイル", selected_labels)

    # ピーク検出
    do_peak = st.sidebar.checkbox("ピーク検出")
    
    # --- データ処理とプロット ---
    target_data = [d for d in st.session_state['data_list'] if d['label'] in selected_labels]

    if target_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for item in target_data:
            x, y = item['x'], item['y'].copy()
            
            # --- ベースライン補正実行 ---
            
            if bl_mode == "Constant (ゼロ点補正)":
                idx = np.abs(x - bl_params['wave']).argmin()
                y -= y[idx]
            elif bl_mode == "Linear (2点指定直線)":
                idx1, idx2 = np.abs(x - bl_params['p1']).argmin(), np.abs(x - bl_params['p2']).argmin()
                slope = (y[idx2] - y[idx1]) / (x[idx2] - x[idx1])
                intercept = y[idx1] - slope * x[idx1]
                y -= (slope * x + intercept)

            style = st.session_state['styles'][item['label']]
            ax.plot(x, y, label=item['label'], color=style['color'], 
                    linewidth=style['linewidth'], linestyle=LINE_STYLES[style['linestyle']], alpha=0.8)

            # --- ガウスフィッティング実行 ---
            if do_fit and item['label'] == fit_params['target']:
                mask = (x >= fit_params['start']) & (x <= fit_params['end'])
                xf, yf = x[mask], y[mask]
                if len(xf) > 5:
                    try:
                        # 初期値推定: [強度, 中心, 幅, オフセット]
                        p0 = [np.max(yf) - np.min(yf), xf[np.argmax(yf)], (xf[-1]-xf[0])/6, np.min(yf)]
                        popt, _ = curve_fit(gaussian, xf, yf, p0=p0)
                        
                        # フィッティング曲線の描画
                        x_fine = np.linspace(xf.min(), xf.max(), 100)
                        ax.plot(x_fine, gaussian(x_fine, *popt), 'r--', lw=2, label="Gaussian Fit")
                        
                        # 結果の計算
                        fwhm = 2.355 * abs(popt[2])
                        fit_results = {
                            "Amplitude": popt[0], "Center (nm)": popt[1],
                            "FWHM (nm)": fwhm, "Offset": popt[3]
                        }
                        
                    except Exception as e:
                        st.warning(f"Fitting failed: {e}")

            # --- ピーク検出実行 ---
            if do_peak:
                peaks, _ = find_peaks(y, prominence=0.01)
                ax.plot(x[peaks], y[peaks], "v", color=style['color'])

        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel(y_label_text)
        if show_grid: ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        # --- 解析結果の表示 ---
        if do_fit and 'fit_results' in locals():
            st.subheader("📊 Gaussian Fitting Results")
            cols = st.columns(4)
            for i, (k, v) in enumerate(fit_results.items()):
                cols[i].metric(k, f"{v:.4f}")

    else:
        st.info("👈 ファイルをアップロードして選択してください。")

    # --- ダウンロードセクション ---
    if target_data:
        st.markdown("---")
        c1, c2 = st.columns(2)
        img_png = io.BytesIO()
        plt.savefig(img_png, format='png', bbox_inches='tight', dpi=300)
        c1.download_button("グラフを画像(PNG)で保存", img_png, "spectra_plot.png", "image/png")

if __name__ == "__main__":
    main()