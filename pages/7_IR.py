import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# モデル関数定義
# ---------------------------------------------------------
def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        amp, cen, sigma = params[i], params[i+1], params[i+2]
        y += amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
    y += params[-1] # offset
    return y

# ---------------------------------------------------------
# データ読み込み (JASCOヘッダー解析機能付き)
# ---------------------------------------------------------
def load_data(uploaded_files):
    data_list = []
    for f in uploaded_files:
        try:
            content = f.getvalue()
            for enc in ['utf-8', 'cp932', 'shift_jis']:
                try: text = content.decode(enc); break
                except: continue
            
            lines = text.splitlines()
            x_unit, y_unit = "Wavelength (nm)", "Intensity"
            use_skip = 0
            
            # ヘッダーから情報を抽出
            for i, line in enumerate(lines):
                if 'XUNITS' in line: x_unit = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                if 'YUNITS' in line: y_unit = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                if 'XYDATA' in line:
                    use_skip = i + 1
                    break
            
            # セパレータの判別
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
        except Exception as e: st.error(f"Error loading {f.name}: {e}")
    return data_list

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="IR Spectra Analyzer", layout="wide")
    st.title("Advanced IR Spectra Analyzer 🧬")

    if 'data_list' not in st.session_state: st.session_state['data_list'] = []

    # --- サイドバー：1. ロード ---
    st.sidebar.header("1. データ読み込み")
    files = st.sidebar.file_uploader("JASCO CSV/TXTファイルをアップロード", accept_multiple_files=True, type=['csv', 'txt'])
    if files:
        st.session_state['data_list'] = load_data(files)

    # --- サイドバー：2. 表示設定 ---
    st.sidebar.header("2. 表示・補正設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    selected = st.sidebar.multiselect("表示ファイル", all_labels, default=all_labels)

    # IR特有の設定
    invert_x = st.sidebar.checkbox("X軸を逆転させる (IR標準)", value=True)
    
    # ベースライン補正
    bl_mode = st.sidebar.selectbox("ベースライン補正", ["None", "Constant", "Linear"])
    bl_params = {}
    if bl_mode != "None":
        bl_params['p1'] = st.sidebar.number_input("基準点1 (x)", 4000.0)
        if bl_mode == "Linear": bl_params['p2'] = st.sidebar.number_input("基準点2 (x)", 500.0)

    # --- サイドバー：3. フィッティング ---
    st.sidebar.header("3. マルチガウスフィッティング")
    do_fit = st.sidebar.checkbox("フィッティング実行")
    num_peaks = st.sidebar.number_input("ピーク数", 1, 10, 1)
    fit_target = st.sidebar.selectbox("対象ファイル", selected) if selected else None
    
    # --- グラフ描画 ---
    if selected:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for item in [d for d in st.session_state['data_list'] if d['label'] in selected]:
            x, y = item['x'], item['y'].copy()

            # ベースライン補正処理
            if bl_mode == "Constant":
                y -= y[np.abs(x - bl_params['p1']).argmin()]
            elif bl_mode == "Linear":
                i1, i2 = np.abs(x - bl_params['p1']).argmin(), np.abs(x - bl_params['p2']).argmin()
                slope = (y[i2] - y[i1]) / (x[i2] - x[i1])
                y -= (slope * (x - x[i1]) + y[i1])

            ax.plot(x, y, label=item['label'], alpha=0.8)

            # フィッティング (前回のロジックを継承)
            if do_fit and item['label'] == fit_target:
                try:
                    # 全範囲で初期値を自動推定
                    p0 = []
                    found, _ = find_peaks(y, prominence=len(y)*0.001)
                    idx_peaks = found[:num_peaks] if len(found) >= num_peaks else np.linspace(0, len(x)-1, num_peaks, dtype=int)
                    for idx in idx_peaks:
                        p0.extend([y[idx], x[idx], 10.0]) # Amp, Cen, Sigma
                    p0.append(np.min(y)) # Offset
                    
                    popt, _ = curve_fit(multi_gaussian, x, y, p0=p0)
                    ax.plot(x, multi_gaussian(x, *popt), 'r--', label="Total Fit")
                    
                    # 各ピークの詳細を表示
                    res_data = []
                    for n in range(num_peaks):
                        res_data.append({"Peak": n+1, "Center": popt[n*3+1], "Amp": popt[n*3], "FWHM": 2.355*abs(popt[n*3+2])})
                    st.sidebar.table(pd.DataFrame(res_data))
                except: st.sidebar.warning("Fitting failed.")

        # 軸設定
        ax.set_xlabel(item['x_unit']); ax.set_ylabel(item['y_unit'])
        if invert_x: ax.invert_xaxis()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        # 保存用
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        st.download_button("グラフを保存 (PNG)", buf.getvalue(), "ir_plot.png")
    else:
        st.info("👈 左側のサイドバーからファイルをロードしてください。")

if __name__ == "__main__": main()