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
    """
    複数のガウス関数の和を計算
    params: [amp1, cen1, sig1, amp2, cen2, sig2, ..., offset]
    """
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        amp = params[i]
        cen = params[i+1]
        sigma = params[i+2]
        y += amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
    y += params[-1] # offset
    return y

# ---------------------------------------------------------
# データ読み込み・スタイル設定 (これまでの機能を継承)
# ---------------------------------------------------------
def init_styles(data_list):
    if 'styles' not in st.session_state: st.session_state['styles'] = {}
    for i, item in enumerate(data_list):
        if item['label'] not in st.session_state['styles']:
            st.session_state['styles'][item['label']] = {
                'color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][i % 5],
                'linewidth': 1.5, 'linestyle': 'Solid (実線)'
            }

def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    for f in uploaded_files:
        try:
            content = f.getvalue()
            for enc in ['utf-8', 'cp932', 'shift_jis', 'latin1']:
                try: text = content.decode(enc); break
                except: continue
            
            use_sep = ',' if separator == 'comma' else '\t'
            use_skip, use_header = skip_rows, (0 if has_header else None)
            
            if 'XYDATA' in text:
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if 'XYDATA' in line:
                        use_skip, use_header = i + 1, None
                        use_sep = ',' if f.name.lower().endswith('.csv') else None
                        break
            
            df = pd.read_csv(io.StringIO(text), sep=use_sep, skiprows=use_skip, header=use_header, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if df.shape[1] >= 2:
                data_list.append({'label': f.name.rsplit('.', 1)[0], 'x': df.iloc[:, 0].values, 'y': df.iloc[:, 1].values})
        except Exception as e: st.error(f"Error: {e}")
    return data_list

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Analyzer Pro", layout="wide")
    st.title("Spectra Analyzer: Multi-Peak Fitting 🧪")

    if 'data_list' not in st.session_state: st.session_state['data_list'] = []

    # --- サイドバー ---
    st.sidebar.header("1. データ読み込み")
    files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True, type=['txt', 'csv', 'dat'])
    if files:
        st.session_state['data_list'] = load_data(files, 'tab', 19, True)
        init_styles(st.session_state['data_list'])

    st.sidebar.header("2. 表示・補正設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    selected = st.sidebar.multiselect("表示ファイル", all_labels, default=all_labels)

    # ベースライン補正
    bl_mode = st.sidebar.selectbox("ベースライン補正", ["None", "Constant", "Linear"])
    bl_params = {}
    if bl_mode == "Constant": bl_params['v'] = st.sidebar.number_input("基準波長(nm)", 700.0)
    if bl_mode == "Linear":
        bl_params['p1'] = st.sidebar.number_input("点1(nm)", 650.0)
        bl_params['p2'] = st.sidebar.number_input("点2(nm)", 750.0)

    # フィッティング設定
    st.sidebar.header("3. マルチガウスフィッティング")
    do_fit = st.sidebar.checkbox("フィッティングを実行")
    num_peaks = st.sidebar.number_input("フィッティングするピーク数", 1, 10, 2)
    fit_range = st.sidebar.slider("解析範囲(nm)", 200, 900, (300, 600))
    fit_target = st.sidebar.selectbox("対象ファイル", selected) if selected else None

    # --- グラフ描画 ---
    if selected:
        fig, ax = plt.subplots(figsize=(10, 6))
        LINE_STYLES = {'Solid (実線)': '-', 'Dashed (破線)': '--', 'Dash-dot (一点鎖線)': '-.', 'Dotted (点線)': ':'}

        for item in [d for d in st.session_state['data_list'] if d['label'] in selected]:
            x, y = item['x'], item['y'].copy()

            # ベースライン補正
            if bl_mode == "Constant":
                y -= y[np.abs(x - bl_params['v']).argmin()]
            elif bl_mode == "Linear":
                i1, i2 = np.abs(x - bl_params['p1']).argmin(), np.abs(x - bl_params['p2']).argmin()
                slope = (y[i2] - y[i1]) / (x[i2] - x[i1])
                y -= (slope * (x - x[i1]) + y[i1])

            style = st.session_state['styles'][item['label']]
            ax.plot(x, y, label=item['label'], color=style['color'], 
                    lw=style['linewidth'], ls=LINE_STYLES[style['linestyle']], alpha=0.7)

            # フィッティング計算
            if do_fit and item['label'] == fit_target:
                mask = (x >= fit_range[0]) & (x <= fit_range[1])
                xf, yf = x[mask], y[mask]

                if len(xf) > (num_peaks * 3):
                    try:
                        # 初期値推定
                        found, _ = find_peaks(yf, prominence=0.005)
                        # ピークが見つからない、または足りない場合は等間隔に配置
                        initial_centers = xf[found][:num_peaks] if len(found) >= num_peaks else np.linspace(xf.min(), xf.max(), num_peaks)
                        
                        p0 = []
                        for c in initial_centers:
                            p0.extend([np.max(yf), c, (xf.max()-xf.min())/10])
                        p0.append(np.min(yf)) # offset

                        popt, _ = curve_fit(multi_gaussian, xf, yf, p0=p0)
                        
                        # フィッティング曲線描画
                        x_fine = np.linspace(xf.min(), xf.max(), 200)
                        ax.plot(x_fine, multi_gaussian(x_fine, *popt), 'r-', lw=2.5, label="Total Fit")
                        
                        # 個別ピークの描画
                        peak_data = []
                        for j in range(num_peaks):
                            p_params = list(popt[j*3 : (j+1)*3]) + [popt[-1]]
                            y_peak = multi_gaussian(x_fine, *p_params) - popt[-1] # オフセット抜き
                            ax.plot(x_fine, y_peak + popt[-1], ':', lw=1.5, label=f"Peak {j+1}")
                            peak_data.append({
                                "Peak": j+1, "Center (nm)": popt[j*3+1], 
                                "Amplitude": popt[j*3], "FWHM (nm)": 2.355 * abs(popt[j*3+2])
                            })
                        
                        st.subheader(f"📊 {fit_target} のフィッティング結果")
                        st.table(pd.DataFrame(peak_data))

                    except Exception as e: st.warning(f"Fitting Error: {e}")

        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 保存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        st.download_button("グラフをPNGで保存", buf.getvalue(), "plot.png", "image/png")
    else:
        st.info("👈 ファイルをロードし、選択してください。")

if __name__ == "__main__": main()