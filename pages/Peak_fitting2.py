import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. 解析用関数（マルチガウスモデル）
# ---------------------------------------------------------
def multi_gaussian(x, *params):
    """
    複数のガウス関数の和 + オフセット
    params: [amp1, cen1, sig1, amp2, cen2, sig2, ..., offset]
    """
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        amp, cen, sigma = params[i], params[i+1], params[i+2]
        y += amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
    y += params[-1] # offset
    return y

def trans_to_abs(y_trans):
    """透過率(%)を吸光度(Abs)に変換"""
    y_clamped = np.clip(y_trans, 1e-5, 100.0)
    return 2.0 - np.log10(y_clamped)

# ---------------------------------------------------------
# 2. データ読み込み (JASCO形式対応)
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
            x_unit, y_unit = "Wavelength (nm)", "Absorbance"
            use_skip = 0
            
            for i, line in enumerate(lines):
                if 'XUNITS' in line: x_unit = line.split()[-1]
                if 'YUNITS' in line: y_unit = line.split()[-1]
                if 'XYDATA' in line:
                    use_skip = i + 1
                    break
            
            # タブまたはカンマ区切りに対応
            df = pd.read_csv(io.StringIO(text), sep=None, skiprows=use_skip, header=None, engine='python')
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
            st.error(f"Error: {e}")
    return data_list

# ---------------------------------------------------------
# 3. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="UV-Vis Waveform Solver", layout="wide")
    st.title("UV-Vis Peak Deconvolution 🧪")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：データ管理 ---
    st.sidebar.header("1. データ読み込み")
    files = st.sidebar.file_uploader("JASCO形式 (.txt / .csv)", accept_multiple_files=True)
    if files:
        if st.sidebar.button("データを読み込む"):
            st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("左側のサイドバーからスペクトルデータをアップロードしてください。")
        return

    # --- サイドバー：表示設定 ---
    st.sidebar.header("2. 表示・補正設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    selected = st.sidebar.multiselect("表示ファイル", all_labels, default=all_labels[:1])
    
    y_mode = st.sidebar.radio("縦軸モード", ["Absorbance", "Transmittance (%)"])
    
    st.sidebar.subheader("表示範囲 (nm)")
    x_min_val = float(np.min([d['x'].min() for d in st.session_state['data_list']]))
    x_max_val = float(np.max([d['x'].max() for d in st.session_state['data_list']]))
    x_range = st.sidebar.slider("範囲選択", x_min_val, x_max_val, (x_min_val, x_max_val))

    # --- サイドバー：波形分解設定 ---
    st.sidebar.header("3. 波形分解 (ガウスフィッティング)")
    do_fit = st.sidebar.checkbox("フィッティングを実行")
    num_peaks = st.sidebar.number_input("推定ピーク数", 1, 10, 2)
    fit_target = st.sidebar.selectbox("解析対象", selected) if selected else None

    # --- メイン描画エリア ---
    if selected:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for item in [d for d in st.session_state['data_list'] if d['label'] in selected]:
            x, y = item['x'], item['y'].copy()

            # 単位の自動変換
            if y_mode == "Absorbance" and "TRANSMITTANCE" in item['y_unit'].upper():
                y = trans_to_abs(y)
            elif y_mode == "Transmittance (%)" and "ABSORBANCE" in item['y_unit'].upper():
                y = 10**(2 - y)

            ax.plot(x, y, label=item['label'], lw=1.5, alpha=0.7)

            # --- フィッティング処理 ---
            if do_fit and item['label'] == fit_target:
                mask = (x >= x_range[0]) & (x <= x_range[1])
                xf, yf = x[mask], y[mask]
                
                try:
                    # 初期値の自動推定 (Scipyのfind_peaksを利用)
                    p0 = []
                    found, _ = find_peaks(yf, prominence=np.ptp(yf)*0.05)
                    # ピークが見つからない場合は等間隔
                    if len(found) < num_peaks:
                        initial_centers = np.linspace(xf.min(), xf.max(), num_peaks)
                    else:
                        initial_centers = xf[found[:num_peaks]]
                    
                    for c in initial_centers:
                        p0.extend([np.max(yf), c, (xf.max()-xf.min())/10])
                    p0.append(np.min(yf)) # offset

                    popt, _ = curve_fit(multi_gaussian, xf, yf, p0=p0)
                    
                    # 合計曲線
                    x_fine = np.linspace(xf.min(), xf.max(), 500)
                    ax.plot(x_fine, multi_gaussian(x_fine, *popt), 'r--', lw=2, label="Total Fit")
                    
                    # 個別ピークの描画
                    res_table = []
                    for n in range(num_peaks):
                        p_single = list(popt[n*3:(n+1)*3]) + [popt[-1]]
                        y_single = multi_gaussian(x_fine, *p_single) - popt[-1] # オフセット抜き
                        ax.fill_between(x_fine, popt[-1], y_single + popt[-1], alpha=0.2, label=f"Peak {n+1}")
                        
                        res_table.append({
                            "Peak": n+1,
                            "Center (nm)": f"{popt[n*3+1]:.2f}",
                            "Height": f"{popt[n*3]:.3f}",
                            "FWHM (nm)": f"{2.355*abs(popt[n*3+2]):.2f}"
                        })
                    
                    st.subheader(f"📊 {fit_target} の波形分解結果")
                    st.table(pd.DataFrame(res_table))
                    
                except Exception as e:
                    st.error(f"フィッティングに失敗しました。範囲やピーク数を見直してください。: {e}")

        ax.set_xlim(x_range)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(y_mode)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)

        # ダウンロード
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        st.download_button("グラフをPNGで保存", buf.getvalue(), "uv_vis_analysis.png", "image/png")

if __name__ == "__main__":
    main()