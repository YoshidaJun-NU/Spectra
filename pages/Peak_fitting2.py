import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.optimize import curve_fit

# --- モデル関数 (Gaussian/Lorentzian) ---
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

def lorentzian(x, amp, cen, sigma):
    return amp * (sigma**2 / ((x - cen)**2 + sigma**2))

def multi_model(x, *params, model_type="Gaussian"):
    y = np.zeros_like(x)
    offset = params[-1]
    for i in range(0, len(params) - 1, 3):
        amp, cen, sig = params[i], params[i+1], params[i+2]
        if model_type == "Gaussian":
            y += gaussian(x, amp, cen, sig)
        else:
            y += lorentzian(x, amp, cen, sig)
    return y + offset

def main():
    st.set_page_config(page_title="Spectra Manual Fit", layout="wide")
    st.title("Spectra Fitting with Manual Peak Positions 🎯")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # 1. データ管理 (省略せず記載)
    st.sidebar.header("1. データ読み込み")
    files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True)
    if files and st.sidebar.button("読み込み"):
        # load_data関数は以前のものを使用
        from main_logic import load_data # もし外部化していれば
        st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("左側からデータを読み込んでください。")
        return

    # 2. フィッティング設定
    st.sidebar.header("2. フィッティング設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    target = st.sidebar.selectbox("解析対象", all_labels)
    data_item = next(d for d in st.session_state['data_list'] if d['label'] == target)
    
    func_mode = st.sidebar.radio("関数選択", ["Gaussian", "Lorentzian"])
    num_peaks = st.sidebar.number_input("ピーク数", 1, 5, 2)

    # --- おおよそのCenter値を入力するセクション ---
    st.sidebar.subheader("各ピークの予想Center値")
    manual_centers = []
    for n in range(num_peaks):
        val = st.sidebar.number_input(f"Peak {n+1} の中心 (nm)", 
                                      value=float(np.median(data_item['x'])), 
                                      step=1.0, key=f"peak_{n}")
        manual_centers.append(val)

    x_min, x_max = float(data_item['x'].min()), float(data_item['x'].max())
    zoom_range = st.sidebar.slider("表示/解析範囲", x_min, x_max, (x_min, x_max))

    # 3. 描画とフィッティング
    fig, ax = plt.subplots(figsize=(10, 5))
    x, y = data_item['x'], data_item['y'].copy()
    ax.plot(x, y, label="Original", color="black", alpha=0.3)

    if st.sidebar.checkbox("フィッティング開始"):
        mask = (x >= zoom_range[0]) & (x <= zoom_range[1])
        xf, yf = x[mask], y[mask]

        try:
            p0 = []
            lower, upper = [], []
            for c_init in manual_centers:
                # 初期値設定: [強度, 中心, 幅]
                p0.extend([np.max(yf), c_init, 5.0])
                # 境界条件: 中心値が指定から±30nm以上離れないように制限
                lower.extend([0, c_init - 30, 0.1])
                upper.extend([np.inf, c_init + 30, 100.0])
            p0.append(np.min(yf)) # offset
            lower.append(-np.inf); upper.append(np.inf)

            popt, _ = curve_fit(
                lambda x, *p: multi_model(x, *p, model_type=func_mode),
                xf, yf, p0=p0, bounds=(lower, upper), maxfev=10000
            )

            # 結果描画
            x_fine = np.linspace(xf.min(), xf.max(), 500)
            ax.plot(x_fine, multi_model(x_fine, *popt, model_type=func_mode), 'r-', label="Total Fit")
            
            res_table = []
            for n in range(num_peaks):
                p_peak = list(popt[n*3:(n+1)*3]) + [popt[-1]]
                y_p = multi_model(x_fine, *p_peak, model_type=func_mode) - popt[-1]
                ax.fill_between(x_fine, popt[-1], y_p + popt[-1], alpha=0.3, label=f"Peak {n+1}")
                res_table.append({"Peak": n+1, "Result Center": f"{popt[n*3+1]:.2f}"})
            st.table(pd.DataFrame(res_table))

        except Exception as e:
            st.error(f"フィッティング失敗: {e}")

    ax.set_xlim(zoom_range)
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()