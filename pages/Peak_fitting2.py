import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. モデル関数
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Solver Pro", layout="wide")
    st.title("Advanced Spectra Fitting Pro 🎯")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # サイドバー：データ管理
    st.sidebar.header("1. データ管理")
    files = st.sidebar.file_uploader("JASCOファイルをアップロード", accept_multiple_files=True)
    if files and st.sidebar.button("データを読み込む"):
        # 以前定義したload_data関数を使用（内部処理は省略せず維持）
        from main_logic import load_data 
        st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("左側のサイドバーからデータをアップロードしてください。")
        return

    # サイドバー：設定
    st.sidebar.header("2. フィッティング・表示設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    target = st.sidebar.selectbox("解析対象", all_labels)
    data_item = next(d for d in st.session_state['data_list'] if d['label'] == target)
    
    func_mode = st.sidebar.radio("使用する関数", ["Gaussian", "Lorentzian"])
    num_peaks = st.sidebar.number_input("ピーク数", 1, 5, 2)

    # --- 解析・表示範囲の手動入力 ---
    st.sidebar.subheader("解析・表示範囲の設定")
    x_min_data = float(data_item['x'].min())
    x_max_data = float(data_item['x'].max())
    
    col1, col2 = st.sidebar.columns(2)
    # ユーザーが直接数値を打ち込めるボックス
    input_start = col1.number_input("開始(nm/cm⁻¹)", value=x_min_data)
    input_end = col2.number_input("終了(nm/cm⁻¹)", value=x_max_data)
    
    # 数値入力をスライダーの初期値として連動
    zoom_range = st.sidebar.slider("範囲をスライダーで微調整", 
                                   x_min_data, x_max_data, 
                                   (input_start, input_end))

    # --- 各ピークのCenter指定 ---
    st.sidebar.subheader("各ピークの予想位置(Center)")
    manual_centers = []
    for n in range(num_peaks):
        # 初期値は範囲の中央付近に分散させる
        default_c = zoom_range[0] + (zoom_range[1]-zoom_range[0]) * (n+1)/(num_peaks+1)
        c_val = st.sidebar.number_input(f"Peak {n+1} 中心", value=float(default_c), key=f"c_{n}")
        manual_centers.append(c_val)

    # --- メイン描画エリア ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x, y = data_item['x'], data_item['y'].copy()
    ax.plot(x, y, label="Original Data", color="black", alpha=0.3, lw=1)

    if st.sidebar.checkbox("フィッティング開始"):
        # 指定された範囲内(zoom_range)のデータのみを抽出
        mask = (x >= zoom_range[0]) & (x <= zoom_range[1])
        xf, yf = x[mask], y[mask]

        try:
            p0 = []
            lower, upper = [], []
            for c_init in manual_centers:
                # [強度, 中心, 幅]
                p0.extend([np.max(yf), c_init, (zoom_range[1]-zoom_range[0])/20])
                # 指定Centerから一定範囲内(±30)に拘束
                lower.extend([0, c_init - 30, 0.1])
                upper.extend([np.inf, c_init + 30, 100.0])
            p0.append(np.min(yf)) # offset
            lower.append(-np.inf); upper.append(np.inf)

            popt, _ = curve_fit(
                lambda x, *p: multi_model(x, *p, model_type=func_mode),
                xf, yf, p0=p0, bounds=(lower, upper), maxfev=15000
            )

            # フィッティング曲線の描画
            x_fine = np.linspace(zoom_range[0], zoom_range[1], 600)
            y_fit = multi_model(x_fine, *popt, model_type=func_mode)
            ax.plot(x_fine, y_fit, 'r-', lw=2, label="Total Fit")

            # 個別ピークの塗りつぶし
            res_list = []
            for n in range(num_peaks):
                p_peak = list(popt[n*3:(n+1)*3]) + [popt[-1]]
                y_p = multi_model(x_fine, *p_peak, model_type=func_mode) - popt[-1]
                ax.fill_between(x_fine, popt[-1], y_p + popt[-1], alpha=0.3, label=f"Peak {n+1}")
                
                fwhm = 2.355*popt[n*3+2] if func_mode == "Gaussian" else 2*popt[n*3+2]
                res_list.append({"Peak": n+1, "Center": f"{popt[n*3+1]:.2f}", "FWHM": f"{fwhm:.2f}"})
            
            st.table(pd.DataFrame(res_list))

        except Exception as e:
            st.error(f"フィッティング失敗: {e}")

    # 表示範囲の適用
    ax.set_xlim(zoom_range)
    # 表示範囲に合わせて縦軸を自動ズーム
    y_visible = y[(x >= zoom_range[0]) & (x <= zoom_range[1])]
    if len(y_visible) > 0:
        ax.set_ylim(y_visible.min() - 0.05, y_visible.max() + 0.1)

    ax.set_xlabel(data_item.get('x_unit', 'X'))
    ax.set_ylabel(data_item.get('y_unit', 'Y'))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

if __name__ == "__main__":
    main()