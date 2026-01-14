import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. モデル関数定義
# ---------------------------------------------------------
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

def lorentzian(x, amp, cen, sigma):
    # sigma は半値全幅(FWHM)の半分として定義
    return amp * (sigma**2 / ((x - cen)**2 + sigma**2))

def multi_model(x, *params, model_type="Gaussian"):
    """複数のピーク + オフセットを計算"""
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
# 2. データ読み込みロジック (JASCO形式)
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
            
            df = pd.read_csv(io.StringIO(text), sep=None, skiprows=use_skip, header=None, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] >= 2:
                data_list.append({
                    'label': f.name.rsplit('.', 1)[0],
                    'x': df.iloc[:, 0].values, 'y': df.iloc[:, 1].values,
                    'x_unit': x_unit, 'y_unit': y_unit
                })
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")
    return data_list

# ---------------------------------------------------------
# 3. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Solver Pro", layout="wide")
    st.title("Advanced Waveform Deconvolution 🧪")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：データ ---
    st.sidebar.header("1. データ管理")
    files = st.sidebar.file_uploader("JASCOファイルをアップロード", accept_multiple_files=True)
    if files and st.sidebar.button("データを読み込む/更新"):
        st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("左側のサイドバーからスペクトルデータを読み込んでください。")
        return

    # --- サイドバー：設定 ---
    st.sidebar.header("2. フィッティング設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    target = st.sidebar.selectbox("解析対象", all_labels)
    
    # 関数の切り替え
    func_mode = st.sidebar.radio("使用する関数", ["Gaussian", "Lorentzian"])
    
    
    num_peaks = st.sidebar.number_input("ピーク数", 1, 10, 2)
    
    # 解析範囲の指定
    data_item = next(d for d in st.session_state['data_list'] if d['label'] == target)
    x_range = st.sidebar.slider("解析範囲(nm)", 
                                float(data_item['x'].min()), float(data_item['x'].max()), 
                                (float(data_item['x'].min()), float(data_item['x'].max())))

    # --- メイン：グラフと解析 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x, y = data_item['x'], data_item['y'].copy()
    ax.plot(x, y, label="Original Data", color="black", lw=1, alpha=0.5)

    if st.sidebar.checkbox("フィッティング開始"):
        mask = (x >= x_range[0]) & (x <= x_range[1])
        xf, yf = x[mask], y[mask]

        # --- 強力なフィッティングロジック ---
        try:
            # 1. 初期値の推定
            found, _ = find_peaks(yf, prominence=np.ptp(yf)*0.05)
            p0 = []
            if len(found) >= num_peaks:
                # 強度の高い順に採用
                top_peaks = found[np.argsort(yf[found])[-num_peaks:]]
                for idx in top_peaks:
                    p0.extend([yf[idx], xf[idx], (xf.max()-xf.min())/20])
            else:
                for c in np.linspace(xf.min(), xf.max(), num_peaks):
                    p0.extend([np.max(yf), c, (xf.max()-xf.min())/20])
            p0.append(np.min(yf)) # offset

            # 2. 境界条件の設定 (負の強度や範囲外を禁止)
            lower, upper = [], []
            for _ in range(num_peaks):
                lower.extend([0, xf.min(), 0.1])
                upper.extend([np.inf, xf.max(), (xf.max()-xf.min())])
            lower.append(-np.inf); upper.append(np.inf)

            # 3. 計算実行 (maxfevを増加)
            popt, _ = curve_fit(
                lambda x, *p: multi_model(x, *p, model_type=func_mode),
                xf, yf, p0=p0, bounds=(lower, upper), maxfev=10000
            )

            # --- 結果の描画 ---
            x_fine = np.linspace(xf.min(), xf.max(), 500)
            y_fit = multi_model(x_fine, *popt, model_type=func_mode)
            ax.plot(x_fine, y_fit, 'r-', lw=2, label=f"Total {func_mode} Fit")

            res_table = []
            for n in range(num_peaks):
                p_peak = list(popt[n*3:(n+1)*3]) + [popt[-1]]
                y_p = multi_model(x_fine, *p_peak, model_type=func_mode) - popt[-1]
                ax.fill_between(x_fine, popt[-1], y_p + popt[-1], alpha=0.3, label=f"Peak {n+1}")
                
                # FWHM計算 (Lorentzianの場合は 2*sigma, Gaussianは 2.355*sigma)
                fwhm = 2.355*popt[n*3+2] if func_mode == "Gaussian" else 2*popt[n*3+2]
                res_table.append({"Peak": n+1, "Center": f"{popt[n*3+1]:.2f}", "FWHM": f"{fwhm:.2f}"})

            st.table(pd.DataFrame(res_table))

        except Exception as e:
            st.error(f"フィッティング失敗: {e}\n\n範囲を狭めるか、関数を切り替えてみてください。")

    ax.set_xlim(x_range)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.legend(bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)

if __name__ == "__main__":
    main()