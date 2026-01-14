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
# 2. データ読み込み (JASCO形式)
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

    st.sidebar.header("1. データ管理")
    files = st.sidebar.file_uploader("JASCOファイルをアップロード", accept_multiple_files=True)
    if files and st.sidebar.button("データを読み込む/更新"):
        st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("左側のサイドバーからデータを読み込んでください。")
        return

    st.sidebar.header("2. フィッティング・表示設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    target = st.sidebar.selectbox("解析対象", all_labels)
    
    func_mode = st.sidebar.radio("使用する関数", ["Gaussian", "Lorentzian"])
    num_peaks = st.sidebar.number_input("ピーク数", 1, 10, 2)
    
    # 解析対象データの取得
    data_item = next(d for d in st.session_state['data_list'] if d['label'] == target)
    x_data = data_item['x']
    
    # --- グラフ表示範囲の改善 ---
    st.sidebar.subheader("表示・解析範囲の設定")
    # 初期表示範囲をデータの「全範囲」ではなく「中央寄り」に設定するロジック
    x_min, x_max = float(x_data.min()), float(x_data.max())
    # ここでデフォルトを少し狭めに（例：下20%〜上20%をカット）設定することも可能
    # 今回は手動入力を強化
    c1, c2 = st.sidebar.columns(2)
    start_x = c1.number_input("開始(nm)", value=x_min)
    end_x = c2.number_input("終了(nm)", value=x_max)
    
    # ズーム用スライダー（微調整用）
    zoom_range = st.sidebar.slider("範囲をスライドで微調整", x_min, x_max, (start_x, end_x))

    fig, ax = plt.subplots(figsize=(10, 5)) # 少し高さを抑えてコンパクトに
    x, y = data_item['x'], data_item['y'].copy()
    ax.plot(x, y, label="Original Data", color="black", lw=1.2, alpha=0.4)

    if st.sidebar.checkbox("フィッティング開始"):
        # スライダーの値を範囲として採用
        mask = (x >= zoom_range[0]) & (x <= zoom_range[1])
        xf, yf = x[mask], y[mask]

        try:
            found, _ = find_peaks(yf, prominence=np.ptp(yf)*0.05)
            p0 = []
            if len(found) >= num_peaks:
                top_peaks = found[np.argsort(yf[found])[-num_peaks:]]
                for idx in top_peaks:
                    p0.extend([yf[idx], xf[idx], (xf.max()-xf.min())/25])
            else:
                for c in np.linspace(xf.min(), xf.max(), num_peaks):
                    p0.extend([np.max(yf), c, (xf.max()-xf.min())/25])
            p0.append(np.min(yf))

            lower, upper = [], []
            for _ in range(num_peaks):
                lower.extend([0, xf.min(), 0.01])
                upper.extend([np.inf, xf.max(), (xf.max()-xf.min())])
            lower.append(-np.inf); upper.append(np.inf)

            popt, _ = curve_fit(
                lambda x, *p: multi_model(x, *p, model_type=func_mode),
                xf, yf, p0=p0, bounds=(lower, upper), maxfev=15000
            )

            x_fine = np.linspace(xf.min(), xf.max(), 600)
            y_fit = multi_model(x_fine, *popt, model_type=func_mode)
            ax.plot(x_fine, y_fit, 'r-', lw=2, label=f"Total {func_mode} Fit")

            res_table = []
            for n in range(num_peaks):
                p_peak = list(popt[n*3:(n+1)*3]) + [popt[-1]]
                y_p = multi_model(x_fine, *p_peak, model_type=func_mode) - popt[-1]
                ax.fill_between(x_fine, popt[-1], y_p + popt[-1], alpha=0.3, label=f"Peak {n+1}")
                fwhm = 2.355*popt[n*3+2] if func_mode == "Gaussian" else 2*popt[n*3+2]
                res_table.append({"Peak": n+1, "Center": f"{popt[n*3+1]:.2f}", "FWHM": f"{fwhm:.2f}"})

            st.table(pd.DataFrame(res_table))

        except Exception as e:
            st.error(f"フィッティング失敗: {e}")

    # グラフ表示範囲をスライダーと連動させて「小さく」制限する
    ax.set_xlim(zoom_range)
    
    # 縦軸も表示範囲内のデータに合わせて自動調整（ズーム感アップ）
    if len(y[(x >= zoom_range[0]) & (x <= zoom_range[1])]) > 0:
        y_visible = y[(x >= zoom_range[0]) & (x <= zoom_range[1])]
        ax.set_ylim(y_visible.min() - 0.05, y_visible.max() + 0.1)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

if __name__ == "__main__":
    main()