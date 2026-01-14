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
            x_unit, y_unit = "Wavelength/Wavenumber", "Intensity"
            use_skip = 0
            for i, line in enumerate(lines):
                if 'XUNITS' in line: x_unit = line.split()[-1].strip(',')
                if 'YUNITS' in line: y_unit = line.split()[-1].strip(',')
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
    st.title("Spectral Fitting 🧪")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：1. データ ---
    st.sidebar.header("1. データ管理")
    files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True)
    if files and st.sidebar.button("データを読み込む"):
        st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("左側のサイドバーからデータをアップロードしてください。")
        return

    # --- サイドバー：2. 設定 ---
    st.sidebar.header("2. フィッティング・表示設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    target = st.sidebar.selectbox("解析対象データ", all_labels)
    data_item = next(d for d in st.session_state['data_list'] if d['label'] == target)
    
    func_mode = st.sidebar.radio("使用する関数", ["Gaussian", "Lorentzian"])
    num_peaks = st.sidebar.number_input("ピーク数", 1, 5, 2)

    # --- 解析範囲の手動入力 ---
    st.sidebar.subheader("解析・表示範囲")
    x_min_data, x_max_data = float(data_item['x'].min()), float(data_item['x'].max())
    
    col1, col2 = st.sidebar.columns(2)
    input_start = col1.number_input("開始", value=x_min_data)
    input_end = col2.number_input("終了", value=x_max_data)
    
    zoom_range = st.sidebar.slider("スライダーで範囲微調整", 
                                   min(x_min_data, x_max_data), 
                                   max(x_min_data, x_max_data), 
                                   (min(input_start, input_end), max(input_start, input_end)))

    # --- 各ピークのCenter指定 ---
    st.sidebar.subheader("予想Center位置")
    manual_centers = []
    for n in range(num_peaks):
        default_c = zoom_range[0] + (zoom_range[1]-zoom_range[0]) * (n+1)/(num_peaks+1)
        c_val = st.sidebar.number_input(f"Peak {n+1} Center", value=float(default_c), key=f"c_{n}")
        manual_centers.append(c_val)

    # --- メイン：グラフ描画 (高さを70%程度に調整) ---
    # figsize=(10, 4) にすることで以前の高さ(6)から大幅にコンパクト化
    fig, ax = plt.subplots(figsize=(10, 4)) 
    x, y = data_item['x'], data_item['y'].copy()
    ax.plot(x, y, label="Experimental", color="black", alpha=0.3, lw=1)

    if st.sidebar.checkbox("フィッティング開始"):
        mask = (x >= min(zoom_range)) & (x <= max(zoom_range))
        xf, yf = x[mask], y[mask]

        try:
            p0 = []
            lower, upper = [], []
            for c_init in manual_centers:
                p0.extend([np.max(yf), c_init, (zoom_range[1]-zoom_range[0])/20])
                lower.extend([0, c_init - 50, 0.1]) # 遊びを±50に設定
                upper.extend([np.inf, c_init + 50, 200.0])
            p0.append(np.min(yf)) # offset
            lower.append(-np.inf); upper.append(np.inf)

            popt, _ = curve_fit(
                lambda x, *p: multi_model(x, *p, model_type=func_mode),
                xf, yf, p0=p0, bounds=(lower, upper), maxfev=15000
            )

            x_fine = np.linspace(zoom_range[0], zoom_range[1], 600)
            y_fit = multi_model(x_fine, *popt, model_type=func_mode)
            ax.plot(x_fine, y_fit, 'r-', lw=2, label=f"Total {func_mode} Fit")

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

    # 軸の設定とズームの適用
    ax.set_xlim(zoom_range)
    y_visible = y[(x >= min(zoom_range)) & (x <= max(zoom_range))]
    if len(y_visible) > 0:
        ax.set_ylim(y_visible.min() - (y_visible.max()-y_visible.min())*0.1, y_visible.max() + (y_visible.max()-y_visible.min())*0.2)

    ax.set_xlabel(data_item.get('x_unit', 'X-axis'))
    ax.set_ylabel(data_item.get('y_unit', 'Y-axis'))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', borderaxespad=0)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # プロットの間隔を調整して画面を有効活用
    plt.tight_layout()
    st.pyplot(fig)

    # 4. ダウンロード
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    st.download_button("グラフをPNG保存", buf.getvalue(), "spectra_result.png")

if __name__ == "__main__":
    main()