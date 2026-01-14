import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
    st.set_page_config(page_title="Spectra Solver Plotly", layout="wide")
    st.title("Interactive Spectra Solver Pro 🧬")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # サイドバー：1. データ
    st.sidebar.header("1. データ管理")
    files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True)
    if files and st.sidebar.button("データを読み込む"):
        st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("👈 左側のサイドバーからデータを読み込んでください。")
        return

    # サイドバー：2. フィッティング設定
    st.sidebar.header("2. フィッティング設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    target = st.sidebar.selectbox("解析対象データ", all_labels)
    data_item = next(d for d in st.session_state['data_list'] if d['label'] == target)
    
    func_mode = st.sidebar.radio("使用する関数", ["Gaussian", "Lorentzian"])
    num_peaks = st.sidebar.number_input("ピーク数", 1, 5, 2)

    # 解析範囲の設定
    st.sidebar.subheader("解析範囲")
    x_min_data, x_max_data = float(data_item['x'].min()), float(data_item['x'].max())
    col1, col2 = st.sidebar.columns(2)
    input_start = col1.number_input("開始", value=x_min_data)
    input_end = col2.number_input("終了", value=x_max_data)
    
    # 予想Center位置の設定
    st.sidebar.subheader("予想Center位置")
    manual_centers = []
    for n in range(num_peaks):
        default_c = input_start + (input_end - input_start) * (n+1)/(num_peaks+1)
        c_val = st.sidebar.number_input(f"Peak {n+1} Center", value=float(default_c), key=f"c_{n}")
        manual_centers.append(c_val)

    # --- Plotlyグラフの作成 ---
    fig = go.Figure()
    x, y = data_item['x'], data_item['y'].copy()

    # オリジナルデータ
    fig.add_trace(go.Scatter(x=x, y=y, name="Experimental", mode='lines', line=dict(color='gray', width=1), opacity=0.5))

    if st.sidebar.checkbox("フィッティング開始"):
        # 範囲選択（数値入力に基づきマスクを作成）
        mask = (x >= min(input_start, input_end)) & (x <= max(input_start, input_end))
        xf, yf = x[mask], y[mask]

        try:
            p0 = []
            lower, upper = [], []
            for c_init in manual_centers:
                p0.extend([np.max(yf), c_init, abs(input_end-input_start)/20])
                lower.extend([0, c_init - 50, 0.1])
                upper.extend([np.inf, c_init + 50, 200.0])
            p0.append(np.min(yf))
            lower.append(-np.inf); upper.append(np.inf)

            popt, _ = curve_fit(
                lambda x, *p: multi_model(x, *p, model_type=func_mode),
                xf, yf, p0=p0, bounds=(lower, upper), maxfev=15000
            )

            # フィッティング合計曲線
            x_fine = np.linspace(min(input_start, input_end), max(input_start, input_end), 800)
            y_fit = multi_model(x_fine, *popt, model_type=func_mode)
            fig.add_trace(go.Scatter(x=x_fine, y=y_fit, name="Total Fit", line=dict(color='red', width=2.5)))

            # 個別ピーク
            res_list = []
            for n in range(num_peaks):
                p_peak = list(popt[n*3:(n+1)*3]) + [popt[-1]]
                y_p = multi_model(x_fine, *p_peak, model_type=func_mode)
                fig.add_trace(go.Scatter(x=x_fine, y=y_p, name=f"Peak {n+1}", fill='tozeroy', opacity=0.3))
                fwhm = 2.355*popt[n*3+2] if func_mode == "Gaussian" else 2*popt[n*3+2]
                res_list.append({"Peak": n+1, "Center": f"{popt[n*3+1]:.2f}", "FWHM": f"{fwhm:.2f}"})
            
            st.table(pd.DataFrame(res_list))
        except Exception as e:
            st.error(f"Fitting failed: {e}")

    # --- レイアウト調整 (高さとインタラクティブ機能) ---
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=data_item.get('x_unit', 'X'),
        yaxis_title=data_item.get('y_unit', 'Y'),
        hovermode="x unified",  # マウス位置の全データを一括表示
        xaxis=dict(
            range=[input_start, input_end],
            showspikes=True, # 十字線
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
        ),
        yaxis=dict(showspikes=True, spikesnap='cursor'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 グラフ上をマウスでホバーすると座標が表示されます。ドラッグで特定範囲をズームできます。")

if __name__ == "__main__":
    main()