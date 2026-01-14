import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. フィッティング用モデル関数
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
            use_skip = 0
            for i, line in enumerate(lines):
                if 'XYDATA' in line:
                    use_skip = i + 1
                    break
            
            # タブ、カンマ、スペース区切りに対応
            df = pd.read_csv(io.StringIO(text), sep=None, skiprows=use_skip, header=None, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] >= 2:
                data_list.append({
                    'label': f.name.rsplit('.', 1)[0],
                    'x': df.iloc[:, 0].values, 
                    'y': df.iloc[:, 1].values
                })
        except Exception as e:
            st.error(f"読み込みエラー: {f.name} ({e})")
    return data_list

# ---------------------------------------------------------
# 3. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="UV-Vis Solver Pro", layout="wide")
    st.title("Spectrum Deconvolution 📊")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # サイドバー：1. データ読み込み
    st.sidebar.header("1. データ管理")
    files = st.sidebar.file_uploader("JASCO形式ファイルをアップロード", accept_multiple_files=True)
    if files and st.sidebar.button("データを読み込む"):
        st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("👈 左側のサイドバーからデータをアップロードしてください。")
        return

    # サイドバー：2. フィッティング設定
    st.sidebar.header("2. フィッティング設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    target = st.sidebar.selectbox("解析対象データ", all_labels)
    data_item = next(d for d in st.session_state['data_list'] if d['label'] == target)
    
    func_mode = st.sidebar.radio("関数選択", ["Gaussian", "Lorentzian"])
    num_peaks = st.sidebar.number_input("ピーク数", 1, 5, 2)

    # 解析範囲の指定（数値入力）
    st.sidebar.subheader("解析範囲 (nm)")
    x_min_data, x_max_data = float(data_item['x'].min()), float(data_item['x'].max())
    col1, col2 = st.sidebar.columns(2)
    input_start = col1.number_input("開始(nm)", value=x_min_data)
    input_end = col2.number_input("終了(nm)", value=x_max_data)
    
    # 予想Center位置
    st.sidebar.subheader("予想Center位置 (nm)")
    manual_centers = []
    for n in range(num_peaks):
        default_c = input_start + (input_end - input_start) * (n+1)/(num_peaks+1)
        c_val = st.sidebar.number_input(f"Peak {n+1} Center", value=float(default_c), key=f"c_{n}")
        manual_centers.append(c_val)

    # --- Plotlyグラフ作成 ---
    fig = go.Figure()
    x_all, y_all = data_item['x'], data_item['y']

    # 元データプロット
    fig.add_trace(go.Scatter(
        x=x_all, y=y_all, 
        name="Experimental", 
        mode='lines', 
        line=dict(color='gray', width=1.5), 
        opacity=0.4
    ))

    if st.sidebar.checkbox("フィッティング開始"):
        # 範囲フィルタリング
        mask = (x_all >= min(input_start, input_end)) & (x_all <= max(input_start, input_end))
        xf, yf = x_all[mask], y_all[mask]

        try:
            p0 = []
            lower, upper = [], []
            for c_init in manual_centers:
                # [Amp, Center, Sigma]
                p0.extend([np.max(yf), c_init, abs(input_end-input_start)/30])
                lower.extend([0, c_init - 50, 0.05])
                upper.extend([np.inf, c_init + 50, 150.0])
            p0.append(np.min(yf)) # offset
            lower.append(-np.inf); upper.append(np.inf)

            popt, _ = curve_fit(
                lambda x, *p: multi_model(x, *p, model_type=func_mode),
                xf, yf, p0=p0, bounds=(lower, upper), maxfev=15000
            )

            # 解析範囲での描画用データ
            x_range_vals = np.linspace(min(input_start, input_end), max(input_start, input_end), 1000)
            y_total_fit = multi_model(x_range_vals, *popt, model_type=func_mode)
            
            # 合計曲線
            fig.add_trace(go.Scatter(x=x_range_vals, y=y_total_fit, name="Total Fit", line=dict(color='red', width=3)))

            # 各成分ピークの描画
            res_list = []
            for n in range(num_peaks):
                p_single = list(popt[n*3:(n+1)*3]) + [popt[-1]]
                y_single = multi_model(x_range_vals, *p_single, model_type=func_mode)
                
                fig.add_trace(go.Scatter(
                    x=x_range_vals, y=y_single, 
                    name=f"Peak {n+1}", 
                    fill='tozeroy', 
                    opacity=0.3
                ))
                
                fwhm = (2.355 * popt[n*3+2]) if func_mode == "Gaussian" else (2 * popt[n*3+2])
                res_list.append({
                    "Peak": n+1, 
                    "Center (nm)": f"{popt[n*3+1]:.2f}", 
                    "Abs. (Height)": f"{popt[n*3]:.4f}",
                    "FWHM (nm)": f"{fwhm:.2f}"
                })
            
            st.subheader("📋 フィッティング結果要約")
            st.table(pd.DataFrame(res_list))

        except Exception as e:
            st.error(f"解析エラー: {e}")

    # レイアウト設定 (軸ラベルとインタラクティブ機能)
    fig.update_layout(
        height=450,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorbance (Abs.)",
        hovermode="x unified",
        xaxis=dict(
            range=[input_start, input_end],
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=0.5,
            spikedash='dot',
            spikecolor='blue'
        ),
        yaxis=dict(showspikes=True, spikesnap='cursor', spikedash='dot', spikecolor='blue'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **💡 操作ガイド:**
    - **マウスホバー:** 波長(nm)と吸光度(Abs.)を精密に読み取れます。
    - **ドラッグ:** 特定の波長域を拡大（ズーム）できます。
    - **ダブルクリック:** ズームをリセットします。
    """)

if __name__ == "__main__":
    main()