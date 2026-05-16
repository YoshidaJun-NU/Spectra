import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io

# ---------------------------------------------------------
# 1. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Lifetime Fitting Pro", layout="wide")
st.title("📉 Multi-Component Lifetime Fitting")

# ---------------------------------------------------------
# 2. 関数定義
# ---------------------------------------------------------
def load_raw_df(uploaded_file):
    try:
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        lines = content.splitlines()
        data_start_idx = 0
        for i, line in enumerate(lines):
            parts = line.replace('\t', ',').split(',')
            try:
                if len(parts) >= 2 and float(parts[0].strip()) is not None:
                    data_start_idx = i
                    break
            except: continue
        df = pd.read_csv(uploaded_file, skiprows=data_start_idx, header=None, sep=None, engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return None

def multi_exp_model(t, b, *params):
    y = b
    for i in range(0, len(params), 2):
        A = params[i]
        tau = params[i+1]
        y += A * np.exp(-t / max(tau, 1e-10))
    return y

# ---------------------------------------------------------
# 3. サイドバー
# ---------------------------------------------------------
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv", "txt", "dat"])

if uploaded_file:
    raw_df = load_raw_df(uploaded_file)
    if raw_df is not None:
        st.sidebar.subheader("列の選択")
        col_names = [f"Column {i}" for i in range(raw_df.shape[1])]
        x_col_idx = st.sidebar.selectbox("Time (X軸)", range(len(col_names)), index=0)
        y_col_idx = st.sidebar.selectbox("Intensity (Y軸)", range(len(col_names)), index=1)
        
        df = pd.DataFrame({'Time': raw_df.iloc[:, x_col_idx], 'Intensity': raw_df.iloc[:, y_col_idx]})

        st.sidebar.header("2. Plot Appearance")
        with st.sidebar.expander("見た目の詳細設定"):
            raw_color = st.color_picker("点の色", value="#808080")
            marker_type = st.selectbox("点の種類", ["o", ".", "x", "None"], index=0)
            marker_size = st.slider("点のサイズ", 0, 15, 4)
            fit_color = st.color_picker("Fit線の色", value="#C12626")
            fit_lw = st.slider("Fit線の太さ", 0.5, 5.0, 4.0)
            show_legend = st.checkbox("凡例を表示する", value=False)
            # 【機能追加】目盛り線のオンオフ
            show_grid = st.checkbox("目盛り線を表示する", value=True)
            tick_fontsize = st.number_input("目盛フォントサイズ", min_value=1, max_value=40, value=16, step=1)

        # ---------------------------------------------------------
        # 4. メインレイアウト (2カラム構成)
        # ---------------------------------------------------------
        col_graph, col_ctrl = st.columns([2, 1])
        
        with col_ctrl:
            st.subheader("Fitting Control")
            n_comp = st.selectbox("成分数 (n)", [1, 2, 3, 4, 5], index=0)
            
            t_min, t_max = float(df['Time'].min()), float(df['Time'].max())
            idx_peak = df['Intensity'].idxmax()
            t_peak = float(df.loc[idx_peak, 'Time'])
            
            fit_range = st.slider("Fitting Range", t_min, t_max, (t_peak, t_max))
            
            mask = (df['Time'] >= fit_range[0]) & (df['Time'] <= fit_range[1])
            df_fit = df[mask].copy()
            t_fit = df_fit['Time'].values - fit_range[0] 
            y_fit = df_fit['Intensity'].values

            b_init = y_fit.min()
            amp_total = y_fit.max() - b_init
            p0 = [b_init]
            bounds_l, bounds_u = [-np.inf], [np.inf]
            for i in range(n_comp):
                p0.extend([amp_total / n_comp, (t_fit.max() / (i+1))])
                bounds_l.extend([0, 1e-10])
                bounds_u.extend([np.inf, np.inf])

            fit_success = False
            try:
                popt, pcov = curve_fit(multi_exp_model, t_fit, y_fit, p0=p0, bounds=(bounds_l, bounds_u), maxfev=20000)
                fit_success = True
                
                residuals = y_fit - multi_exp_model(t_fit, *popt)
                r_squared = 1 - (np.sum(residuals**2) / np.sum((y_fit - np.mean(y_fit))**2))
            except Exception as e:
                st.error(f"Fitting Failed: {e}")

            if fit_success:
                st.divider()
                st.subheader("📋 Detailed Report")
                
                c1, c2 = st.columns(2)
                c1.metric("R-squared ($R^2$)", f"{r_squared:.4f}")
                c2.metric("Baseline ($b$)", f"{popt[0]:.2e}")

                st.latex(r"I(t) = b + \sum A_i e^{-t/\tau_i}")
                
                amps = [popt[2*i+1] for i in range(n_comp)]
                taus = [popt[2*i+2] for i in range(n_comp)]
                total_amp = sum(amps)
                
                rows = []
                for i in range(n_comp):
                    contribution = (amps[i] / total_amp) * 100
                    rows.append({
                        "Comp": i+1,
                        "Amplitude": f"{amps[i]:.2e}",
                        "τ": f"{taus[i]:.3f}",
                        "%": f"{contribution:.1f}"
                    })
                st.table(pd.DataFrame(rows))

        with col_graph:
            is_log = st.checkbox("Y軸をログスケールにする", value=False)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(df['Time'], df['Intensity'], color=raw_color, marker=marker_type, ls='None', 
                    markersize=marker_size, alpha=0.4, label='Raw Data')

            if fit_success:
                t_smooth = np.linspace(t_fit.min(), t_fit.max(), 1000)
                y_smooth = multi_exp_model(t_smooth, *popt)
                ax.plot(t_smooth + fit_range[0], y_smooth, color=fit_color, lw=fit_lw, label='Total Fit')
                
                if n_comp > 1:
                    for i in range(n_comp):
                        y_comp = popt[0] + popt[2*i+1] * np.exp(-t_smooth / popt[2*i+2])
                        ax.plot(t_smooth + fit_range[0], y_comp, '--', lw=1, label=f'τ_{i+1}={popt[2*i+2]:.2f}')

            ax.set_xlabel("Time (μs)", fontsize=20)
            ax.set_ylabel("Intensity", fontsize=20)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            
            # 【機能追加】目盛り線の制御
            if show_grid:
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
            else:
                ax.grid(False)
            
            if is_log:
                ax.set_yscale('log')
                ax.set_ylim(bottom=max(df['Intensity'].min(), 1e-6))
            
            if show_legend:
                ax.legend(frameon=False, loc='upper right')
            
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("📊 グラフを画像で保存", buf.getvalue(), f"fit_n{n_comp}.png", "image/png")

else:
    st.info("👈 サイドバーからCSV/テキストファイルを読み込んでください。")