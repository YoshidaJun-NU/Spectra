import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ページ設定
st.set_page_config(page_title="Multi-Exp Lifetime Fitting", layout="wide")

st.title("📉 Multi-Component Lifetime Fitting")
st.markdown("発光寿命測定データに対し、複数の指数関数の和でフィッティングを行います。")

# --- サイドバー: データ読み込み ---
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

# --- 関数定義: 多成分指数関数モデル ---
def create_multiexp_model(n, b_fixed):
    def model(t, *params):
        t_arr = np.array(t)
        y = np.full(t_arr.shape, b_fixed, dtype=np.float64)
        for i in range(n):
            A = params[2*i]
            tau = params[2*i+1]
            if abs(tau) < 1e-9:
                term = np.zeros_like(t_arr)
            else:
                term = A * np.exp(-t_arr / tau)
            y += term
        return y
    return model

# --- メイン処理 ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, skiprows=1, header=None)
        
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].copy()
            df.columns = ['Time', 'Intensity']
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
            df.dropna(inplace=True)
        else:
            st.error("データ列が不足しています。")
            st.stop()

        col_graph, col_ctrl = st.columns([2, 1])

        with col_ctrl:
            st.subheader("Fitting Parameters")
            n_components = st.selectbox("Number of Components (n)", options=[1, 2, 3, 4, 5], index=0)

            lowest_5_percent = df['Intensity'].nsmallest(int(len(df) * 0.05))
            default_b = float(lowest_5_percent.mean())

            st.markdown("#### 1. Baseline ($b$)")
            b_value = st.number_input("Baseline Value (Volt)", value=default_b, format="%.6e")

            st.markdown("#### 2. Time Range")
            idx_max = df['Intensity'].idxmax()
            t_at_max = df.loc[idx_max, 'Time']
            t_end = df['Time'].max()
            t_min_file = df['Time'].min()

            fit_range = st.slider(
                "Fitting Range (μs)",
                min_value=float(t_min_file),
                max_value=float(t_end),
                value=(float(t_at_max), float(t_end)),
                step=0.01
            )
            t_start_fit, t_end_fit = fit_range

            mask = (df['Time'] >= t_start_fit) & (df['Time'] <= t_end_fit)
            df_fit = df[mask].copy()

            if len(df_fit) == 0:
                st.warning("選択された範囲にデータがありません。")
                st.stop()

            y_max_range = df_fit['Intensity'].max() - b_value
            time_span = t_end_fit - t_start_fit
            if time_span <= 0: time_span = 1.0

            p0, bounds_min, bounds_max = [], [], []
            for i in range(n_components):
                p0.append(y_max_range / n_components)
                p0.append(time_span / (2 * (5 ** i)))
                bounds_min.extend([0, 1e-6]) 
                bounds_max.extend([np.inf, np.inf])

            fit_func = create_multiexp_model(n_components, b_value)

            try:
                popt, pcov = curve_fit(
                    fit_func, df_fit['Time'].values, df_fit['Intensity'].values, 
                    p0=p0, bounds=(bounds_min, bounds_max), maxfev=10000
                )
                
                st.markdown("### Results")
                st.latex(r"I(t) = \sum_{i=1}^{" + str(n_components) + r"} A_i e^{-t/\tau_i} + b")

                residuals = df_fit['Intensity'].values - fit_func(df_fit['Time'].values, *popt)
                r_squared = 1 - (np.sum(residuals**2) / np.sum((df_fit['Intensity'].values - df_fit['Intensity'].mean())**2))
                
                st.write(f"**$R^2$**: {r_squared:.5f}")
                st.write(f"**Fixed $b$**: {b_value:.4e}")

                res_data = []
                for i in range(n_components):
                    res_data.append({
                        "Component": f"Comp {i+1}",
                        "Tau (μs)": f"{popt[2*i+1]:.4f}",
                        "Amplitude (A)": f"{popt[2*i]:.4e}"
                    })
                st.table(pd.DataFrame(res_data))

                t_smooth = np.linspace(t_start_fit, t_end_fit, 1000)
                y_smooth = fit_func(t_smooth, *popt)

            except Exception as e:
                st.error(f"Fitting Failed: {e}")
                y_smooth = None

        # ---------------------------------------------------------
        # 3. グラフ描画 (Matplotlib版)
        # ---------------------------------------------------------
        with col_graph:
            is_log = st.checkbox("Log Scale Y-axis", value=False)
            
            # Figureの作成
            fig, ax = plt.subplots(figsize=(8, 6))

            # Raw Data
            ax.plot(df['Time'], df['Intensity'], color='lightgray', label='Raw Data', linewidth=1, alpha=0.7)

            # Selected Data
            ax.scatter(df_fit['Time'], df_fit['Intensity'], color='blue', s=2, alpha=0.3, label='Fitting Region')

            # Fit Curve
            if y_smooth is not None:
                ax.plot(t_smooth, y_smooth, color='red', linewidth=2, label=f'Fit (n={n_components})')
                
                # 各成分の表示
                if n_components > 1:
                    for i in range(n_components):
                        y_comp = popt[2*i] * np.exp(-t_smooth / popt[2*i+1]) + b_value
                        ax.plot(t_smooth, y_comp, linestyle='--', linewidth=1, label=f'Comp {i+1} (τ={popt[2*i+1]:.2f})')

            # グラフの装飾
            ax.set_title(f"Decay Fit (n={n_components})")
            ax.set_xlabel("Time (μs)")
            ax.set_ylabel("Intensity (Volt)")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, which="both", ls="-", alpha=0.2)

            if is_log:
                ax.set_yscale('log')
                # ログスケール時の表示範囲調整（0以下があるとエラーになるため）
                ymin = max(df['Intensity'].min(), 1e-6)
                ax.set_ylim(bottom=ymin)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Please upload a CSV file.")