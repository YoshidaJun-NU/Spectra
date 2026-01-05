import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ページ設定
st.set_page_config(page_title="Multi-Exp Lifetime Fitting", layout="wide")

st.title("📉 Multi-Component Luminescence Lifetime Fitting")
st.markdown("発光寿命測定データに対し、複数の指数関数の和でフィッティングを行います。")

# --- サイドバー: データ読み込み ---
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

# --- 関数定義: 多成分指数関数モデル ---
def create_multiexp_model(n, b_fixed):
    """
    n成分の指数関数モデルを生成するクロージャ
    I(t) = sum(Ai * exp(-t/tau_i)) + b
    params: [A1, tau1, A2, tau2, ..., An, taun]
    """
    def model(t, *params):
        y = np.full_like(t, b_fixed, dtype=np.float64)
        for i in range(n):
            A = params[2*i]
            tau = params[2*i+1]
            # オーバーフロー対策
            safe_div = np.divide(-t, tau, out=np.zeros_like(t), where=tau!=0)
            y += A * np.exp(safe_div)
        return y
    return model

# --- メイン処理 ---
if uploaded_file is not None:
    try:
        # 1. データの読み込み
        df = pd.read_csv(uploaded_file, skiprows=1, header=None)
        
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['Time', 'Intensity']
        else:
            st.error("データ列が不足しています。")
            st.stop()

        # ---------------------------------------------------------
        # 2. パラメータ設定 (サイドバー & メイン)
        # ---------------------------------------------------------
        col_graph, col_ctrl = st.columns([2, 1])

        with col_ctrl:
            st.subheader("Fitting Parameters")

            # --- 成分数 n の選択 ---
            n_components = st.selectbox(
                "Number of Components (n)", 
                options=[1, 2, 3, 4, 5], 
                index=0,
                help="I(t) = Σ A_i * exp(-t/τ_i) + b の成分数"
            )

            # --- ベースライン (b) ---
            lowest_5_percent = df['Intensity'].nsmallest(int(len(df) * 0.05))
            default_b = float(lowest_5_percent.mean())

            st.markdown("#### 1. Baseline ($b$)")
            b_value = st.number_input(
                "Baseline Value (Volt)", 
                value=default_b, 
                format="%.6e"
            )

            # --- フィッティング範囲 ---
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

            # --- 解析実行 ---
            mask = (df['Time'] >= t_start_fit) & (df['Time'] <= t_end_fit)
            df_fit = df[mask].copy()

            # 初期値 (p0) と境界 (bounds) の作成
            # 振幅(A)の合計が最大強度付近になるように分割
            # 寿命(tau)は時間範囲内で対数的に分散させる (多成分解析の安定化のため)
            
            y_max_range = df_fit['Intensity'].max() - b_value
            time_span = t_end_fit - t_start_fit
            if time_span <= 0: time_span = 1.0

            p0 = []
            bounds_min = []
            bounds_max = []

            for i in range(n_components):
                # Aの初期値: 均等割り
                p0.append(y_max_range / n_components) 
                
                # tauの初期値: 成分が増えるごとに短くなるように分散
                # 例: n=2 -> tau1=span/2, tau2=span/10
                factor = 2 * (5 ** i) 
                guess_tau = time_span / factor
                p0.append(guess_tau)

                # 境界設定 (A > 0, tau > 0)
                bounds_min.extend([0, 0])
                bounds_max.extend([np.inf, np.inf])

            # フィッティング関数生成 (bは固定値としてクロージャに埋め込む)
            fit_func = create_multiexp_model(n_components, b_value)

            try:
                # curve_fit実行
                popt, pcov = curve_fit(
                    fit_func, 
                    df_fit['Time'], 
                    df_fit['Intensity'], 
                    p0=p0,
                    bounds=(bounds_min, bounds_max),
                    maxfev=10000
                )
                
                # --- 結果表示 ---
                st.markdown("### Results")
                
                # 数式の表示
                latex_str = r"I(t) = \sum_{i=1}^{" + str(n_components) + r"} A_i e^{-t/\tau_i} + b"
                st.latex(latex_str)

                # R2乗値
                residuals = df_fit['Intensity'] - fit_func(df_fit['Time'], *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((df_fit['Intensity'] - df_fit['Intensity'].mean())**2)
                r_squared = 1 - (ss_res / ss_tot)
                st.write(f"**$R^2$**: {r_squared:.5f}")
                st.write(f"**Fixed $b$**: {b_value:.4e}")

                # パラメータテーブル作成
                res_data = []
                for i in range(n_components):
                    A_i = popt[2*i]
                    tau_i = popt[2*i+1]
                    res_data.append({
                        "Component": f"Comp {i+1}",
                        "Tau (μs)": f"{tau_i:.4f}",
                        "Amplitude (A)": f"{A_i:.4e}"
                    })
                
                st.table(pd.DataFrame(res_data))

                # プロット用データ生成
                t_smooth = np.linspace(t_start_fit, t_end_fit, 1000)
                y_smooth = fit_func(t_smooth, *popt)

            except Exception as e:
                st.error(f"Fitting Failed: {e}")
                st.warning("ヒント: ベースラインを調整するか、範囲を変更してみてください。")
                y_smooth = None

        # ---------------------------------------------------------
        # 3. グラフ描画
        # ---------------------------------------------------------
        with col_graph:
            fig = go.Figure()

            # Raw Data
            fig.add_trace(go.Scatter(
                x=df['Time'], y=df['Intensity'],
                mode='lines', name='Raw Data',
                line=dict(color='lightgray', width=1)
            ))

            # Selected Data
            fig.add_trace(go.Scatter(
                x=df_fit['Time'], y=df_fit['Intensity'],
                mode='markers', name='Fitting Region',
                marker=dict(color='blue', size=2, opacity=0.5)
            ))

            # Fit Curve
            if 'y_smooth' in locals() and y_smooth is not None:
                fig.add_trace(go.Scatter(
                    x=t_smooth, y=y_smooth,
                    mode='lines', name=f'Fit (n={n_components})',
                    line=dict(color='red', width=2)
                ))
                
                # 各成分の分解表示 (n > 1の場合のみ)
                if n_components > 1:
                    for i in range(n_components):
                        A_i = popt[2*i]
                        tau_i = popt[2*i+1]
                        # 各成分単独の曲線 (ベースライン除く)
                        y_comp = A_i * np.exp(-t_smooth / tau_i) + b_value
                        fig.add_trace(go.Scatter(
                            x=t_smooth, y=y_comp,
                            mode='lines', 
                            name=f'Comp {i+1} (τ={tau_i:.2f})',
                            line=dict(dash='dash', width=1)
                        ))

            fig.update_layout(
                title=f"Decay Fit (n={n_components})",
                xaxis_title="Time (μs)",
                yaxis_title="Intensity (Volt)",
                height=600,
                legend=dict(x=0.65, y=0.95, bgcolor='rgba(255,255,255,0.8)')
            )
            
            # Log Scale Switch
            is_log = st.checkbox("Log Scale Y-axis", value=False)
            if is_log:
                fig.update_yaxes(type="log")

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Please upload a CSV file.")