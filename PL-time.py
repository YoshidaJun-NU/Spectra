import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ページ設定
st.set_page_config(page_title="Multi-Exp Lifetime Fitting", layout="wide")

st.title("📉 Multi-Component Lifetime Fitting")
st.markdown("発光寿命測定データに対し、複数の指数関数の和でフィッティングを行います。")

# --- サイドバー: データ読み込み ---
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

# --- 関数定義: 多成分指数関数モデル (修正版) ---
def create_multiexp_model(n, b_fixed):
    """
    n成分の指数関数モデルを生成するクロージャ
    I(t) = sum(Ai * exp(-t/tau_i)) + b
    """
    def model(t, *params):
        # エラー回避: Pandas Seriesなどが来ても強制的にNumPy配列にする
        t_arr = np.array(t)
        
        # ベースラインで初期化 (サイズをt_arrに合わせる)
        y = np.full(t_arr.shape, b_fixed, dtype=np.float64)
        
        for i in range(n):
            A = params[2*i]
            tau = params[2*i+1]
            
            # ゼロ除算回避: tauが極端に小さい場合はその項を0とみなすなど安全策をとる
            # 通常のカーブフィッティングではboundsを設定するため0にはならないはずだが念のため
            if abs(tau) < 1e-9:
                # tau ~ 0 の場合、exp(-t/tau) は一瞬で0になるため寄与なしとする
                term = np.zeros_like(t_arr)
            else:
                term = A * np.exp(-t_arr / tau)
            
            y += term
        return y
    return model

# --- メイン処理 ---
if uploaded_file is not None:
    try:
        # 1. データの読み込み
        df = pd.read_csv(uploaded_file, skiprows=1, header=None)
        
        if df.shape[1] >= 2:
            # 必要な列だけ抽出し、列名を付与
            df = df.iloc[:, :2].copy()
            df.columns = ['Time', 'Intensity']
            
            # 計算用に数値型であることを保証
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
            df.dropna(inplace=True) # 数値変換できなかった行を削除
        else:
            st.error("データ列が不足しています。")
            st.stop()

        # ---------------------------------------------------------
        # 2. パラメータ設定
        # ---------------------------------------------------------
        col_graph, col_ctrl = st.columns([2, 1])

        with col_ctrl:
            st.subheader("Fitting Parameters")

            # --- 成分数 n ---
            n_components = st.selectbox(
                "Number of Components (n)", 
                options=[1, 2, 3, 4, 5], 
                index=0
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

            # データが空でないかチェック
            if len(df_fit) == 0:
                st.warning("選択された範囲にデータがありません。")
                st.stop()

            # 初期値 (p0) と境界 (bounds)
            y_max_range = df_fit['Intensity'].max() - b_value
            time_span = t_end_fit - t_start_fit
            if time_span <= 0: time_span = 1.0

            p0 = []
            bounds_min = []
            bounds_max = []

            for i in range(n_components):
                # 初期値
                p0.append(y_max_range / n_components) # A
                
                factor = 2 * (5 ** i) 
                guess_tau = time_span / factor
                p0.append(guess_tau) # tau

                # 境界 (A >= 0, tau > 1e-9)
                # tauの下限を0より少し大きくしてゼロ除算を絶対防ぐ
                bounds_min.extend([0, 1e-6]) 
                bounds_max.extend([np.inf, np.inf])

            fit_func = create_multiexp_model(n_components, b_value)

            try:
                # curve_fit実行
                # xデータ, yデータともに .values を使って明示的にNumPy配列を渡す
                popt, pcov = curve_fit(
                    fit_func, 
                    df_fit['Time'].values, 
                    df_fit['Intensity'].values, 
                    p0=p0,
                    bounds=(bounds_min, bounds_max),
                    maxfev=10000
                )
                
                # --- 結果表示 ---
                st.markdown("### Results")
                latex_str = r"I(t) = \sum_{i=1}^{" + str(n_components) + r"} A_i e^{-t/\tau_i} + b"
                st.latex(latex_str)

                # R2乗値
                residuals = df_fit['Intensity'].values - fit_func(df_fit['Time'].values, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((df_fit['Intensity'].values - df_fit['Intensity'].mean())**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                st.write(f"**$R^2$**: {r_squared:.5f}")
                st.write(f"**Fixed $b$**: {b_value:.4e}")

                # パラメータテーブル
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

                # プロット用データ
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
                
                # 各成分の表示
                if n_components > 1:
                    for i in range(n_components):
                        A_i = popt[2*i]
                        tau_i = popt[2*i+1]
                        # ベースラインを含めずに成分のみ描画するか、ベースラインに乗せるか
                        # ここでは成分の寄与を見るため b_value を足して表示
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
            
            is_log = st.checkbox("Log Scale Y-axis", value=False)
            if is_log:
                fig.update_yaxes(type="log")

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Please upload a CSV file.")