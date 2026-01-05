import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ページ設定
st.set_page_config(page_title="Lifetime Fitting App", layout="wide")

st.title("📉 Luminescence Lifetime Fitting")
st.markdown("発光寿命測定データをアップロードし、指数関数減衰フィッティングを行います。")

# --- サイドバー: データ読み込み ---
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

# --- 関数定義: フィッティングモデル ---
def decay_model(t, I0, tau, b):
    """
    I(t) = I0 * exp(-t/tau) + b
    """
    return I0 * np.exp(-t / tau) + b

# --- メイン処理 ---
if uploaded_file is not None:
    try:
        # 1. データの読み込み (メタデータ行スキップ)
        # 1行目: ", 600" -> スキップ
        # 2行目以降: データ
        df = pd.read_csv(uploaded_file, skiprows=1, header=None)
        
        # 列名を設定 (ご指定の軸)
        if df.shape[1] >= 2:
            df = df.iloc[:, :2] # 最初の2列のみ使用
            df.columns = ['Time', 'Intensity']
        else:
            st.error("データ列が不足しています。")
            st.stop()

        # ---------------------------------------------------------
        # 2. パラメータ設定セクション (画面左側: サイドバーまたは列)
        # ---------------------------------------------------------
        
        # レイアウト: 左にグラフ、右に設定と結果
        col_graph, col_ctrl = st.columns([2, 1])

        with col_ctrl:
            st.subheader("Fitting Parameters")

            # --- ベースライン (b) の設定 ---
            # デフォルト値: 強度が最も低いデータの下位5%の平均値
            # これによりノイズの影響を抑えたベースライン推定を行います
            lowest_5_percent = df['Intensity'].nsmallest(int(len(df) * 0.05))
            default_b = float(lowest_5_percent.mean())

            st.markdown("#### 1. Baseline ($b$)")
            b_value = st.number_input(
                "Baseline Value (Volt)", 
                value=default_b, 
                format="%.6e",
                help="I(t) = I0 * exp(-t/tau) + b の bの値。デフォルトは最小値周辺の平均です。"
            )

            # --- フィッティング範囲の設定 ---
            st.markdown("#### 2. Time Range")
            
            # デフォルトの開始位置: 強度が最大の点（ピーク）から
            # デフォルトの終了位置: データの最後
            idx_max = df['Intensity'].idxmax()
            t_at_max = df.loc[idx_max, 'Time']
            t_end = df['Time'].max()
            t_min_file = df['Time'].min()

            fit_range = st.slider(
                "Fitting Range (μs)",
                min_value=float(t_min_file),
                max_value=float(t_end),
                value=(float(t_at_max), float(t_end)), # デフォルト範囲
                step=0.01
            )
            
            t_start_fit, t_end_fit = fit_range

            # --- フィッティング実行 ---
            # 選択範囲のデータを抽出
            mask = (df['Time'] >= t_start_fit) & (df['Time'] <= t_end_fit)
            df_fit = df[mask].copy()

            # データオフセットの補正 (計算安定化のため)
            # t=0 をフィッティング開始点とみなすよう一時的にシフトする場合もありますが、
            # ここでは物理的な時間軸(t)をそのまま使い、I0がその時刻での強度となるよう計算します。
            
            # 初期値の推定 (p0)
            # I0_guess: 範囲内の最大強度 - ベースライン
            I0_guess = df_fit['Intensity'].max() - b_value
            tau_guess = 1.0 # 仮の初期値
            
            # bを固定するか、最適化パラメータに含めるか
            # ご要望は「bの値も入力できるように」かつ「式は I0*exp(-t/tau)+b」
            # ここではユーザー入力を「固定値」として扱い、I0とtauだけを探させます。
            # (bも変数にすると、テール部分のノイズでtauが大きく変動しやすいため、入力値を信頼する設計にします)
            
            def fit_func_fixed_b(t, I0, tau):
                return decay_model(t, I0, tau, b_value)

            try:
                popt, pcov = curve_fit(
                    fit_func_fixed_b, 
                    df_fit['Time'], 
                    df_fit['Intensity'], 
                    p0=[I0_guess, tau_guess],
                    maxfev=5000
                )
                
                calc_I0, calc_tau = popt
                
                # 結果表示
                st.markdown("### Results")
                st.latex(r"I(t) = I_0 \cdot e^{-t/\tau} + b")
                
                st.success(f"**Lifetime ($\\tau$): {calc_tau:.4f} $\\mu$s**")
                st.write(f"**$I_0$**: {calc_I0:.4e}")
                st.write(f"**$b$ (Fixed)**: {b_value:.4e}")
                
                # R2乗値の計算 (当てはまりの良さ)
                residuals = df_fit['Intensity'] - fit_func_fixed_b(df_fit['Time'], *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((df_fit['Intensity'] - df_fit['Intensity'].mean())**2)
                r_squared = 1 - (ss_res / ss_tot)
                st.write(f"**$R^2$**: {r_squared:.4f}")

                # フィッティングカーブの生成 (描画用)
                # 滑らかに見せるため、範囲内を細かく分割
                t_smooth = np.linspace(t_start_fit, t_end_fit, 500)
                y_smooth = fit_func_fixed_b(t_smooth, *popt)

            except Exception as e:
                st.error(f"フィッティングに失敗しました: {e}")
                calc_tau = None

        # ---------------------------------------------------------
        # 3. グラフ描画 (画面右側 -> 左側へ配置)
        # ---------------------------------------------------------
        with col_graph:
            fig = go.Figure()

            # 生データ (全範囲)
            fig.add_trace(go.Scatter(
                x=df['Time'], 
                y=df['Intensity'],
                mode='lines',
                name='Raw Data',
                line=dict(color='lightgray', width=1.5),
                opacity=0.7
            ))

            # フィッティング対象データ（選択範囲）
            fig.add_trace(go.Scatter(
                x=df_fit['Time'], 
                y=df_fit['Intensity'],
                mode='markers',
                name='Selected Data',
                marker=dict(color='blue', size=2)
            ))

            # フィッティング曲線
            if 'calc_tau' in locals() and calc_tau is not None:
                fig.add_trace(go.Scatter(
                    x=t_smooth, 
                    y=y_smooth,
                    mode='lines',
                    name=f'Fit (τ={calc_tau:.2f}μs)',
                    line=dict(color='red', width=2)
                ))

            # グラフのレイアウト
            fig.update_layout(
                title=f"Decay Profile: {uploaded_file.name}",
                xaxis_title="Time (μs)",
                yaxis_title="Intensity (Volt)",
                template="plotly_white",
                height=600,
                legend=dict(x=0.7, y=0.9)
            )
            
            # y軸を対数表示にするオプション
            log_scale = st.checkbox("Log Scale (Y-axis)", value=False)
            if log_scale:
                fig.update_yaxes(type="log")

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"ファイルの処理中にエラーが発生しました: {e}")

else:
    st.info("👈 CSVファイルをアップロードしてください。")