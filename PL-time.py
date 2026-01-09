import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import io

# ---------------------------------------------------------
# 1. ページ設定と基本スタイル
# ---------------------------------------------------------
st.set_page_config(page_title="Lifetime Fitting Pro", layout="wide")
st.title("📉 Multi-Component Lifetime Fitting")

# ---------------------------------------------------------
# 2. 関数定義
# ---------------------------------------------------------
def load_smart_csv(uploaded_file):
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
        df.columns = [f"Col_{i}" for i in range(df.shape[1])]
        return df
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return None

def multi_exp_model(t, b, *params):
    y = b
    for i in range(0, len(params), 2):
        A = params[i]
        tau = params[i+1]
        y += A * np.exp(-t / tau)
    return y

# ---------------------------------------------------------
# 3. サイドバー設定
# ---------------------------------------------------------
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("寿命測定データをアップロード", type=["csv", "txt", "dat"])

st.sidebar.header("2. Global Style")
with st.sidebar.expander("文字・グラフ設定"):
    font_family = st.selectbox("Font Family", ["sans-serif", "serif", "monospace"])
    base_size = st.slider("基本文字サイズ", 8, 30, 14)
    label_size = st.slider("軸ラベルサイズ", 8, 40, 18)
    line_width = st.slider("線の太さ (Fit)", 1.0, 5.0, 2.0)
    # 凡例のオンオフ設定
    show_legend = st.checkbox("凡例を表示する", value=True)

# ---------------------------------------------------------
# 4. メイン処理
# ---------------------------------------------------------
if uploaded_file:
    df = load_smart_csv(uploaded_file)
    
    if df is not None:
        # 列選択
        st.sidebar.header("3. Column Selection")
        col_options = df.columns.tolist()
        x_col = st.sidebar.selectbox("Time 軸 (X)", col_options, index=0)
        y_col = st.sidebar.selectbox("Intensity 軸 (Y)", col_options, index=1)
        
        # フィッティング設定
        st.sidebar.header("4. Fitting Settings")
        n_comp = st.sidebar.selectbox("成分数 (n)", [1, 2, 3], index=0)
        
        _, main_col, _ = st.columns([0.05, 0.9, 0.05])
        
        with main_col:
            t_min, t_max = float(df[x_col].min()), float(df[x_col].max())
            idx_peak = df[y_col].idxmax()
            t_peak = float(df.loc[idx_peak, x_col])
            
            fit_range = st.slider("Fitting Range (μs)", t_min, t_max, (t_peak, t_max))
            
            mask = (df[x_col] >= fit_range[0]) & (df[x_col] <= fit_range[1])
            df_fit = df[mask].copy()
            
            # フィッティング実行
            b_init = df[y_col].min()
            p0 = [b_init]
            bounds_l, bounds_u = [0], [np.inf]
            for i in range(n_comp):
                p0.extend([df_fit[y_col].max() / n_comp, (fit_range[1]-fit_range[0])/ (i+2)])
                bounds_l.extend([0, 1e-9])
                bounds_u.extend([np.inf, np.inf])

            fit_success = False
            try:
                popt, pcov = curve_fit(multi_exp_model, df_fit[x_col], df_fit[y_col], p0=p0, bounds=(bounds_l, bounds_u))
                fit_success = True
            except:
                st.error("フィッティングに失敗しました。")

            # --- グラフ描画 ---
            plt.rcParams['font.family'] = font_family
            plt.rcParams['font.size'] = base_size
            
            fig, ax = plt.subplots(figsize=(10, 6))
            is_log = st.checkbox("Y軸をログスケールにする", value=True)
            
            ax.scatter(df[x_col], df[y_col], s=5, color='gray', alpha=0.3, label='Raw Data')
            
            if fit_success:
                t_plot = np.linspace(fit_range[0], fit_range[1], 500)
                y_plot = multi_exp_model(t_plot, *popt)
                ax.plot(t_plot, y_plot, color='red', lw=line_width, label=f'Total Fit (n={n_comp})')
                
                if n_comp > 1:
                    colors = ['blue', 'green', 'orange']
                    for i in range(n_comp):
                        A_i = popt[2*i+1]
                        tau_i = popt[2*i+2]
                        y_comp = popt[0] + A_i * np.exp(-t_plot / tau_i)
                        ax.plot(t_plot, y_comp, '--', lw=1, color=colors[i%3], label=f'Comp {i+1} (τ={tau_i:.3f})')
            
            ax.set_xlabel("Time (μs)", fontsize=label_size)
            ax.set_ylabel("Intensity", fontsize=label_size)
            if is_log: ax.set_yscale('log')
            
            # 凡例のオンオフ制御
            if show_legend:
                ax.legend(frameon=False)
            
            st.pyplot(fig)

            # --- 画像ダウンロード機能 ---
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button(
                label="画像を保存 (PNG)",
                data=buf.getvalue(),
                file_name=f"lifetime_fit_n{n_comp}.png",
                mime="image/png"
            )

            # --- 結果表示 ---
            if fit_success:
                st.subheader("Fitting Results")
                cols = st.columns(n_comp + 1)
                cols[0].metric("Baseline (b)", f"{popt[0]:.4e}")
                for i in range(n_comp):
                    cols[i+1].metric(f"Component {i+1} (τ)", f"{popt[2*i+2]:.4f} μs")
                
                res_df = pd.DataFrame({
                    "Parameter": ["Baseline"] + [f"Amp {i+1}" for i in range(n_comp)] + [f"Tau {i+1}" for i in range(n_comp)],
                    "Value": [f"{popt[0]:.4e}"] + [f"{popt[2*i+1]:.4e}" for i in range(n_comp)] + [f"{popt[2*i+2]:.4e}" for i in range(n_comp)]
                })
                st.table(res_df)

else:
    st.info("👈 サイドバーからデータをアップロードしてください。")

# ---------------------------------------------------------
# 5. 説明（最下部）
# ---------------------------------------------------------
st.divider()
st.subheader("📖 使い方")
st.markdown("""
1. **画像のダウンロード**: グラフの下にある「画像を保存 (PNG)」ボタンを押すと、300DPIの高解像度画像が保存されます。
2. **凡例の表示切り替え**: サイドバーの「文字・グラフ設定」内にあるチェックボックスで、凡例の表示/非表示を切り替えられます。
3. **対数表示**: グラフ上のチェックボックスでY軸のログスケールを切り替え可能です。
""")