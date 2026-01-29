import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors

# ---------------------------------------------------------
# 関数: データ読み込み (強化版)
# ---------------------------------------------------------
def load_spectral_data(uploaded_file, params):
    """
    ファイルを読み込み、指定された列マッピングに基づいて辞書を返す。
    エラー発生時の自動リカバリ機能付き。
    """
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        # --- 1. JASCO形式 (XYDATA) の自動検出 ---
        # (params['force_custom'] が True の場合はスキップして設定通りの読み込みを優先させることも可能だが
        #  基本的にはJASCO形式は自動判定で読む方が安全)
        jasco_skip = 0
        is_jasco = False
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                jasco_skip = i + 1
                is_jasco = True
                break
        
        df = None
        
        # --- 2. JASCO形式読み込み ---
        if is_jasco:
            try:
                df = pd.read_csv(io.StringIO(content), skiprows=jasco_skip, sep='\t', header=None, engine='python')
                if df.shape[1] < 2:
                     df = pd.read_csv(io.StringIO(content), skiprows=jasco_skip, sep='\s+', header=None, engine='python')
            except:
                pass 
        
        # --- 3. 汎用読み込み ---
        if df is None:
            sep_char = params['sep']
            sep_arg = None if sep_char == 'auto' else sep_char
            comment_arg = params['comment']
            skip_rows = params['skip_rows']

            # コメント文字自動設定
            if not comment_arg and lines and lines[0].strip().startswith('#'):
                comment_arg = '#'
            
            try:
                # 指定設定で読み込み
                df = pd.read_csv(
                    io.StringIO(content), 
                    skiprows=skip_rows, 
                    sep=sep_arg, 
                    comment=comment_arg, 
                    header=None, 
                    engine='python'
                )
            except Exception:
                # 失敗時、区切り文字自動判定(None)でリトライ、もしくはスペース区切りでリトライ
                if sep_arg is None:
                    try:
                        df = pd.read_csv(
                            io.StringIO(content), 
                            skiprows=skip_rows, 
                            sep='\s+', 
                            comment=comment_arg, 
                            header=None, 
                            engine='python'
                        )
                    except Exception as e:
                        return None, f"読み込み失敗(Retry): {e}"
                else:
                    return None, f"読み込み失敗: 設定を確認してください"

        # 数値変換
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.empty:
            return None, "有効なデータ行がありません (ヘッダー行数やコメント文字を確認してください)"

        # --- 4. 列データの抽出 (マッピング適用) ---
        def get_col_data(df, idx):
            if 0 <= idx < df.shape[1]:
                return df.iloc[:, idx].values
            return np.zeros(len(df))

        col_x_idx = params['cols']['x']
        col_ir_idx = params['cols']['ir']
        col_vcd_idx = params['cols']['vcd']
        col_noise_idx = params['cols']['noise']

        if col_x_idx >= df.shape[1]:
             return None, f"指定されたX列({col_x_idx+1}列目)がデータ内に存在しません (全{df.shape[1]}列)"

        x = get_col_data(df, col_x_idx)
        col_ir = get_col_data(df, col_ir_idx)
        col_vcd = get_col_data(df, col_vcd_idx)
        col_noise = get_col_data(df, col_noise_idx)

        # 先頭5行 (確認用DF作成)
        head_df = pd.DataFrame()
        head_df[f'Col{col_x_idx+1}(X)'] = x[:5]
        if col_ir_idx < df.shape[1]: head_df[f'Col{col_ir_idx+1}(IR)'] = col_ir[:5]
        if col_vcd_idx < df.shape[1]: head_df[f'Col{col_vcd_idx+1}(VCD)'] = col_vcd[:5]
        
        return {
            'filename': uploaded_file.name,
            'x': x,
            'ir': col_ir,  
            'vcd': col_vcd,
            'noise': col_noise,
            'head': head_df
        }, None

    except Exception as e:
        return None, f"読み込み例外: {e}"

# ---------------------------------------------------------
# 関数: データ結合
# ---------------------------------------------------------
def merge_vcd_ir_data(vcd_source, ir_source, new_filename):
    x_master = vcd_source['x']
    
    if np.all(vcd_source['vcd'] == 0) and not np.all(vcd_source['ir'] == 0):
        vcd_vals = vcd_source['ir']
    else:
        vcd_vals = vcd_source['vcd']

    ir_x = ir_source['x']
    ir_vals_raw = ir_source['ir']
    
    if len(x_master) > 1 and x_master[0] > x_master[-1]: 
        new_ir = np.interp(x_master, ir_x[::-1], ir_vals_raw[::-1])
    else:
        new_ir = np.interp(x_master, ir_x, ir_vals_raw)

    head_df = pd.DataFrame({'X': x_master[:5], 'Combined_IR': new_ir[:5], 'Combined_VCD': vcd_vals[:5]})

    return {
        'filename': new_filename,
        'x': x_master,
        'ir': new_ir,
        'vcd': vcd_vals,
        'noise': np.zeros_like(x_master),
        'head': head_df
    }

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="VCD/LD Analyzer", layout="wide")
    st.title("VCD / LD Spectra Analyzer")

    if 'vcd_data' not in st.session_state: st.session_state['vcd_data'] = []
    if 'ld_data' not in st.session_state: st.session_state['ld_data'] = []
    if 'calc_data' not in st.session_state: st.session_state['calc_data'] = []

    # ==========================================
    # 1. サイドバー: データ読み込み設定 (分離)
    # ==========================================
    st.sidebar.header("📂 ファイル読み込み")

    sep_map = {"自動 (Space/Tab)": "auto", "カンマ (,)": ",", "タブ (\\t)": "\t"}

    # --- 実験データ用設定 ---
    with st.sidebar.expander("⚙️ 実験データ読み込み設定 (非JASCO)", expanded=False):
        st.caption("※VCD/LD実験データに適用されます")
        c_e1, c_e2 = st.columns(2)
        exp_skip = c_e1.number_input("Skip Rows", value=0, min_value=0, key="exp_skip")
        exp_sep_mode = c_e2.selectbox("Separator", list(sep_map.keys()), key="exp_sep")
        exp_comment = st.text_input("Comment Char", value="", key="exp_comment")
        
        st.markdown("**列番号 (1始まり)**")
        ce_c1, ce_c2 = st.columns(2)
        exp_col_x = ce_c1.number_input("X (波数)", value=1, min_value=1, key="exp_cx")
        exp_col_ir = ce_c2.number_input("IR/Abs (2)", value=2, min_value=1, key="exp_ci")
        exp_col_vcd = ce_c1.number_input("VCD/Sig (3)", value=3, min_value=1, key="exp_cv")
        exp_col_noise = ce_c2.number_input("Noise (4)", value=4, min_value=1, key="exp_cn")

    params_exp = {
        "skip_rows": exp_skip,
        "sep": sep_map[exp_sep_mode],
        "comment": exp_comment if exp_comment else None,
        "cols": {"x": exp_col_x-1, "ir": exp_col_ir-1, "vcd": exp_col_vcd-1, "noise": exp_col_noise-1}
    }

    # --- 計算データ用設定 ---
    with st.sidebar.expander("⚙️ 計算データ読み込み設定", expanded=False):
        st.caption("※計算データ(Calc)に適用されます")
        c_c1, c_c2 = st.columns(2)
        calc_skip = c_c1.number_input("Skip Rows", value=0, min_value=0, key="calc_skip")
        calc_sep_mode = c_c2.selectbox("Separator", list(sep_map.keys()), key="calc_sep")
        calc_comment = st.text_input("Comment Char", value="#", key="calc_comment") # デフォルトで#を入れる
        
        st.markdown("**列番号 (1始まり)**")
        cc_c1, cc_c2 = st.columns(2)
        calc_col_x = cc_c1.number_input("X (波数)", value=1, min_value=1, key="calc_cx")
        calc_col_ir = cc_c2.number_input("IR (2)", value=2, min_value=1, key="calc_ci")
        calc_col_vcd = cc_c1.number_input("VCD (3)", value=3, min_value=1, key="calc_cv")
        # 計算データにはノイズ列がないことが多いのでデフォルト4のまま放置

    params_calc = {
        "skip_rows": calc_skip,
        "sep": sep_map[calc_sep_mode],
        "comment": calc_comment if calc_comment else None,
        "cols": {"x": calc_col_x-1, "ir": calc_col_ir-1, "vcd": calc_col_vcd-1, "noise": 999} # noiseは無効な列にしておく
    }

    # --- アップローダー群 ---
    st.sidebar.subheader("1. 実験データ (Exp)")
    uploaded_vcd = st.sidebar.file_uploader("VCD/IR 実験ファイル", accept_multiple_files=True, key="up_vcd", type=['txt', 'csv', 'dat'])
    if uploaded_vcd:
        for f in uploaded_vcd:
            if not any(d['filename'] == f.name for d in st.session_state['vcd_data']):
                # 実験用設定を使用
                data, err = load_spectral_data(f, params_exp)
                if data: st.session_state['vcd_data'].append(data)
                else: st.sidebar.error(f"{f.name}: {err}")

    uploaded_ld = st.sidebar.file_uploader("LD 実験ファイル", accept_multiple_files=True, key="up_ld", type=['txt', 'csv', 'dat'])
    if uploaded_ld:
        for f in uploaded_ld:
            if not any(d['filename'] == f.name for d in st.session_state['ld_data']):
                # 実験用設定を使用
                data, err = load_spectral_data(f, params_exp)
                if data: st.session_state['ld_data'].append(data)
                else: st.sidebar.error(f"{f.name}: {err}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 計算データ (Calc)")
    uploaded_calc = st.sidebar.file_uploader("計算データ (.txt/.csv)", accept_multiple_files=True, key="up_calc")
    if uploaded_calc:
        for f in uploaded_calc:
            if not any(d['filename'] == f.name for d in st.session_state['calc_data']):
                # 計算用設定を使用
                data, err = load_spectral_data(f, params_calc)
                if data: 
                    st.session_state['calc_data'].append(data)
                    st.sidebar.success(f"Calc: {f.name} 読込")
                else: st.sidebar.error(f"{f.name}: {err}")

    # ==========================================
    # タブ構成
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 VCD: 個別解析", 
        "📈 VCD: 比較", 
        "📏 LD解析", 
        "🔬 実験 vs 計算"
    ])

    vcd_data = st.session_state['vcd_data']
    ld_data = st.session_state['ld_data']
    calc_data = st.session_state['calc_data']

    # --------------------------------------------------
    # Tab 1: VCD 個別
    # --------------------------------------------------
    with tab1:
        if not vcd_data:
            st.info("サイドバーから実験データ(VCD)を読み込んでください。")
        else:
            st.subheader("Experimental Data Analysis")
            col_sel, col_opt = st.columns([1, 2])
            with col_sel:
                f_names = [d['filename'] for d in vcd_data]
                sel_idx = st.selectbox("Select File", range(len(f_names)), format_func=lambda x: f_names[x], key="t1_sel")
                sel_d = vcd_data[sel_idx]
            
            with col_opt:
                show_peak = st.checkbox("Peak Picking", value=False)
                p_th = 0.05
                if show_peak:
                    p_th = st.slider("Threshold", 0.0, 1.0, 0.05)

            if sel_d:
                x, ir, vcd = sel_d['x'], sel_d['ir'], sel_d['vcd']
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=(f"VCD: {sel_d['filename']}", "IR / Absorbance"))
                
                fig.add_trace(go.Scatter(x=x, y=vcd, name="VCD", line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=ir, name="IR", line=dict(color='red')), row=2, col=1)
                
                if show_peak:
                    peaks, _ = find_peaks(ir, height=p_th, distance=10)
                    fig.add_trace(go.Scatter(x=x[peaks], y=ir[peaks], mode='markers', name='Peaks', marker=dict(color='black', size=8)), row=2, col=1)

                fig.update_layout(height=600, hovermode="x unified")
                fig.update_xaxes(autorange="reversed", row=2, col=1)
                fig.update_xaxes(autorange="reversed", row=1, col=1)
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Tab 2: VCD 比較
    # --------------------------------------------------
    with tab2:
        if not vcd_data:
            st.info("データがありません。")
        else:
            st.subheader("Multi-Spectra Comparison")
            render_matplotlib_comparison(vcd_data, "vcd", "VCD Intensity", "Absorbance")

    # --------------------------------------------------
    # Tab 3: LD 解析
    # --------------------------------------------------
    with tab3:
        if not ld_data:
            st.info("LDデータがありません。")
        else:
            st.subheader("LD Analysis")
            render_matplotlib_comparison(ld_data, "ld", "LD Signal", "Absorbance")

    # --------------------------------------------------
    # Tab 4: 実験 vs 計算
    # --------------------------------------------------
    with tab4:
        st.subheader("🔬 Experimental vs Computational Comparison")
        
        c_exp, c_calc = st.columns(2)
        
        with c_exp:
            st.markdown("##### 1. 実験データ (Experimental)")
            if not vcd_data:
                st.warning("実験データがありません")
                sel_exp_data = None
            else:
                exp_names = [d['filename'] for d in vcd_data]
                sel_exp_name = st.selectbox("ファイル選択", exp_names, key="tv_exp_sel")
                sel_exp_data = next(d for d in vcd_data if d['filename'] == sel_exp_name)

        with c_calc:
            st.markdown("##### 2. 計算データ (Computational)")
            if not calc_data:
                st.warning("計算データがありません")
                sel_calc_data = None
            else:
                calc_names = [d['filename'] for d in calc_data]
                sel_calc_name = st.selectbox("ファイル選択", calc_names, key="tv_calc_sel")
                sel_calc_data = next(d for d in calc_data if d['filename'] == sel_calc_name)

        st.markdown("---")

        if sel_exp_data and sel_calc_data:
            with st.expander("🎚️ シミュレーション・フィッティング設定", expanded=True):
                col_para1, col_para2, col_para3 = st.columns(3)
                
                with col_para1:
                    st.markdown("**X軸 (波数) 補正**")
                    scale_freq = st.number_input("Scaling Factor (freq * x)", value=0.980, step=0.001, format="%.4f")
                    shift_freq = st.number_input("Shift (freq + x)", value=0.0, step=1.0)
                
                with col_para2:
                    st.markdown("**Y軸 (強度) 倍率**")
                    scale_int_vcd = st.number_input("VCD Intensity Scale", value=1.0, step=0.1)
                    scale_int_ir = st.number_input("IR Intensity Scale", value=1.0, step=0.1)
                
                with col_para3:
                    st.markdown("**表示設定**")
                    use_dual_axis = st.checkbox("2軸プロット (Dual Y-Axis)", value=True, help="実験値と計算値の桁が違う場合に有効")
                    plot_range = st.slider("表示範囲 (cm-1)", 0, 4000, (800, 2000))

            # データ加工
            exp_x = sel_exp_data['x']
            exp_vcd = sel_exp_data['vcd']
            exp_ir = sel_exp_data['ir']
            
            raw_calc_x = sel_calc_data['x']
            calc_x = raw_calc_x * scale_freq + shift_freq
            calc_vcd = sel_calc_data['vcd'] * scale_int_vcd
            calc_ir = sel_calc_data['ir'] * scale_int_ir

            # Plotly
            fig_cmp = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.1,
                specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                subplot_titles=("VCD Comparison", "IR Comparison")
            )

            # --- VCD Plot ---
            fig_cmp.add_trace(
                go.Scatter(x=exp_x, y=exp_vcd, name=f"Exp: {sel_exp_data['filename']}", 
                           line=dict(color='blue', width=2)), 
                row=1, col=1, secondary_y=False
            )
            fig_cmp.add_trace(
                go.Scatter(x=calc_x, y=calc_vcd, name=f"Calc: {sel_calc_data['filename']}", 
                           line=dict(color='red', width=1.5, dash='dash')), 
                row=1, col=1, secondary_y=use_dual_axis
            )

            # --- IR Plot ---
            fig_cmp.add_trace(
                go.Scatter(x=exp_x, y=exp_ir, name="Exp IR", 
                           line=dict(color='darkblue', width=2), showlegend=False), 
                row=2, col=1, secondary_y=False
            )
            fig_cmp.add_trace(
                go.Scatter(x=calc_x, y=calc_ir, name="Calc IR", 
                           line=dict(color='darkred', width=1.5, dash='dash'), showlegend=False), 
                row=2, col=1, secondary_y=use_dual_axis
            )

            fig_cmp.update_layout(height=700, hovermode="x unified")
            
            # X軸範囲設定 (降順)
            fig_cmp.update_xaxes(range=[plot_range[1], plot_range[0]], row=2, col=1, title_text="Wavenumber (cm⁻¹)")
            fig_cmp.update_xaxes(range=[plot_range[1], plot_range[0]], row=1, col=1)

            fig_cmp.update_yaxes(title_text="Exp Signal", secondary_y=False)
            if use_dual_axis:
                fig_cmp.update_yaxes(title_text="Calc Signal", secondary_y=True, showgrid=False)

            st.plotly_chart(fig_cmp, use_container_width=True)

# ---------------------------------------------------------
# Matplotlib 比較描画
# ---------------------------------------------------------
def render_matplotlib_comparison(data_source, prefix, label_y1, label_y2):
    c_sel, c_st = st.columns([1, 2])
    with c_sel:
        all_f = [d['filename'] for d in data_source]
        sel_f = st.multiselect("Select Files", all_f, default=all_f, key=f"{prefix}_ms")
        target = [d for d in data_source if d['filename'] in sel_f]
    
    if not target: return

    with c_st:
        with st.form(f"{prefix}_form"):
            st.write("Graph Settings")
            col1, col2 = st.columns(2)
            y_man = col1.checkbox("Manual Y-Range", key=f"{prefix}_yman")
            y1_lim = (col1.number_input("Y1 Min", value=-0.001, format="%.5f"), col1.number_input("Y1 Max", value=0.001, format="%.5f"))
            y2_lim = (col2.number_input("Y2 Min", value=0.0), col2.number_input("Y2 Max", value=1.5))
            x_lim = st.slider("X Range", 0, 4000, (800, 2000), key=f"{prefix}_xlim")
            submitted = st.form_submit_button("Update Plot")

    if submitted or target:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, d in enumerate(target):
            c = colors[i % len(colors)]
            ax1.plot(d['x'], d['vcd'] if prefix=='vcd' else d['vcd'], label=d['filename'], color=c, linewidth=1.2)
            ax2.plot(d['x'], d['ir'], color=c, linewidth=1.2)

        ax1.set_xlim(x_lim[1], x_lim[0])
        ax1.axhline(0, color='black', lw=0.5)
        ax1.set_ylabel(label_y1)
        ax1.legend(fontsize='small')
        
        ax2.set_xlabel("Wavenumber (cm-1)")
        ax2.set_ylabel(label_y2)
        
        if y_man:
            ax1.set_ylim(y1_lim[0], y1_lim[1])
            ax2.set_ylim(y2_lim[0], y2_lim[1])

        st.pyplot(fig)

if __name__ == "__main__":
    main()