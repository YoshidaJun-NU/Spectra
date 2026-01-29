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
# 関数: データ読み込み (JASCO形式 or 汎用CSV/TXT)
# ---------------------------------------------------------
def load_spectral_data(uploaded_file, params):
    """
    ファイルを読み込み、指定された列マッピングに基づいて辞書を返す。
    params: {
        'skip_rows': int, 
        'comment': str or None, 
        'sep': str, 
        'cols': {'x': int, 'ir': int, 'vcd': int, 'noise': int} (0-based index)
    }
    """
    try:
        # バイナリデータをテキストとしてデコード
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        # --- 1. JASCO形式 (XYDATA) の自動検出 ---
        jasco_skip = 0
        is_jasco = False
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                jasco_skip = i + 1
                is_jasco = True
                break
        
        df = None
        
        # --- 2. 読み込み処理 ---
        # JASCO形式と判定された場合は優先的にその仕様で読む
        if is_jasco:
            try:
                # JASCOはタブまたはスペース区切り
                df = pd.read_csv(io.StringIO(content), skiprows=jasco_skip, sep='\t', header=None, engine='python')
                if df.shape[1] < 2:
                     df = pd.read_csv(io.StringIO(content), skiprows=jasco_skip, sep='\s+', header=None, engine='python')
            except:
                pass # 失敗したら汎用読み込みへ
        
        # 汎用読み込み (JASCOでない、またはJASCO読み込み失敗時)
        if df is None:
            sep_char = params['sep']
            # 'auto'の場合は sep=None (Python engineで自動判定)
            sep_arg = None if sep_char == 'auto' else sep_char
            
            try:
                df = pd.read_csv(
                    io.StringIO(content), 
                    skiprows=params['skip_rows'], 
                    sep=sep_arg, 
                    comment=params['comment'], 
                    header=None, 
                    engine='python'
                )
            except Exception as e:
                return None, f"CSV読み込みエラー: {e}"

        # 数値変換
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.empty:
            return None, "有効なデータ行がありません (ヘッダー行数やコメント文字を確認してください)"

        # --- 3. 列データの抽出 (マッピング適用) ---
        # データフレームの列数チェック
        max_col_idx = max(params['cols'].values())
        if df.shape[1] <= max_col_idx:
            # 必須列(X, IR, VCD)が含まれているかチェック
            # 最低でもXとIR(またはVCD)が必要
            required_max = max(params['cols']['x'], params['cols']['ir'], params['cols']['vcd'])
            # ノイズ列などが指定されていて、実際のデータにない場合は許容し、0埋めする処理が必要
            # ここではシンプルに、「指定されたインデックスが範囲外なら0埋め」にする
        
        def get_col_data(df, idx):
            if 0 <= idx < df.shape[1]:
                return df.iloc[:, idx].values
            return np.zeros(len(df))

        x = get_col_data(df, params['cols']['x'])
        # X軸が全て0なら読み込み失敗の可能性が高い
        if np.all(x == 0) and df.shape[1] > 0:
             # マッピングミスの可能性: ユーザー指定列が範囲外の場合
             return None, f"指定されたX列({params['cols']['x']+1}列目)が見つかりません (データ列数: {df.shape[1]})"

        col_ir = get_col_data(df, params['cols']['ir'])
        col_vcd = get_col_data(df, params['cols']['vcd'])
        col_noise = get_col_data(df, params['cols']['noise'])

        # 先頭5行 (確認用)
        # 表示用に列名を付ける
        head_df = pd.DataFrame()
        head_df['X'] = x[:5]
        head_df['IR/Abs'] = col_ir[:5]
        head_df['VCD/Sig'] = col_vcd[:5]
        if params['cols']['noise'] < df.shape[1]:
            head_df['Noise'] = col_noise[:5]
        
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
# 関数: データ結合 (VCDファイル + IRファイル)
# ---------------------------------------------------------
def merge_vcd_ir_data(vcd_source, ir_source, new_filename):
    x_master = vcd_source['x']
    
    # VCDデータの取得 (VCDソースのVCD列を使用、なければIR列を使用)
    if np.all(vcd_source['vcd'] == 0) and not np.all(vcd_source['ir'] == 0):
        vcd_vals = vcd_source['ir']
    else:
        vcd_vals = vcd_source['vcd']

    # IRデータの取得と補間
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
# 関数: Gnuplot用パッケージ作成
# ---------------------------------------------------------
def create_gnuplot_package(data_list, style_dict, x_lim, y1_lim, y2_lim, y3_lim, 
                           label_y1="Signal", label_y2="Absorbance", label_y3="Noise", include_noise=False):
    if not data_list: return None
    
    all_x = []
    for d in data_list:
        all_x.extend(d['x'])
    common_x = np.sort(np.unique(all_x))[::-1] 
    
    df_out = pd.DataFrame({'Wavenumber': common_x})
    plot_cmds_y1 = []
    plot_cmds_y2 = []
    plot_cmds_y3 = []
    
    current_col = 2
    for i, d in enumerate(data_list):
        fname = d['filename']
        style = style_dict.get(fname, {'color': 'black', 'scale': 1.0})
        color = style['color']
        scale = style['scale']
        
        y2_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])          
        y1_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1]) * scale 
        y3_interp = np.interp(common_x, d['x'][::-1], d['noise'][::-1]) * scale 
        
        safe_name = f"File_{i+1}"
        df_out[f"{safe_name}_Abs"] = y2_interp
        df_out[f"{safe_name}_Sig"] = y1_interp
        df_out[f"{safe_name}_Nse"] = y3_interp
        
        title = fname.replace('_', '\\_')
        if scale != 1.0: title += f" (x{scale})"
        
        plot_cmds_y2.append(f"'data.dat' u 1:{current_col} w l lc rgb '{color}' title '{title}'")
        plot_cmds_y1.append(f"'data.dat' u 1:{current_col+1} w l lc rgb '{color}' notitle")
        if include_noise:
            plot_cmds_y3.append(f"'data.dat' u 1:{current_col+2} w l lc rgb '{color}' notitle")
        
        current_col += 3

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.6f')

    xr = f"[{x_lim[0]}:{x_lim[1]}]"
    yr_y1 = f"[{y1_lim[0]}:{y1_lim[1]}]" if y1_lim[0] is not None else "[:]"
    yr_y2 = f"[{y2_lim[0]}:{y2_lim[1]}]" if y2_lim[0] is not None else "[:]"
    yr_y3 = f"[{y3_lim[0]}:{y3_lim[1]}]" if y3_lim[0] is not None else "[:]"

    layout_rows = 3 if include_noise else 2
    height = 900 if include_noise else 800
    
    # Plot blocks
    p1 = f"""
set ylabel "{label_y1}"
set yrange {yr_y1}
set bmargin 0
set format x ""
set xzeroaxis lt 1 lc rgb "black" lw 1
plot {', '.join(plot_cmds_y1)}
"""
    p2 = f"""
set ylabel "{label_y2}"
set yrange {yr_y2}
set bmargin {0 if include_noise else 4}
set format x {"''" if include_noise else "'%g'"}
{'' if include_noise else 'set xlabel "Wavenumber (cm^{-1})"'}
plot {', '.join(plot_cmds_y2)}
"""
    p3 = ""
    if include_noise:
        p3 = f"""
set ylabel "{label_y3}"
set yrange {yr_y3}
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set format x "%g"
plot {', '.join(plot_cmds_y3)}
"""

    script = f"""
set terminal pngcairo size 800,{height} font "Arial,12"
set output 'plot.png'
set multiplot layout {layout_rows},1 margins 0.15, 0.95, 0.1, 0.95 spacing 0.05
set xrange {xr}
set grid ls 1 lc rgb "gray" lw 0.5 dt 2
set lmargin 12
set tmargin 0
{p1}
{p2}
{p3}
unset multiplot
    """
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("data.dat", data_str)
        zf.writestr("plot.plt", script)
    zip_buffer.seek(0)
    return zip_buffer

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="VCD/LD Analyzer", layout="wide")
    st.title("VCD / LD Spectra Analyzer")

    if 'vcd_data' not in st.session_state: st.session_state['vcd_data'] = []
    if 'ld_data' not in st.session_state: st.session_state['ld_data'] = []

    # ==========================================
    # 1. サイドバー: データ読み込み設定
    # ==========================================
    st.sidebar.header("📂 ファイル読み込み")

    # --- 読み込み詳細設定 ---
    with st.sidebar.expander("⚙️ 読み込み詳細設定 (非JASCO形式)", expanded=False):
        st.caption("JASCO(XYDATA)形式以外の場合に適用されます。")
        
        c_p1, c_p2 = st.columns(2)
        p_skip = c_p1.number_input("ヘッダー行数 (Skip)", value=0, min_value=0)
        p_sep_mode = c_p2.selectbox("区切り文字", ["自動 (Space/Tab)", "カンマ (,)", "タブ (\\t)"])
        p_comment = st.text_input("コメント文字 (例: #)", value="")
        
        st.markdown("**列番号の指定 (1始まり)**")
        c_col1, c_col2 = st.columns(2)
        col_x = c_col1.number_input("X (波数)", value=1, min_value=1)
        col_ir = c_col2.number_input("IR/Abs (2段目)", value=2, min_value=1)
        col_vcd = c_col1.number_input("VCD/Sig (1段目)", value=3, min_value=1)
        col_noise = c_col2.number_input("Noise (3段目)", value=4, min_value=1)

    # パラメータ辞書作成
    sep_map = {"自動 (Space/Tab)": "auto", "カンマ (,)": ",", "タブ (\\t)": "\t"}
    load_params = {
        "skip_rows": p_skip,
        "sep": sep_map[p_sep_mode],
        "comment": p_comment if p_comment else None,
        # 0-based indexに変換
        "cols": {"x": col_x-1, "ir": col_ir-1, "vcd": col_vcd-1, "noise": col_noise-1}
    }

    # --- アップローダー ---
    st.sidebar.subheader("VCD解析用 (Tab 1, 2)")
    uploaded_vcd = st.sidebar.file_uploader(
        "VCDファイルをアップロード", 
        accept_multiple_files=True,
        key="up_vcd",
        type=['txt', 'csv', 'dat'],
        help="波数, IR, VCD, (Noise) のデータ"
    )
    if uploaded_vcd:
        data_list = []
        for f in uploaded_vcd:
            data, error_msg = load_spectral_data(f, load_params)
            if data: data_list.append(data)
            else: st.sidebar.error(f"VCD Error {f.name}: {error_msg}")
        if data_list:
            current_files = {d['filename'] for d in st.session_state['vcd_data']}
            for d in data_list:
                if d['filename'] not in current_files:
                    st.session_state['vcd_data'].append(d)
            st.sidebar.success(f"VCD: {len(data_list)}件 読込完了")

    st.sidebar.markdown("---")

    st.sidebar.subheader("LD解析用 (Tab 3)")
    uploaded_ld = st.sidebar.file_uploader(
        "LDファイルをアップロード", 
        accept_multiple_files=True,
        key="up_ld",
        type=['txt', 'csv', 'dat'],
        help="波数, Abs, LD のデータ"
    )
    if uploaded_ld:
        data_list = []
        for f in uploaded_ld:
            # LDの場合、UI上の「VCD/Sig」設定をLD列として読む
            data, error_msg = load_spectral_data(f, load_params)
            if data: data_list.append(data)
            else: st.sidebar.error(f"LD Error {f.name}: {error_msg}")
        if data_list:
            current_files = {d['filename'] for d in st.session_state['ld_data']}
            for d in data_list:
                if d['filename'] not in current_files:
                    st.session_state['ld_data'].append(d)
            st.sidebar.success(f"LD: {len(data_list)}件 読込完了")
    
    # === ファイル結合ツール ===
    if st.session_state['vcd_data']:
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔗 データの結合 (VCD + IR)", expanded=False):
            st.caption("別々のファイルを結合して1つのデータセットにします。")
            
            loaded_files = st.session_state['vcd_data']
            filenames = [d['filename'] for d in loaded_files]
            
            f_vcd = st.selectbox("VCDデータとして使うファイル", filenames, key="sel_merge_vcd")
            f_ir = st.selectbox("IRデータとして使うファイル", filenames, key="sel_merge_ir")
            new_name = st.text_input("新しい結合ファイル名", value=f"Combined_{f_vcd}")
            
            if st.button("結合してリストに追加"):
                obj_vcd = next(d for d in loaded_files if d['filename'] == f_vcd)
                obj_ir = next(d for d in loaded_files if d['filename'] == f_ir)
                merged_data = merge_vcd_ir_data(obj_vcd, obj_ir, new_name)
                st.session_state['vcd_data'].append(merged_data)
                st.sidebar.success(f"結合完了: {new_name}")

    # === データ確認 (先頭5行) ===
    all_loaded = st.session_state['vcd_data'] + st.session_state['ld_data']
    if all_loaded:
        st.sidebar.markdown("---")
        with st.sidebar.expander("📄 読み込みデータの確認 (先頭5行)"):
            file_opts = [d['filename'] for d in all_loaded]
            sel_check = st.selectbox("確認するファイル", file_opts)
            for d in all_loaded:
                if d['filename'] == sel_check:
                    st.caption("※設定に基づいて読み込まれたデータ")
                    st.dataframe(d['head'])
                    break

    # ==========================================
    # タブ構成
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📊 VCD: 個別解析", "📈 VCD: 比較プロット", "📏 LD解析 (Linear Dichroism)"])

    vcd_data = st.session_state['vcd_data']
    ld_data = st.session_state['ld_data']

    # ==========================================
    # Tab 1: VCD 個別解析 (Interactive)
    # ==========================================
    with tab1:
        if not vcd_data:
            st.info("サイドバーからVCDファイルを読み込んでください。")
        else:
            st.subheader("VCD: Single Spectrum Analysis")
            col_sel, col_peak = st.columns([1, 2])
            
            with col_sel:
                file_names = [d['filename'] for d in vcd_data]
                selected_idx = st.selectbox("ファイル選択", range(len(file_names)), format_func=lambda x: file_names[x], key="vcd_sel")
                selected_data = vcd_data[selected_idx]
                
                with st.expander("軸範囲設定", expanded=False):
                    man_t1 = st.checkbox("手動設定", key="t1_man")
                    c1, c2 = st.columns(2)
                    t1_x_high = c1.number_input("X Left", value=2000.0, key="t1_xh")
                    t1_x_low = c2.number_input("X Right", value=800.0, key="t1_xl")
                    t1_vcd_min, t1_vcd_max = None, None
                    t1_ir_min, t1_ir_max = None, None
                    if man_t1:
                        t1_vcd_max = c1.number_input("VCD Max", value=0.001, format="%.5f", key="t1_vmax")
                        t1_vcd_min = c2.number_input("VCD Min", value=-0.001, format="%.5f", key="t1_vmin")
                        t1_ir_max = c1.number_input("IR Max", value=1.5, key="t1_imax")
                        t1_ir_min = c2.number_input("IR Min", value=0.0, key="t1_imin")

            with col_peak:
                do_peak = st.checkbox("ピーク検出", value=True, key="vcd_peak")
                peak_th = st.slider("しきい値", 0.0, 2.0, 0.05, 0.01, key="vcd_th")

            if selected_data:
                x, ir, vcd = selected_data['x'], selected_data['ir'], selected_data['vcd']
                peaks, _ = find_peaks(ir, height=peak_th, distance=10)
                peak_x = x[peaks]
                peak_ir = ir[peaks]
                peak_vcd = vcd[peaks]

                fig_p = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, 
                    subplot_titles=(f"VCD: {selected_data['filename']}", "IR Spectrum"),
                    row_heights=[0.5, 0.5]
                )
                fig_p.add_trace(go.Scatter(x=x, y=vcd, mode='lines', name='VCD', line=dict(color='#00008B', width=1.5)), row=1, col=1)
                fig_p.add_trace(go.Scatter(x=x, y=ir, mode='lines', name='IR', line=dict(color='#8B0000', width=1.5)), row=2, col=1)
                
                if do_peak and len(peak_x) > 0:
                    fig_p.add_trace(go.Scatter(x=peak_x, y=peak_vcd, mode='markers', marker=dict(symbol='x', color='black'), showlegend=False), row=1, col=1)
                    fig_p.add_trace(go.Scatter(x=peak_x, y=peak_ir, mode='markers', marker=dict(symbol='circle', color='red'), showlegend=False), row=2, col=1)

                fig_p.update_layout(height=600, hovermode="x unified", showlegend=False)
                fig_p.update_xaxes(title_text="Wavenumber (cm⁻¹)", range=[t1_x_high, t1_x_low], row=2, col=1)
                fig_p.update_xaxes(range=[t1_x_high, t1_x_low], row=1, col=1)
                if man_t1:
                    fig_p.update_yaxes(range=[t1_vcd_min, t1_vcd_max], row=1, col=1)
                    fig_p.update_yaxes(range=[t1_ir_min, t1_ir_max], row=2, col=1)
                fig_p.add_hline(y=0, line_width=1, line_color="black", row=1, col=1)
                st.plotly_chart(fig_p, use_container_width=True)

    # ==========================================
    # Tab 2: VCD 比較プロット (Comparison)
    # ==========================================
    with tab2:
        if not vcd_data:
            st.info("サイドバーからVCDファイルを読み込んでください。")
        else:
            st.subheader("VCD: Multi-Spectra Comparison")
            render_comparison_plot(vcd_data, "vcd", "VCD Intensity", "Absorbance", allow_noise=True)

    # ==========================================
    # Tab 3: LD解析 (Linear Dichroism)
    # ==========================================
    with tab3:
        if not ld_data:
            st.info("サイドバーの「LD解析用」エリアからファイルを読み込んでください。")
        else:
            st.subheader("LD (Linear Dichroism) Analysis")
            render_comparison_plot(ld_data, "ld", "LD Signal (3rd Col)", "Absorbance (2nd Col)", allow_noise=False)


# ---------------------------------------------------------
# 共通描画ロジック (VCD/LD共用)
# ---------------------------------------------------------
def render_comparison_plot(data_source, prefix, label_y1, label_y2, allow_noise=False):
    col_c_sel, col_c_opt = st.columns([1, 2])
    
    with col_c_sel:
        st.markdown("##### データ選択")
        all_filenames = [d['filename'] for d in data_source]
        selected_files = st.multiselect(
            "プロット対象", all_filenames, default=all_filenames, key=f"{prefix}_multi"
        )
        target_data = [d for d in data_source if d['filename'] in selected_files]
    
    with col_c_opt:
        st.markdown("##### グラフ設定")
        with st.form(key=f"{prefix}_plot_form"):
            c_leg, c_noise = st.columns(2)
            show_legend = c_leg.checkbox("凡例を表示", value=True, key=f"{prefix}_leg")
            
            show_noise = False
            if allow_noise:
                show_noise = c_noise.checkbox("ノイズ (4列目) を表示", value=False, key=f"{prefix}_nse")
            
            with st.expander("軸範囲設定", expanded=False):
                c1, c2 = st.columns(2)
                x_high = c1.number_input("X Left", value=2000.0, key=f"{prefix}_xh")
                x_low = c2.number_input("X Right", value=800.0, key=f"{prefix}_xl")
                
                man_y = st.checkbox("Y軸範囲固定", key=f"{prefix}_many")
                y1_min, y1_max = None, None
                y2_min, y2_max = None, None
                y3_min, y3_max = None, None
                
                if man_y:
                    y1_max = c1.number_input(f"1段目({label_y1}) Max", value=0.0005, format="%.5f", key=f"{prefix}_y1x")
                    y1_min = c2.number_input(f"1段目({label_y1}) Min", value=-0.0005, format="%.5f", key=f"{prefix}_y1n")
                    y2_max = c1.number_input(f"2段目({label_y2}) Max", value=1.0, key=f"{prefix}_y2x")
                    y2_min = c2.number_input(f"2段目({label_y2}) Min", value=0.0, key=f"{prefix}_y2n")
                    y3_max = c1.number_input("3段目(Noise) Max", value=0.0005, format="%.5f", key=f"{prefix}_y3x")
                    y3_min = c2.number_input("3段目(Noise) Min", value=-0.0005, format="%.5f", key=f"{prefix}_y3n")

            st.markdown("---")
            st.markdown("##### 🎨 スタイル設定 (色・太さ・倍率)")
            
            default_colors = list(mcolors.TABLEAU_COLORS.values())
            plot_styles = {}

            if target_data:
                with st.expander("設定パネルを開く", expanded=True):
                    cols = st.columns(3)
                    for i, d in enumerate(target_data):
                        fname = d['filename']
                        default_c = default_colors[i % len(default_colors)]
                        with cols[i % 3]:
                            st.caption(f"**{fname}**")
                            cc, cw, cs = st.columns([1, 1, 1])
                            p_color = cc.color_picker("Col", value=default_c, key=f"{prefix}_c_{fname}")
                            p_width = cw.number_input("Wid", value=1.5, step=0.5, key=f"{prefix}_w_{fname}")
                            p_scale = cs.number_input("Scl", value=1.0, step=0.5, key=f"{prefix}_s_{fname}")
                            plot_styles[fname] = {'color': p_color, 'width': p_width, 'scale': p_scale}

            submit_btn = st.form_submit_button("グラフを更新 (再プロット)")

    if submit_btn:
        if not target_data:
            st.warning("表示するデータがありません。")
            return

        layout_rows = 3 if show_noise else 2
        height = 10 if show_noise else 8
        fig, axes = plt.subplots(layout_rows, 1, sharex=True, figsize=(10, height), 
                                 gridspec_kw={'height_ratios': [1]*layout_rows})
        
        # axesをリスト化して扱いやすくする
        if layout_rows == 2:
            ax1, ax2 = axes
            ax3 = None
        else:
            ax1, ax2, ax3 = axes

        plt.subplots_adjust(hspace=0.05)
        
        for d in target_data:
            fname = d['filename']
            style = plot_styles.get(fname, {'color': 'black', 'width': 1.0, 'scale': 1.0})
            color = style['color']
            width = style['width']
            factor = style['scale']
            
            x_vals = d['x']
            y1_vals = d['vcd'] * factor
            y2_vals = d['ir']
            y3_vals = d['noise'] * factor
            
            label = f"{fname}" + (f" (x{factor})" if factor != 1.0 else "")
            
            ax1.plot(x_vals, y1_vals, color=color, linewidth=width, label=label)
            ax2.plot(x_vals, y2_vals, color=color, linewidth=width)
            if show_noise and ax3 is not None:
                ax3.plot(x_vals, y3_vals, color=color, linewidth=width)
        
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel(label_y1)
        ax1.set_xlim(x_high, x_low)
        if man_y: ax1.set_ylim(y1_min, y1_max)
        if show_legend: ax1.legend(loc='upper right', fontsize='small', framealpha=0.5)
        
        ax2.set_ylabel(label_y2)
        if man_y: ax2.set_ylim(y2_min, y2_max)
        
        if show_noise and ax3 is not None:
            ax3.axhline(0, color='black', linewidth=0.8)
            ax3.set_ylabel("Noise (4th Col)")
            ax3.set_xlabel("Wavenumber ($cm^{-1}$)")
            if man_y: ax3.set_ylim(y3_min, y3_max)
        else:
            ax2.set_xlabel("Wavenumber ($cm^{-1}$)")
        
        st.pyplot(fig)
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        c1.download_button(f"画像保存 ({prefix}_plot.png)", buf, f"{prefix}_plot.png", "image/png")
        
        zip_dat = create_gnuplot_package(
            target_data, plot_styles, (x_high, x_low), 
            (y1_min, y1_max), (y2_min, y2_max), (y3_min, y3_max),
            label_y1, label_y2, "Noise", include_noise=show_noise
        )
        if zip_dat:
            c2.download_button("Gnuplotデータ (.zip)", zip_dat, f"{prefix}_gnuplot.zip", "application/zip")
    
    elif target_data:
        st.info("👆 設定を変更し、「グラフを更新」ボタンを押してプロットしてください。")

if __name__ == "__main__":
    main()