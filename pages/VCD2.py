import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors

# ---------------------------------------------------------
# 関数: データ読み込み (強化版)
# ---------------------------------------------------------
def load_spectral_data(uploaded_file, params):
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        jasco_skip = 0
        is_jasco = False
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                jasco_skip = i + 1
                is_jasco = True
                break
        
        df = None
        
        if is_jasco:
            try:
                df = pd.read_csv(io.StringIO(content), skiprows=jasco_skip, sep='\t', header=None, engine='python')
                if df.shape[1] < 2:
                     df = pd.read_csv(io.StringIO(content), skiprows=jasco_skip, sep='\s+', header=None, engine='python')
            except:
                pass 
        
        if df is None:
            sep_char = params['sep']
            sep_arg = None if sep_char == 'auto' else sep_char
            comment_arg = params['comment']
            skip_rows = params['skip_rows']

            if not comment_arg and lines and lines[0].strip().startswith('#'):
                comment_arg = '#'
            
            try:
                df = pd.read_csv(
                    io.StringIO(content), 
                    skiprows=skip_rows, 
                    sep=sep_arg, 
                    comment=comment_arg, 
                    header=None, 
                    engine='python'
                )
            except Exception:
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

        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.empty:
            return None, "有効なデータ行がありません"

        def get_col_data(df, idx):
            if 0 <= idx < df.shape[1]:
                return df.iloc[:, idx].values
            return np.zeros(len(df))

        col_x_idx = params['cols']['x']
        col_ir_idx = params['cols']['ir']
        col_vcd_idx = params['cols']['vcd']
        col_noise_idx = params['cols']['noise']

        if col_x_idx >= df.shape[1]:
             return None, f"指定されたX列({col_x_idx+1}列目)がデータ内に存在しません"

        x = get_col_data(df, col_x_idx)
        col_ir = get_col_data(df, col_ir_idx)
        col_vcd = get_col_data(df, col_vcd_idx)
        col_noise = get_col_data(df, col_noise_idx)

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
# 関数: Gnuplot用パッケージ作成 (汎用)
# ---------------------------------------------------------
def create_gnuplot_package(data_list, settings_dict, x_lim, y_labels, show_noise=False):
    if not data_list: return None
    
    all_x = []
    for d in data_list:
        all_x.extend(d['x'])
    common_x = np.sort(np.unique(all_x))[::-1] 
    
    df_out = pd.DataFrame({'Wavenumber': common_x})
    cmds_y1 = []
    cmds_y2 = []
    cmds_y3 = []
    
    current_col = 2
    
    dt_map = {'solid': 1, 'dash': 2, 'dot': 3, 'dashdot': 4}

    for d in data_list:
        fname = d['filename']
        st = settings_dict.get(fname, {})
        
        c = st.get('color', 'black')
        w = st.get('width', 2.0)
        dt = dt_map.get(st.get('dash', 'solid'), 1)
        
        v_s, v_o = st.get('vcd_scale', 1.0), st.get('vcd_offset', 0.0)
        i_s, i_o = st.get('ir_scale', 1.0), st.get('ir_offset', 0.0)
        
        y1_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1]) * v_s + v_o
        y2_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1]) * i_s + i_o
        y3_interp = np.interp(common_x, d['x'][::-1], d['noise'][::-1]) * v_s 
        
        safe_name = f"File_{current_col//3}"
        df_out[f"{safe_name}_Y1"] = y1_interp
        df_out[f"{safe_name}_Y2"] = y2_interp
        df_out[f"{safe_name}_Y3"] = y3_interp
        
        title = fname.replace('_', '\\_')
        common_style = f"w l lc rgb '{c}' lw {w} dt {dt}"
        cmds_y1.append(f"'data.dat' u 1:{current_col} {common_style} title '{title}'")
        cmds_y2.append(f"'data.dat' u 1:{current_col+1} {common_style} notitle") 
        if show_noise:
            cmds_y3.append(f"'data.dat' u 1:{current_col+2} {common_style} notitle")
            
        current_col += 3

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.6f')
    xr = f"[{x_lim[1]}:{x_lim[0]}]" 

    layout_rows = 3 if show_noise else 2
    height = 900 if show_noise else 800
    
    p1 = f"""
set ylabel "{y_labels[0]}"
set bmargin 0
set format x ""
set xzeroaxis lt 1 lc rgb "black" lw 1
plot {', '.join(cmds_y1)}
"""
    p2 = f"""
set ylabel "{y_labels[1]}"
set bmargin {0 if show_noise else 4}
set format x {"''" if show_noise else "'%g'"}
{'' if show_noise else 'set xlabel "Wavenumber (cm^{-1})"'}
plot {', '.join(cmds_y2)}
"""
    p3 = ""
    if show_noise:
        p3 = f"""
set ylabel "{y_labels[2]}"
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set format x "%g"
plot {', '.join(cmds_y3)}
"""

    script = f"""
set terminal pngcairo size 800,{height} font "Arial,12"
set output 'plot.png'
set multiplot layout {layout_rows},1 margins 0.15, 0.85, 0.1, 0.95 spacing 0.05
set xrange {xr}
set grid ls 1 lc rgb "gray" lw 0.5 dt 2
set lmargin 12
set rmargin 5
set tmargin 0
set key right top font ",10"
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
# Matplotlib 比較描画 (強化版)
# ---------------------------------------------------------
def render_matplotlib_comparison_advanced(data_source, prefix, label_y1, label_y2, allow_noise=False):
    col_c_sel, col_c_opt = st.columns([1, 2])
    
    with col_c_sel:
        st.markdown("##### データ選択")
        all_filenames = [d['filename'] for d in data_source]
        selected_files = st.multiselect(
            "プロット対象", all_filenames, default=all_filenames, key=f"{prefix}_multi"
        )
        target_data = [d for d in data_source if d['filename'] in selected_files]
    
    with col_c_opt:
        with st.expander("⚙️ グラフ設定 (軸・範囲・スタイル) [クリックで開閉]", expanded=True):
            with st.form(key=f"{prefix}_plot_form"):
                st.markdown("###### 全般設定")
                c_leg, c_noise = st.columns(2)
                show_legend = c_leg.checkbox("凡例を表示", value=True, key=f"{prefix}_leg")
                
                show_noise = False
                if allow_noise:
                    show_noise = c_noise.checkbox("ノイズ (4列目) を表示", value=False, key=f"{prefix}_nse")
                
                st.markdown("---")
                st.markdown("###### 軸範囲設定 (Axis Range)")
                c1, c2 = st.columns(2)
                x_high = c1.number_input("Left (High cm-1)", value=2000.0, key=f"{prefix}_xh")
                x_low = c2.number_input("Right (Low cm-1)", value=800.0, key=f"{prefix}_xl")
                
                st.markdown("---")
                st.markdown("###### Y軸 (強度)")
                man_y = st.checkbox("Y軸の範囲を手動で固定する", key=f"{prefix}_many")
                
                y1_min, y1_max = None, None
                y2_min, y2_max = None, None
                y3_min, y3_max = None, None
                
                if man_y:
                    st.caption(f"**1段目: {label_y1}**")
                    c_y1_1, c_y1_2 = st.columns(2)
                    y1_max = c_y1_1.number_input("Max", value=0.0005, format="%.5f", key=f"{prefix}_y1x")
                    y1_min = c_y1_2.number_input("Min", value=-0.0005, format="%.5f", key=f"{prefix}_y1n")
                    
                    st.caption(f"**2段目: {label_y2}**")
                    c_y2_1, c_y2_2 = st.columns(2)
                    y2_max = c_y2_1.number_input("Max", value=1.0, key=f"{prefix}_y2x")
                    y2_min = c_y2_2.number_input("Min", value=0.0, key=f"{prefix}_y2n")
                    
                    if allow_noise and show_noise:
                        st.caption("**3段目: Noise**")
                        c_y3_1, c_y3_2 = st.columns(2)
                        y3_max = c_y3_1.number_input("Max", value=0.0005, format="%.5f", key=f"{prefix}_y3x")
                        y3_min = c_y3_2.number_input("Min", value=-0.0005, format="%.5f", key=f"{prefix}_y3n")

                st.markdown("---")
                st.markdown("###### 🎨 詳細スタイル & スペクトル操作")
                
                default_colors = list(mcolors.TABLEAU_COLORS.values())
                plot_settings = {} 

                if target_data:
                    for i, d in enumerate(target_data):
                        fname = d['filename']
                        def_c = default_colors[i % len(default_colors)]
                        
                        st.markdown(f"**{i+1}. {fname}**")
                        
                        c_s1, c_s2, c_s3 = st.columns(3)
                        p_color = c_s1.color_picker("Color", value=def_c, key=f"{prefix}_c_{i}")
                        p_width = c_s2.number_input("Width", value=1.5, step=0.5, key=f"{prefix}_w_{i}")
                        p_style = c_s3.selectbox("Line Style", ["solid", "dash", "dot", "dashdot"], index=0, key=f"{prefix}_ls_{i}")
                        
                        c_o1, c_o2, c_o3, c_o4 = st.columns(4)
                        v_scale = c_o1.number_input(f"{label_y1} x", value=1.0, step=0.1, key=f"{prefix}_vs_{i}")
                        v_offset = c_o2.number_input(f"{label_y1} +", value=0.0, step=0.0001, format="%.5f", key=f"{prefix}_vo_{i}")
                        i_scale = c_o3.number_input(f"{label_y2} x", value=1.0, step=0.1, key=f"{prefix}_is_{i}")
                        i_offset = c_o4.number_input(f"{label_y2} +", value=0.0, step=0.1, key=f"{prefix}_io_{i}")
                        
                        plot_settings[fname] = {
                            'color': p_color, 'width': p_width, 'dash': p_style,
                            'vcd_scale': v_scale, 'vcd_offset': v_offset,
                            'ir_scale': i_scale, 'ir_offset': i_offset
                        }
                        st.divider()

                submit_btn = st.form_submit_button("グラフを更新 (再プロット)")

    if submit_btn:
        if not target_data:
            st.warning("表示するデータがありません。")
            return

        layout_rows = 3 if show_noise else 2
        height = 10 if show_noise else 8
        fig, axes = plt.subplots(layout_rows, 1, sharex=True, figsize=(10, height), 
                                 gridspec_kw={'height_ratios': [1]*layout_rows})
        
        if layout_rows == 2:
            ax1, ax2 = axes
            ax3 = None
        else:
            ax1, ax2, ax3 = axes

        plt.subplots_adjust(hspace=0.05)
        
        for d in target_data:
            fname = d['filename']
            stt = plot_settings.get(fname, {})
            
            color = stt.get('color', 'black')
            width = stt.get('width', 1.5)
            ls = stt.get('dash', 'solid')
            
            v_s, v_o = stt.get('vcd_scale', 1.0), stt.get('vcd_offset', 0.0)
            i_s, i_o = stt.get('ir_scale', 1.0), stt.get('ir_offset', 0.0)
            
            x_vals = d['x']
            y1_vals = d['vcd'] * v_s + v_o
            y2_vals = d['ir'] * i_s + i_o
            y3_vals = d['noise'] * v_s 
            
            label = fname
            
            ax1.plot(x_vals, y1_vals, color=color, linewidth=width, linestyle=ls, label=label)
            ax2.plot(x_vals, y2_vals, color=color, linewidth=width, linestyle=ls)
            if show_noise and ax3 is not None:
                ax3.plot(x_vals, y3_vals, color=color, linewidth=width, linestyle=ls)
        
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel(label_y1)
        ax1.set_xlim(x_high, x_low)
        if man_y: ax1.set_ylim(y1_min, y1_max)
        if show_legend: ax1.legend(loc='upper right', fontsize='small', framealpha=0.5)
        
        ax2.set_ylabel(label_y2)
        if man_y: ax2.set_ylim(y2_min, y2_max)
        
        if show_noise and ax3 is not None:
            ax3.axhline(0, color='black', linewidth=0.8)
            ax3.set_ylabel("Noise")
            ax3.set_xlabel("Wavenumber ($cm^{-1}$)")
            if man_y: ax3.set_ylim(y3_min, y3_max)
        else:
            ax2.set_xlabel("Wavenumber ($cm^{-1}$)")
        
        st.pyplot(fig)
        
        st.markdown("---")
        c_dl, _ = st.columns([1, 2])
        zip_dat = create_gnuplot_package(
            target_data, plot_settings, (x_high, x_low), 
            (label_y1, label_y2, "Noise"), show_noise
        )
        if zip_dat:
            c_dl.download_button("💾 Gnuplotデータ (.zip) を保存", zip_dat, f"{prefix}_gnuplot.zip", "application/zip")
    
    elif target_data:
        st.info("👆 設定を変更し、「グラフを更新」ボタンを押してプロットしてください。")

# ---------------------------------------------------------
# 関数: Gnuplot用パッケージ作成 (比較用) [Tab 4用]
# ---------------------------------------------------------
def create_gnuplot_comparison_package(exp_list, calc_list, style_dict, x_lim, use_dual_axis):
    if not exp_list and not calc_list: return None
    all_x = []
    for d in exp_list + calc_list: all_x.extend(d['x'])
    common_x = np.sort(np.unique(all_x))[::-1] 
    df_out = pd.DataFrame({'Wavenumber': common_x})
    plot_cmds_vcd = []
    plot_cmds_ir = []
    current_col = 2
    
    def process_dataset(data_list, group_name):
        nonlocal current_col
        cmds_vcd = []
        cmds_ir = []
        for i, d in enumerate(data_list):
            fname = d['filename']
            style = style_dict.get(fname, {'color': 'black', 'width': 2.0, 'dash': 'solid'})
            color = style['color']
            width = style['width']
            dt_map = {'solid': 1, 'dash': 2, 'dot': 3, 'dashdot': 4}
            dt = dt_map.get(style['dash'], 1)
            
            ir_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
            vcd_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1])
            safe_name = f"{group_name}_{i+1}"
            df_out[f"{safe_name}_IR"] = ir_interp
            df_out[f"{safe_name}_VCD"] = vcd_interp
            title = fname.replace('_', '\\_')
            
            axes_opt = "axes x1y2" if (use_dual_axis and group_name == "Calc") else ""
            cmds_vcd.append(f"'data.dat' u 1:{current_col+1} w l {axes_opt} lc rgb '{color}' lw {width} dt {dt} title '{title}'")
            cmds_ir.append(f"'data.dat' u 1:{current_col} w l {axes_opt} lc rgb '{color}' lw {width} dt {dt} title '{title}'")
            current_col += 2
        return cmds_vcd, cmds_ir

    vcd_cmds_exp, ir_cmds_exp = process_dataset(exp_list, "Exp")
    plot_cmds_vcd.extend(vcd_cmds_exp)
    plot_cmds_ir.extend(ir_cmds_exp)
    vcd_cmds_calc, ir_cmds_calc = process_dataset(calc_list, "Calc")
    plot_cmds_vcd.extend(vcd_cmds_calc)
    plot_cmds_ir.extend(ir_cmds_calc)

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.6f')
    xr = f"[{x_lim[1]}:{x_lim[0]}]"
    
    # Gnuplot側も2段構成に戻す
    dual_axis_setup_vcd = "set ytics nomirror\nset y2tics" if use_dual_axis else ""
    dual_axis_setup_ir = "set ytics nomirror\nset y2tics" if use_dual_axis else ""

    script = f"""
set terminal pngcairo size 1000,800 font "Arial,12"
set output 'comparison.png'
set multiplot layout 2,1 margins 0.15, 0.85, 0.1, 0.95 spacing 0.05
set xrange {xr}
set grid ls 1 lc rgb "gray" lw 0.5 dt 2
set lmargin 12
set rmargin 12
set bmargin 0
set format x ""
set xzeroaxis lt 1 lc rgb "black" lw 1
set key right top font ",10"

# VCD Plot
set ylabel 'VCD Intensity'
{dual_axis_setup_vcd}
plot {', '.join(plot_cmds_vcd)}

# IR Plot
set ylabel 'IR Absorbance'
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set tmargin 0
set format x "%g"
{dual_axis_setup_ir}
plot {', '.join(plot_cmds_ir)}

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
    if 'calc_data' not in st.session_state: st.session_state['calc_data'] = []

    # ==========================================
    # 1. サイドバー: データ読み込み設定
    # ==========================================
    st.sidebar.header("📂 ファイル読み込み")

    sep_map = {"自動 (Space/Tab)": "auto", "カンマ (,)": ",", "タブ (\\t)": "\t"}

    with st.sidebar.expander("⚙️ 実験データ読み込み設定 (非JASCO)", expanded=False):
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

    with st.sidebar.expander("⚙️ 計算データ読み込み設定", expanded=False):
        c_c1, c_c2 = st.columns(2)
        calc_skip = c_c1.number_input("Skip Rows", value=0, min_value=0, key="calc_skip")
        calc_sep_mode = c_c2.selectbox("Separator", list(sep_map.keys()), key="calc_sep")
        calc_comment = st.text_input("Comment Char", value="#", key="calc_comment")
        st.markdown("**列番号 (1始まり)**")
        cc_c1, cc_c2 = st.columns(2)
        calc_col_x = cc_c1.number_input("X (波数)", value=1, min_value=1, key="calc_cx")
        calc_col_ir = cc_c2.number_input("IR (2)", value=2, min_value=1, key="calc_ci")
        calc_col_vcd = cc_c1.number_input("VCD (3)", value=3, min_value=1, key="calc_cv")

    params_calc = {
        "skip_rows": calc_skip,
        "sep": sep_map[calc_sep_mode],
        "comment": calc_comment if calc_comment else None,
        "cols": {"x": calc_col_x-1, "ir": calc_col_ir-1, "vcd": calc_col_vcd-1, "noise": 999}
    }

    st.sidebar.subheader("1. 実験データ (Exp)")
    uploaded_vcd = st.sidebar.file_uploader("VCD/IR 実験ファイル", accept_multiple_files=True, key="up_vcd", type=['txt', 'csv', 'dat'])
    if uploaded_vcd:
        for f in uploaded_vcd:
            if not any(d['filename'] == f.name for d in st.session_state['vcd_data']):
                data, err = load_spectral_data(f, params_exp)
                if data: st.session_state['vcd_data'].append(data)
                else: st.sidebar.error(f"{f.name}: {err}")

    uploaded_ld = st.sidebar.file_uploader("LD 実験ファイル", accept_multiple_files=True, key="up_ld", type=['txt', 'csv', 'dat'])
    if uploaded_ld:
        for f in uploaded_ld:
            if not any(d['filename'] == f.name for d in st.session_state['ld_data']):
                data, err = load_spectral_data(f, params_exp)
                if data: st.session_state['ld_data'].append(data)
                else: st.sidebar.error(f"{f.name}: {err}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 計算データ (Calc)")
    uploaded_calc = st.sidebar.file_uploader("計算データ (.txt/.csv)", accept_multiple_files=True, key="up_calc")
    if uploaded_calc:
        for f in uploaded_calc:
            if not any(d['filename'] == f.name for d in st.session_state['calc_data']):
                data, err = load_spectral_data(f, params_calc)
                if data: st.session_state['calc_data'].append(data)
                else: st.sidebar.error(f"{f.name}: {err}")
    
    if st.session_state['vcd_data']:
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔗 データの結合 (VCD + IR)", expanded=False):
            loaded_files = st.session_state['vcd_data']
            filenames = [d['filename'] for d in loaded_files]
            f_vcd = st.selectbox("VCDデータ", filenames, key="sel_merge_vcd")
            f_ir = st.selectbox("IRデータ", filenames, key="sel_merge_ir")
            new_name = st.text_input("結合後ファイル名", value=f"Combined_{f_vcd}")
            if st.button("結合して追加"):
                obj_vcd = next(d for d in loaded_files if d['filename'] == f_vcd)
                obj_ir = next(d for d in loaded_files if d['filename'] == f_ir)
                merged_data = merge_vcd_ir_data(obj_vcd, obj_ir, new_name)
                st.session_state['vcd_data'].append(merged_data)
                st.sidebar.success(f"結合完了: {new_name}")

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

    # Tab 1: VCD 個別
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
                if show_peak: p_th = st.slider("Threshold", 0.0, 1.0, 0.05)
                with st.expander("Axis Settings", expanded=False):
                    c1, c2 = st.columns(2)
                    x_left = c1.number_input("X Left", value=2000.0, step=100.0, key="t1_xl")
                    x_right = c2.number_input("X Right", value=800.0, step=100.0, key="t1_xr")
                    man_y = st.checkbox("Manual Y-Range", key="t1_man_y")
                    y_vcd_min, y_vcd_max = None, None
                    y_ir_min, y_ir_max = None, None
                    if man_y:
                        c3, c4 = st.columns(2)
                        y_vcd_max = c3.number_input("VCD Max", value=0.001, format="%.5f", key="t1_vmax")
                        y_vcd_min = c4.number_input("VCD Min", value=-0.001, format="%.5f", key="t1_vmin")
                        c5, c6 = st.columns(2)
                        y_ir_max = c5.number_input("IR Max", value=1.5, key="t1_imax")
                        y_ir_min = c6.number_input("IR Min", value=0.0, key="t1_imin")

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
                fig.update_xaxes(range=[x_left, x_right], row=1, col=1)
                fig.update_xaxes(range=[x_left, x_right], title_text="Wavenumber (cm⁻¹)", row=2, col=1)
                if man_y:
                    fig.update_yaxes(range=[y_vcd_min, y_vcd_max], row=1, col=1)
                    fig.update_yaxes(range=[y_ir_min, y_ir_max], row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

    # Tab 2: VCD 比較
    with tab2:
        if not vcd_data: st.info("データがありません。")
        else:
            st.subheader("Multi-Spectra Comparison (VCD)")
            render_matplotlib_comparison_advanced(vcd_data, "vcd", "VCD Intensity", "Absorbance", allow_noise=True)

    # Tab 3: LD 解析
    with tab3:
        if not ld_data: st.info("LDデータがありません。")
        else:
            st.subheader("LD Analysis")
            render_matplotlib_comparison_advanced(ld_data, "ld", "LD Signal", "Absorbance", allow_noise=False)

    # Tab 4: 実験 vs 計算 (修正: 2段表示 + 手動Y軸)
    with tab4:
        st.subheader("🔬 Experimental vs Computational Comparison")
        c_exp, c_calc = st.columns(2)
        with c_exp:
            st.markdown("##### 1. 実験データ (Exp)")
            if not vcd_data:
                st.warning("実験データがありません")
                target_exp_data = []
            else:
                exp_names = [d['filename'] for d in vcd_data]
                sel_exp_names = st.multiselect("ファイル選択", exp_names, default=[exp_names[0]], key="tv_exp_multi")
                target_exp_data = [d for d in vcd_data if d['filename'] in sel_exp_names]
        with c_calc:
            st.markdown("##### 2. 計算データ (Calc)")
            if not calc_data:
                st.warning("計算データがありません")
                target_calc_data = []
            else:
                calc_names = [d['filename'] for d in calc_data]
                sel_calc_names = st.multiselect("ファイル選択", calc_names, default=[calc_names[0]] if calc_names else None, key="tv_calc_multi")
                target_calc_data = [d for d in calc_data if d['filename'] in sel_calc_names]

        st.markdown("---")

        if target_exp_data or target_calc_data:
            with st.expander("🎚️ パラメータ & 🎨 スタイル設定", expanded=True):
                col_para1, col_para2, col_para3 = st.columns(3)
                with col_para1:
                    st.markdown("**X軸 (波数) 補正 [Calcのみ]**")
                    scale_freq = st.number_input("Scaling Factor", value=0.980, step=0.001, format="%.4f")
                    shift_freq = st.number_input("Shift (+/-)", value=0.0, step=1.0)
                with col_para2:
                    st.markdown("**Y軸 (強度) 倍率 [Calcのみ]**")
                    # 修正: 0.001単位で設定可能に
                    scale_int_vcd = st.number_input("VCD Scale", value=1.0, step=0.001, format="%.4f")
                    scale_int_ir = st.number_input("IR Scale", value=1.0, step=0.001, format="%.4f")
                with col_para3:
                    st.markdown("**表示設定**")
                    use_dual_axis = st.checkbox("Calcを右軸にする (Dual Axis)", value=True)
                    plot_range = st.slider("表示範囲 (cm-1)", 0, 4000, (800, 2000))
                
                # Manual Y Range
                st.markdown("---")
                st.markdown("###### Y軸 手動範囲設定 (Exp軸/主軸)")
                use_manual_y = st.checkbox("Y軸の範囲を手動で固定する", value=False, key="t4_manual_y")
                
                t4_vcd_min, t4_vcd_max = None, None
                t4_ir_min, t4_ir_max = None, None
                
                if use_manual_y:
                    c_my1, c_my2 = st.columns(2)
                    with c_my1:
                        st.caption("VCD Range (上段)")
                        t4_vcd_max = st.number_input("VCD Max", value=0.0001, format="%.6f", key="t4_vmx")
                        t4_vcd_min = st.number_input("VCD Min", value=-0.0001, format="%.6f", key="t4_vmn")
                    with c_my2:
                        st.caption("IR Range (下段)")
                        t4_ir_max = st.number_input("IR Max", value=1.0, format="%.2f", key="t4_imx")
                        t4_ir_min = st.number_input("IR Min", value=0.0, format="%.2f", key="t4_imn")
                
                st.markdown("---")
                st.markdown("##### グラフスタイル詳細設定")
                style_dict = {} 
                default_colors = pc.qualitative.Plotly
                
                if target_exp_data:
                    st.caption("実験データ")
                    cols_e = st.columns(3)
                    for i, d in enumerate(target_exp_data):
                        fname = d['filename']
                        def_c = default_colors[i % len(default_colors)]
                        with cols_e[i % 3]:
                            st.markdown(f"**{fname}**")
                            c = st.color_picker("Color", def_c, key=f"ec_{fname}")
                            w = st.number_input("Width", 1.0, 5.0, 2.0, 0.5, key=f"ew_{fname}")
                            s = st.selectbox("Style", ["solid", "dash", "dot", "dashdot"], index=0, key=f"es_{fname}")
                            style_dict[fname] = {'color': c, 'width': w, 'dash': s}
                
                if target_calc_data:
                    st.caption("計算データ")
                    cols_c = st.columns(3)
                    offset = len(target_exp_data)
                    for i, d in enumerate(target_calc_data):
                        fname = d['filename']
                        def_c = default_colors[(offset + i) % len(default_colors)]
                        with cols_c[i % 3]:
                            st.markdown(f"**{fname}**")
                            c = st.color_picker("Color", def_c, key=f"cc_{fname}")
                            w = st.number_input("Width", 1.0, 5.0, 1.5, 0.5, key=f"cw_{fname}")
                            s = st.selectbox("Style", ["solid", "dash", "dot", "dashdot"], index=1, key=f"cs_{fname}")
                            style_dict[fname] = {'color': c, 'width': w, 'dash': s}

            # --------------------------------------------------------
            # プロット作成: 上下2段 (Row1: VCD, Row2: IR)
            # --------------------------------------------------------
            fig_cmp = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.1,
                specs=[[{"secondary_y": True}], [{"secondary_y": True}]], 
                subplot_titles=("VCD Comparison", "IR Comparison")
            )
            
            processed_calc_data = []

            # 実験データ (Primary Y)
            for d in target_exp_data:
                style = style_dict[d['filename']]
                # VCD -> Row 1, Expは常に左軸(secondary_y=False)
                fig_cmp.add_trace(go.Scatter(x=d['x'], y=d['vcd'], name=f"Exp: {d['filename']}", 
                                             line=dict(color=style['color'], width=style['width'], dash=style['dash'])), 
                                  row=1, col=1, secondary_y=False)
                # IR -> Row 2, Expは常に左軸
                fig_cmp.add_trace(go.Scatter(x=d['x'], y=d['ir'], name=f"Exp IR: {d['filename']}", 
                                             line=dict(color=style['color'], width=style['width'], dash=style['dash']), showlegend=False), 
                                  row=2, col=1, secondary_y=False)

            # 計算データ (Primary or Secondary Y)
            for d in target_calc_data:
                style = style_dict[d['filename']]
                calc_x = d['x'] * scale_freq + shift_freq
                calc_vcd = d['vcd'] * scale_int_vcd
                calc_ir = d['ir'] * scale_int_ir
                processed_calc_data.append({'filename': d['filename'], 'x': calc_x, 'vcd': calc_vcd, 'ir': calc_ir})

                # VCD -> Row 1
                fig_cmp.add_trace(go.Scatter(x=calc_x, y=calc_vcd, name=f"Calc: {d['filename']}", 
                                             line=dict(color=style['color'], width=style['width'], dash=style['dash'])), 
                                  row=1, col=1, secondary_y=use_dual_axis)
                # IR -> Row 2
                fig_cmp.add_trace(go.Scatter(x=calc_x, y=calc_ir, name=f"Calc IR: {d['filename']}", 
                                             line=dict(color=style['color'], width=style['width'], dash=style['dash']), showlegend=False), 
                                  row=2, col=1, secondary_y=use_dual_axis)

            fig_cmp.update_layout(height=700, hovermode="x unified")
            fig_cmp.update_xaxes(range=[plot_range[1], plot_range[0]], row=2, col=1, title_text="Wavenumber (cm⁻¹)")
            fig_cmp.update_xaxes(range=[plot_range[1], plot_range[0]], row=1, col=1)
            
            # 軸ラベル
            fig_cmp.update_yaxes(title_text="Exp Signal", secondary_y=False, row=1, col=1)
            fig_cmp.update_yaxes(title_text="Absorbance", secondary_y=False, row=2, col=1)
            
            if use_dual_axis: 
                fig_cmp.update_yaxes(title_text="Calc Signal", secondary_y=True, showgrid=False, row=1, col=1)
                fig_cmp.update_yaxes(title_text="Calc Absorbance", secondary_y=True, showgrid=False, row=2, col=1)
            
            # 手動Y軸設定 (Primary Axisのみ適用)
            if use_manual_y:
                if t4_vcd_min is not None and t4_vcd_max is not None:
                    fig_cmp.update_yaxes(range=[t4_vcd_min, t4_vcd_max], secondary_y=False, row=1, col=1)
                if t4_ir_min is not None and t4_ir_max is not None:
                    fig_cmp.update_yaxes(range=[t4_ir_min, t4_ir_max], secondary_y=False, row=2, col=1)

            st.plotly_chart(fig_cmp, use_container_width=True)
            
            st.markdown("---")
            col_dl, _ = st.columns([1, 2])
            zip_dat = create_gnuplot_comparison_package(
                target_exp_data, processed_calc_data, style_dict, plot_range, use_dual_axis
            )
            if zip_dat:
                col_dl.download_button("💾 Gnuplotデータ (.zip) を保存", zip_dat, "comparison_gnuplot.zip", "application/zip")

if __name__ == "__main__":
    main()