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
# 関数: JASCO形式等のファイル読み込み
# ---------------------------------------------------------
def load_spectral_data(uploaded_file):
    """
    JASCO形式のテキストファイルなどを読み込み、
    {'filename': str, 'x': array, 'ir': array, 'vcd': array, 'noise': array, 'head': df} を返す。
    """
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        skip_rows = 0
        header_found = False
        
        # 'XYDATA' 行を探す
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                skip_rows = i + 1
                header_found = True
                break
        
        # 読み込み処理
        try:
            if header_found:
                df = pd.read_csv(io.StringIO(content), skiprows=skip_rows, sep='\t', header=None, engine='python')
                if df.shape[1] < 2:
                     df = pd.read_csv(io.StringIO(content), skiprows=skip_rows, sep='\s+', header=None, engine='python')
            else:
                df = pd.read_csv(io.StringIO(content), sep=None, engine='python', header=None)
        except Exception as e:
            return None, f"パースエラー: {e}"

        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        # --- 変更点: 2列でも許容する ---
        if df.shape[1] < 2:
            return None, "列数が不足しています (最低2列必要)"

        # データ抽出
        x = df.iloc[:, 0].values
        col2 = df.iloc[:, 1].values # 通常はIR/Abs。2列ファイルの場合はここが強度データ。
        
        # 3列目 (VCD/LD)
        if df.shape[1] >= 3:
            col3 = df.iloc[:, 2].values
        else:
            col3 = np.zeros_like(x) # ない場合は0埋め
        
        # 4列目 (Noise)
        if df.shape[1] >= 4:
            col4 = df.iloc[:, 3].values
        else:
            col4 = np.zeros_like(x)
        
        # 先頭5行を取得（確認用）
        head_df = df.head(5)
        
        return {
            'filename': uploaded_file.name,
            'x': x,
            'ir': col2,  
            'vcd': col3,
            'noise': col4,
            'head': head_df
        }, None

    except Exception as e:
        return None, f"読み込み例外: {e}"

# ---------------------------------------------------------
# 関数: データ結合 (VCDファイル + IRファイル)
# ---------------------------------------------------------
def merge_vcd_ir_data(vcd_source, ir_source, new_filename):
    """
    2つのデータオブジェクトを結合する。
    基準はVCD側のX軸とし、IR側を線形補間して合わせる。
    vcd_source: VCDデータを持つオブジェクト (2列ファイルの場合は ir キーにデータが入っている可能性があるため考慮)
    ir_source: IRデータを持つオブジェクト
    """
    # X軸の基準はVCDファイル側とする
    x_master = vcd_source['x']
    
    # VCDデータの取得
    # もしVCDソースが3列以上なら3列目(vcd)を使う。2列しかなかった場合は2列目(ir)に入っているとみなす。
    if np.all(vcd_source['vcd'] == 0) and not np.all(vcd_source['ir'] == 0):
        # 2列ファイルとして読み込まれた場合、データは 'ir' キーに入っている
        vcd_vals = vcd_source['ir']
    else:
        vcd_vals = vcd_source['vcd']

    # IRデータの取得と補間
    # IRソースは常に 'ir' キー (2列目) を使う
    ir_x = ir_source['x']
    ir_vals_raw = ir_source['ir']
    
    # 補間 (np.interpはxが増加順である必要があるため、降順の場合は反転させて処理)
    if x_master[0] > x_master[-1]: # 降順の場合
        new_ir = np.interp(x_master, ir_x[::-1], ir_vals_raw[::-1])
    else:
        new_ir = np.interp(x_master, ir_x, ir_vals_raw)

    return {
        'filename': new_filename,
        'x': x_master,
        'ir': new_ir,
        'vcd': vcd_vals,
        'noise': np.zeros_like(x_master),
        'head': pd.DataFrame({'Wavenumber': x_master[:5], 'Combined_IR': new_ir[:5], 'Combined_VCD': vcd_vals[:5]})
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
    common_x = np.sort(np.unique(all_x))[::-1] # 降順
    
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

    if include_noise:
        layout_cfg = "3,1 margins 0.15, 0.95, 0.1, 0.95 spacing 0.05"
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
set bmargin 0
set format x ""
plot {', '.join(plot_cmds_y2)}
"""
        p3 = f"""
set ylabel "{label_y3}"
set yrange {yr_y3}
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set format x "%g"
plot {', '.join(plot_cmds_y3)}
"""
        plot_body = p1 + p2 + p3
    else:
        layout_cfg = "2,1 margins 0.15, 0.95, 0.1, 0.95 spacing 0.05"
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
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set format x "%g"
plot {', '.join(plot_cmds_y2)}
"""
        plot_body = p1 + p2

    script = f"""
set terminal pngcairo size 800,{900 if include_noise else 800} font "Arial,12"
set output 'plot.png'
set multiplot layout {layout_cfg}
set xrange {xr}
set grid ls 1 lc rgb "gray" lw 0.5 dt 2
set lmargin 12
set tmargin 0
{plot_body}
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
    # 1. サイドバー: データ読み込み
    # ==========================================
    st.sidebar.header("📂 ファイル読み込み")
    
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
            data, error_msg = load_spectral_data(f)
            if data: data_list.append(data)
            else: st.sidebar.error(f"VCD Error {f.name}: {error_msg}")
        if data_list:
            # 既存データに追加する形にするか、上書きするか。ここでは追加して重複排除
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
            data, error_msg = load_spectral_data(f)
            if data: data_list.append(data)
            else: st.sidebar.error(f"LD Error {f.name}: {error_msg}")
        if data_list:
            current_files = {d['filename'] for d in st.session_state['ld_data']}
            for d in data_list:
                if d['filename'] not in current_files:
                    st.session_state['ld_data'].append(d)
            st.sidebar.success(f"LD: {len(data_list)}件 読込完了")
    
    # === 追加機能: ファイル結合ツール ===
    # VCDデータとして読み込まれたファイル群を使って結合を行う
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
                # オブジェクト取得
                obj_vcd = next(d for d in loaded_files if d['filename'] == f_vcd)
                obj_ir = next(d for d in loaded_files if d['filename'] == f_ir)
                
                # 結合処理
                merged_data = merge_vcd_ir_data(obj_vcd, obj_ir, new_name)
                
                # リストに追加
                st.session_state['vcd_data'].append(merged_data)
                st.sidebar.success(f"結合完了: {new_name}")

    # === 追加機能: ファイル先頭行の確認 ===
    all_loaded = st.session_state['vcd_data'] + st.session_state['ld_data']
    if all_loaded:
        st.sidebar.markdown("---")
        with st.sidebar.expander("📄 読み込みデータの確認 (先頭5行)"):
            file_opts = [d['filename'] for d in all_loaded]
            sel_check = st.selectbox("確認するファイル", file_opts)
            for d in all_loaded:
                if d['filename'] == sel_check:
                    st.caption("※パース後の数値データ")
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
                # リストが更新された場合のためにindexエラー回避
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
    """
    比較プロットを描画する共通関数
    """
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

        if show_noise:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10), 
                                                gridspec_kw={'height_ratios': [1, 1, 1]})
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                           gridspec_kw={'height_ratios': [1, 1]})
            ax3 = None

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