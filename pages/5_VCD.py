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
    {'filename': str, 'x': np.array, 'ir': np.array, 'vcd': np.array} の辞書を返す。
    """
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        skip_rows = 0
        header_found = False
        
        # 'XYDATA' 行を探す (JASCO形式対応)
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                skip_rows = i + 1
                header_found = True
                break
        
        try:
            if header_found:
                # XYDATAが見つかった場合はその次から読む
                df = pd.read_csv(io.StringIO(content), skiprows=skip_rows, sep='\t', header=None, engine='python')
                if df.shape[1] < 3: # タブで失敗したらスペースで再試行
                     df = pd.read_csv(io.StringIO(content), skiprows=skip_rows, sep='\s+', header=None, engine='python')
            else:
                # 見つからない場合は自動推論
                df = pd.read_csv(io.StringIO(content), sep=None, engine='python', header=None)
        except Exception as e:
            return None, f"パースエラー: {e}"

        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] < 3:
            return None, "列数が不足しています (波数, IR, VCDが必要です)"

        x = df.iloc[:, 0].values
        ir = df.iloc[:, 1].values
        vcd = df.iloc[:, 2].values
        
        return {
            'filename': uploaded_file.name,
            'x': x,
            'ir': ir,
            'vcd': vcd
        }, None

    except Exception as e:
        return None, f"読み込み例外: {e}"

# ---------------------------------------------------------
# 関数: Gnuplot用パッケージ作成
# ---------------------------------------------------------
def create_gnuplot_package(data_list, style_dict, x_lim, vcd_lim, ir_lim):
    if not data_list: return None
    
    all_x = []
    for d in data_list:
        all_x.extend(d['x'])
    common_x = np.sort(np.unique(all_x))[::-1] # 降順
    
    df_out = pd.DataFrame({'Wavenumber': common_x})
    plot_cmds_vcd = []
    plot_cmds_ir = []
    
    current_col = 2
    for i, d in enumerate(data_list):
        fname = d['filename']
        # スタイル辞書から設定を取得（なければデフォルト）
        style = style_dict.get(fname, {'color': 'black', 'scale': 1.0})
        color = style['color']
        scale = style['scale']
        
        # 共通軸へ補間
        ir_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
        vcd_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1]) * scale # ここで倍率反映
        
        safe_name = f"File_{i+1}"
        df_out[f"{safe_name}_IR"] = ir_interp
        df_out[f"{safe_name}_VCD"] = vcd_interp
        
        title = fname.replace('_', '\\_')
        if scale != 1.0:
            title += f" (x{scale})"
        
        plot_cmds_ir.append(f"'data.dat' u 1:{current_col} w l lc rgb '{color}' title '{title}'")
        plot_cmds_vcd.append(f"'data.dat' u 1:{current_col+1} w l lc rgb '{color}' notitle")
        current_col += 2

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.6f')

    xr = f"[{x_lim[0]}:{x_lim[1]}]"
    yr_vcd = f"[{vcd_lim[0]}:{vcd_lim[1]}]" if vcd_lim[0] is not None else "[:]"
    yr_ir = f"[{ir_lim[0]}:{ir_lim[1]}]" if ir_lim[0] is not None else "[:]"

    script = f"""
set terminal pngcairo size 800,800 font "Arial,12"
set output 'vcd_plot.png'
set multiplot layout 2,1 margins 0.15, 0.95, 0.1, 0.95 spacing 0.05
set xrange {xr}
set grid ls 1 lc rgb "gray" lw 0.5 dt 2
set ylabel "VCD Intensity"
set yrange {yr_vcd}
set lmargin 12
set bmargin 0
set format x ""
set xzeroaxis lt 1 lc rgb "black" lw 1
plot {', '.join(plot_cmds_vcd)}
set ylabel "Absorbance"
set yrange {yr_ir}
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set tmargin 0
set format x "%g"
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
    st.set_page_config(page_title="VCD Analyzer", layout="wide")
    st.title("VCD/IR Spectra Analyzer")

    if 'loaded_data' not in st.session_state:
        st.session_state['loaded_data'] = []

    # ==========================================
    # 1. サイドバー: 一括データ読み込み
    # ==========================================
    st.sidebar.header("📂 ファイル読み込み")
    
    uploaded_files = st.sidebar.file_uploader(
        "スペクトルファイルをアップロード (複数可)", 
        accept_multiple_files=True,
        type=['txt', 'csv', 'dat'],
        help="JASCO形式のテキストファイルなどに対応しています。"
    )
    
    if uploaded_files:
        data_list = []
        for f in uploaded_files:
            data, error_msg = load_spectral_data(f)
            if data:
                data_list.append(data)
            else:
                st.sidebar.error(f"{f.name}: {error_msg}")
        
        if data_list:
            st.session_state['loaded_data'] = data_list
            st.sidebar.success(f"{len(data_list)} ファイルを読み込みました。")

    loaded_data = st.session_state['loaded_data']

    if not loaded_data:
        st.info("👈 サイドバーからファイルをアップロードしてください。")
        return

    # ==========================================
    # タブ構成
    # ==========================================
    tab1, tab2 = st.tabs(["📊 個別解析 (Interactive)", "📈 比較プロット (Static / Report)"])

    # ==========================================
    # Tab 1: 個別解析 (Interactive / Plotly)
    # ==========================================
    with tab1:
        st.subheader("Single Spectrum Analysis")
        
        col_sel, col_peak = st.columns([1, 2])
        
        with col_sel:
            file_names = [d['filename'] for d in loaded_data]
            selected_idx = st.selectbox("解析するファイルを選択", range(len(file_names)), format_func=lambda x: file_names[x])
            selected_data = loaded_data[selected_idx]
            
            with st.expander("軸範囲設定", expanded=False):
                man_t1 = st.checkbox("手動設定を有効化", key="t1_man")
                c1, c2 = st.columns(2)
                t1_x_high = c1.number_input("X High (Left)", value=2000.0, key="t1_xh")
                t1_x_low = c2.number_input("X Low (Right)", value=800.0, key="t1_xl")
                
                t1_vcd_min, t1_vcd_max = None, None
                t1_ir_min, t1_ir_max = None, None
                
                if man_t1:
                    t1_vcd_max = c1.number_input("VCD Max", value=0.001, format="%.5f", key="t1_vmax")
                    t1_vcd_min = c2.number_input("VCD Min", value=-0.001, format="%.5f", key="t1_vmin")
                    t1_ir_max = c1.number_input("IR Max", value=1.5, key="t1_imax")
                    t1_ir_min = c2.number_input("IR Min", value=0.0, key="t1_imin")

        with col_peak:
            st.markdown("**ピーク検出設定**")
            do_peak = st.checkbox("ピーク検出マーカーを表示", value=True)
            peak_th = st.slider("ピークしきい値 (IR Abs)", 0.0, 2.0, 0.05, 0.01)

        if selected_data:
            x, ir, vcd = selected_data['x'], selected_data['ir'], selected_data['vcd']
            
            peaks, _ = find_peaks(ir, height=peak_th, distance=10)
            peak_x = x[peaks]
            peak_ir = ir[peaks]
            peak_vcd = vcd[peaks]

            fig_p = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.15, 
                subplot_titles=(f"VCD: {selected_data['filename']}", "IR Spectrum"),
                row_heights=[0.5, 0.5]
            )

            fig_p.add_trace(go.Scatter(
                x=x, y=vcd, mode='lines', name='VCD',
                line=dict(color='#00008B', width=1.5),
                hovertemplate="Wave: %{x:.1f}<br>VCD: %{y:.6f}<extra></extra>"
            ), row=1, col=1)

            fig_p.add_trace(go.Scatter(
                x=x, y=ir, mode='lines', name='IR',
                line=dict(color='#8B0000', width=1.5),
                hovertemplate="Wave: %{x:.1f}<br>Abs: %{y:.4f}<extra></extra>"
            ), row=2, col=1)
            
            if do_peak and len(peak_x) > 0:
                fig_p.add_trace(go.Scatter(
                    x=peak_x, y=peak_vcd, mode='markers', name='Peaks',
                    marker=dict(symbol='x', size=8, color='black'),
                    showlegend=False
                ), row=1, col=1)
                fig_p.add_trace(go.Scatter(
                    x=peak_x, y=peak_ir, mode='markers', name='Peaks',
                    marker=dict(symbol='circle', size=8, color='red'),
                    showlegend=False
                ), row=2, col=1)

            fig_p.update_layout(height=700, hovermode="x unified", showlegend=False)
            fig_p.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
            
            x_range = [t1_x_high, t1_x_low]
            fig_p.update_xaxes(range=x_range, row=1, col=1)
            fig_p.update_xaxes(range=x_range, row=2, col=1)
            
            if man_t1:
                fig_p.update_yaxes(range=[t1_vcd_min, t1_vcd_max], row=1, col=1)
                fig_p.update_yaxes(range=[t1_ir_min, t1_ir_max], row=2, col=1)
            
            fig_p.add_hline(y=0, line_width=1, line_color="black", row=1, col=1)
            
            st.plotly_chart(fig_p, use_container_width=True)

            if do_peak and len(peak_x) > 0:
                with st.expander("ピーク詳細データ"):
                    df_p = pd.DataFrame({"Wavenumber": peak_x, "IR Abs": peak_ir, "VCD Int": peak_vcd})
                    st.dataframe(df_p)

    # ==========================================
    # Tab 2: 比較プロット (Comparison)
    # ==========================================
    with tab2:
        st.subheader("Multi-Spectra Comparison")
        
        col_c_sel, col_c_opt = st.columns([1, 2])
        
        with col_c_sel:
            st.markdown("##### 表示データの選択")
            all_filenames = [d['filename'] for d in loaded_data]
            selected_files_compare = st.multiselect(
                "プロットするファイルを選択", 
                all_filenames, 
                default=all_filenames
            )
            target_data = [d for d in loaded_data if d['filename'] in selected_files_compare]
        
        with col_c_opt:
            st.markdown("##### グラフ全体設定")
            c_leg, c_dummy = st.columns(2)
            show_legend = c_leg.checkbox("凡例 (Legend) を表示", value=True)
            
            with st.expander("軸範囲の設定", expanded=False):
                c1, c2 = st.columns(2)
                t2_x_high = c1.number_input("X High", value=2000.0, key="t2_xh")
                t2_x_low = c2.number_input("X Low", value=800.0, key="t2_xl")
                
                man_t2 = st.checkbox("Y軸範囲固定", key="t2_man")
                t2_vcd_min, t2_vcd_max = None, None
                t2_ir_min, t2_ir_max = None, None
                if man_t2:
                    t2_vcd_max = c1.number_input("VCD Max", value=0.0005, format="%.5f", key="t2_vmax")
                    t2_vcd_min = c2.number_input("VCD Min", value=-0.0005, format="%.5f", key="t2_vmin")
                    t2_ir_max = c1.number_input("IR Max", value=1.0, key="t2_imax")
                    t2_ir_min = c2.number_input("IR Min", value=0.0, key="t2_imin")

        # --- 個別スタイル設定 ---
        st.markdown("---")
        st.markdown("##### 🎨 各プロットの詳細設定 (色・太さ・倍率)")
        
        # デフォルトカラーの準備
        default_colors = list(mcolors.TABLEAU_COLORS.values())
        plot_styles = {} # プロット時に使う設定を格納する辞書

        if target_data:
            # 多くの設定項目が並ぶのでExpanderに入れる
            with st.expander("設定パネルを開く", expanded=True):
                # 3カラムで順次表示
                cols = st.columns(3)
                for i, d in enumerate(target_data):
                    fname = d['filename']
                    default_c = default_colors[i % len(default_colors)]
                    
                    with cols[i % 3]:
                        st.markdown(f"**{fname}**")
                        # 色、太さ、倍率
                        c_col, c_wid, c_scl = st.columns([1, 1, 1])
                        p_color = c_col.color_picker("Color", value=default_c, key=f"c_{fname}")
                        p_width = c_wid.number_input("Width", value=1.5, step=0.5, key=f"w_{fname}")
                        p_scale = c_scl.number_input("Scale(x)", value=1.0, step=0.5, key=f"s_{fname}")
                        
                        plot_styles[fname] = {
                            'color': p_color,
                            'width': p_width,
                            'scale': p_scale
                        }

            # --- プロット作成 ---
            fig2, (ax_vcd, ax_ir) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                                 gridspec_kw={'height_ratios': [1, 1]})
            plt.subplots_adjust(hspace=0.05)
            
            for d in target_data:
                fname = d['filename']
                style = plot_styles[fname]
                
                # スタイル適用
                color = style['color']
                width = style['width']
                factor = style['scale']
                
                x_vals = d['x']
                vcd_vals = d['vcd'] * factor
                ir_vals = d['ir']
                
                label = f"{fname}"
                if factor != 1.0:
                    label += f" (x{factor})"
                
                ax_vcd.plot(x_vals, vcd_vals, color=color, linewidth=width, label=label)
                ax_ir.plot(x_vals, ir_vals, color=color, linewidth=width)
            
            ax_vcd.axhline(0, color='black', linewidth=0.8)
            ax_vcd.set_ylabel("VCD Intensity")
            ax_vcd.set_xlim(t2_x_high, t2_x_low)
            if man_t2: ax_vcd.set_ylim(t2_vcd_min, t2_vcd_max)
            
            if show_legend:
                ax_vcd.legend(loc='upper right', fontsize='small', framealpha=0.5)
            
            ax_ir.set_ylabel("Absorbance")
            ax_ir.set_xlabel("Wavenumber ($cm^{-1}$)")
            if man_t2: ax_ir.set_ylim(t2_ir_min, t2_ir_max)
            
            st.pyplot(fig2)
            
            # --- ダウンロード ---
            st.markdown("---")
            c1, c2 = st.columns(2)
            buf = io.BytesIO()
            fig2.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            c1.download_button("グラフ画像を保存 (PNG)", buf, "comparison_plot.png", "image/png")
            
            zip_dat = create_gnuplot_package(
                target_data, plot_styles,
                (t2_x_high, t2_x_low), 
                (t2_vcd_min, t2_vcd_max), 
                (t2_ir_min, t2_ir_max)
            )
            if zip_dat:
                c2.download_button("Gnuplotデータを保存 (.zip)", zip_dat, "gnuplot_data.zip", "application/zip")

        else:
            st.warning("表示するデータが選択されていません。")

if __name__ == "__main__":
    main()