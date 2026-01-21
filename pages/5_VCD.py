import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# ---------------------------------------------------------
# 定数設定: デフォルト色コード
# ---------------------------------------------------------
DEFAULT_COLOR_DELTA = '#8B0000'  # 暗めの赤 (Dark Red)
DEFAULT_COLOR_LAMBDA = '#00008B' # 暗めの青 (Dark Blue)

# ---------------------------------------------------------
# 関数: ダミーデータ生成 (Delta/Lambda)
# ---------------------------------------------------------
def generate_vcd_dummy(isomer_type='Delta'):
    """
    Delta体またはLambda体のVCD/IRダミーデータを生成
    """
    x = np.linspace(800, 3000, 500)
    
    # ピーク定義 (中心波数, 幅, IR高さ, VCD符号基準)
    peaks = [
        (1750, 20, 0.8, +1.0), # C=O stretch
        (1650, 25, 0.3, -0.4), # Amide I like
        (1450, 15, 0.4, -0.5), # CH bending
        (1200, 15, 0.5, +0.8),
        (1050, 10, 0.2, -0.3),
    ]
    
    y_ir = np.zeros_like(x)
    y_vcd = np.zeros_like(x)
    
    # ノイズ
    noise = np.random.normal(0, 0.003, len(x))
    
    for center, width, h_ir, sign_vcd in peaks:
        y_ir += h_ir * (width**2 / ((x - center)**2 + width**2))
        actual_sign = sign_vcd if isomer_type == 'Delta' else -sign_vcd
        y_vcd += (h_ir * 0.1 * actual_sign) * (width**2 / ((x - center)**2 + width**2))

    y_ir += np.abs(noise)
    y_vcd += noise * 0.1
    
    # データフレーム化（降順ソート）
    df = pd.DataFrame({'Wavenumber': x, 'IR': y_ir, 'VCD': y_vcd})
    df = df.sort_values('Wavenumber', ascending=False)
    
    return df, x, y_ir, y_vcd

# ---------------------------------------------------------
# 関数: ファイル読み込み
# ---------------------------------------------------------
def load_vcd_data(uploaded_file, sep_char, skip_rows):
    try:
        df = pd.read_csv(uploaded_file, sep=sep_char, skiprows=skip_rows, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] < 3:
            st.error(f"{uploaded_file.name}: 列数が不足しています")
            return None

        x = df.iloc[:, 0].values
        ir = df.iloc[:, 1].values
        vcd = df.iloc[:, 2].values
        
        return {'filename': uploaded_file.name, 'x': x, 'ir': ir, 'vcd': vcd}
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return None

# ---------------------------------------------------------
# 関数: Gnuplot用パッケージ作成
# ---------------------------------------------------------
def create_gnuplot_package(delta_list, lambda_list, x_lim, vcd_lim, ir_lim):
    all_x = []
    for d in delta_list + lambda_list:
        all_x.extend(d['x'])
    if not all_x: return None
    
    common_x = np.sort(np.unique(all_x))[::-1]
    
    df_out = pd.DataFrame({'Wavenumber': common_x})
    col_names = []

    for i, d in enumerate(delta_list):
        ir_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
        vcd_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1])
        label = f"Delta_{i+1}"
        df_out[f"{label}_IR"] = ir_interp
        df_out[f"{label}_VCD"] = vcd_interp
        col_names.append({'type': 'Delta', 'label': d['filename'], 'col_idx': len(df_out.columns)-1})

    for i, d in enumerate(lambda_list):
        ir_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
        vcd_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1])
        label = f"Lambda_{i+1}"
        df_out[f"{label}_IR"] = ir_interp
        df_out[f"{label}_VCD"] = vcd_interp
        col_names.append({'type': 'Lambda', 'label': d['filename'], 'col_idx': len(df_out.columns)-1})

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.5f')

    plot_cmds_vcd = []
    plot_cmds_ir = []
    current_col = 2
    
    for item in col_names:
        color = DEFAULT_COLOR_DELTA if item['type'] == 'Delta' else DEFAULT_COLOR_LAMBDA
        title = item['label'].replace('_', '\\_')
        plot_cmds_ir.append(f"'data.dat' u 1:{current_col} w l lc rgb '{color}' title '{title} ({item['type']})'")
        plot_cmds_vcd.append(f"'data.dat' u 1:{current_col+1} w l lc rgb '{color}' notitle")
        current_col += 2

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
    st.set_page_config(page_title="VCD Plotter Pro", layout="wide")
    st.title("VCD Spectra Plotter (Delta / Lambda)")
    
    if 'delta_data' not in st.session_state: st.session_state['delta_data'] = []
    if 'lambda_data' not in st.session_state: st.session_state['lambda_data'] = []

    # ==========================================
    # 1. サイドバー: データ読み込み (共通)
    # ==========================================
    st.sidebar.header("📂 データ読み込み")
    
    if st.sidebar.button("ダミーデータをロード (Sample 1 & 2)"):
        d_df, d_x, d_ir, d_vcd = generate_vcd_dummy('Delta')
        st.session_state['delta_data'] = [{'filename': 'Dummy_Delta', 'x': d_x, 'ir': d_ir, 'vcd': d_vcd}]
        l_df, l_x, l_ir, l_vcd = generate_vcd_dummy('Lambda')
        st.session_state['lambda_data'] = [{'filename': 'Dummy_Lambda', 'x': l_x, 'ir': l_ir, 'vcd': l_vcd}]
        st.sidebar.success("ダミーデータを生成しました。")

    st.sidebar.markdown("---")
    st.sidebar.subheader("ファイルから読み込み")
    st.sidebar.caption("形式: 1列目=波数, 2列目=IR, 3列目=VCD")
    
    sep_mode = st.sidebar.radio("区切り文字", ["カンマ (,)", "タブ (TAB)"])
    sep_char = ',' if "カンマ" in sep_mode else '\t'
    skip_row = st.sidebar.number_input("スキップ行数", 0, value=0)

    up_delta = st.sidebar.file_uploader("Sample 1 (Delta体) - Default:赤", accept_multiple_files=True, key="up_d")
    if up_delta:
        d_list = []
        for f in up_delta:
            res = load_vcd_data(f, sep_char, skip_row)
            if res: d_list.append(res)
        st.session_state['delta_data'] = d_list

    up_lambda = st.sidebar.file_uploader("Sample 2 (Lambda体) - Default:青", accept_multiple_files=True, key="up_l")
    if up_lambda:
        l_list = []
        for f in up_lambda:
            res = load_vcd_data(f, sep_char, skip_row)
            if res: l_list.append(res)
        st.session_state['lambda_data'] = l_list

    delta_data = st.session_state['delta_data']
    lambda_data = st.session_state['lambda_data']

    if not delta_data and not lambda_data:
        st.info("👈 サイドバーからデータをロードしてください。")
        return

    # ==========================================
    # タブ構成
    # ==========================================
    tab1, tab2 = st.tabs(["📊 個別解析 (Analysis)", "📈 重ね書き (Comparison)"])

    # ==========================================
    # Tab 1: 個別解析 (Single Spectrum)
    # ==========================================
    with tab1:
        st.subheader("Single Spectrum Analysis")
        
        # 1-1. 解析対象の選択リスト作成
        all_options = []
        for i, d in enumerate(delta_data):
            all_options.append({'label': f"[Delta] {d['filename']}", 'data': d, 'color': DEFAULT_COLOR_DELTA})
        for i, d in enumerate(lambda_data):
            all_options.append({'label': f"[Lambda] {d['filename']}", 'data': d, 'color': DEFAULT_COLOR_LAMBDA})
            
        col_sel, col_peak = st.columns([1, 2])
        
        with col_sel:
            # 修正ポイント: 直接オブジェクトを渡さず、インデックスを使用する
            option_indices = range(len(all_options))
            selected_idx = st.selectbox(
                "解析するデータを選択", 
                options=option_indices, 
                format_func=lambda i: all_options[i]['label']
            )
            selected_item = all_options[selected_idx]
            
            with st.expander("軸範囲の設定 (Tab1)", expanded=False):
                t1_x_high = st.number_input("X High (Left)", value=3000.0, key="t1_xh")
                t1_x_low = st.number_input("X Low (Right)", value=800.0, key="t1_xl")
                
                man_t1 = st.checkbox("Y軸範囲を指定", key="t1_man_y")
                t1_vcd_min, t1_vcd_max = None, None
                t1_ir_min, t1_ir_max = None, None
                
                if man_t1:
                    c1, c2 = st.columns(2)
                    t1_vcd_max = c1.number_input("VCD Max", value=0.1, key="t1_vmax")
                    t1_vcd_min = c2.number_input("VCD Min", value=-0.1, key="t1_vmin")
                    t1_ir_max = c1.number_input("IR Max", value=1.0, key="t1_imax")
                    t1_ir_min = c2.number_input("IR Min", value=0.0, key="t1_imin")

        with col_peak:
            st.markdown("**IRピーク検出設定**")
            do_peak = st.checkbox("IRのピーク位置をVCDに表示する", value=True)
            peak_th = st.slider("ピークしきい値 (IR Abs)", 0.0, 1.0, 0.1, 0.05)
            
        # 1-2. プロット作成
        if selected_item:
            data = selected_item['data']
            x, ir, vcd = data['x'], data['ir'], data['vcd']
            color = selected_item['color']
            
            peaks, _ = find_peaks(ir, height=peak_th, distance=20)
            peak_x = x[peaks]
            peak_ir = ir[peaks]
            peak_vcd = vcd[peaks]

            fig1, (ax1_vcd, ax1_ir) = plt.subplots(2, 1, sharex=True, figsize=(8, 8), 
                                                gridspec_kw={'height_ratios': [1, 1]})
            plt.subplots_adjust(hspace=0.05)
            
            ax1_vcd.axhline(0, color='black', lw=0.8)
            ax1_vcd.plot(x, vcd, color=color, lw=1.5, label="VCD")
            if do_peak:
                for px, py in zip(peak_x, peak_vcd):
                    ax1_vcd.axvline(x=px, color='gray', linestyle=':', alpha=0.6)
                    ax1_vcd.plot(px, py, 'x', color='black', markersize=6)

            ax1_vcd.set_ylabel("VCD Intensity")
            ax1_vcd.set_title(f"Analysis: {data['filename']}")
            if man_t1: ax1_vcd.set_ylim(t1_vcd_min, t1_vcd_max)

            ax1_ir.plot(x, ir, color=color, lw=1.5, label="IR")
            if do_peak:
                ax1_ir.plot(peak_x, peak_ir, 'o', color='red', markersize=5, alpha=0.7)
                for px in peak_x:
                     ax1_ir.axvline(x=px, color='gray', linestyle=':', alpha=0.6)

            ax1_ir.set_ylabel("Absorbance")
            ax1_ir.set_xlabel("Wavenumber ($cm^{-1}$)")
            ax1_ir.set_xlim(t1_x_high, t1_x_low)
            if man_t1: ax1_ir.set_ylim(t1_ir_min, t1_ir_max)

            st.pyplot(fig1)

            if do_peak and len(peak_x) > 0:
                with st.expander("検出されたピークリスト"):
                    df_peaks = pd.DataFrame({
                        "Wavenumber": peak_x,
                        "IR Abs": peak_ir,
                        "VCD Int": peak_vcd
                    })
                    st.dataframe(df_peaks.style.format("{:.4f}"))

    # ==========================================
    # Tab 2: 重ね書き (Comparison)
    # ==========================================
    with tab2:
        st.subheader("Multi-Spectra Comparison")
        
        with st.expander("軸範囲の設定 (Tab2)", expanded=False):
            col_x1, col_x2 = st.columns(2)
            t2_x_high = col_x1.number_input("X High (Left)", value=3000.0, key="t2_xh")
            t2_x_low = col_x2.number_input("X Low (Right)", value=800.0, key="t2_xl")
            
            man_t2 = st.checkbox("Y軸範囲を指定", key="t2_man_y")
            t2_vcd_min, t2_vcd_max = None, None
            t2_ir_min, t2_ir_max = None, None
            if man_t2:
                c1, c2 = st.columns(2)
                t2_vcd_max = c1.number_input("VCD Max", value=0.1, key="t2_vmax")
                t2_vcd_min = c2.number_input("VCD Min", value=-0.1, key="t2_vmin")
                t2_ir_max = c1.number_input("IR Max", value=1.0, key="t2_imax")
                t2_ir_min = c2.number_input("IR Min", value=0.0, key="t2_imin")

        with st.expander("グラフスタイル設定 (色・太さ・フォント・凡例)", expanded=True):
            c_font, c_leg = st.columns(2)
            font_size = c_font.number_input("文字サイズ (Font Size)", 8, 24, 12, key="t2_fontsize")
            show_legend = c_leg.checkbox("凡例を表示する (Show Legend)", value=True, key="t2_legend")
            st.divider()
            
            st.markdown("**Sample 1 (Delta体) のスタイル**")
            c_d1, c_d2 = st.columns(2)
            color_delta_cust = c_d1.color_picker("線の色", DEFAULT_COLOR_DELTA, key="c_delta")
            width_delta_cust = c_d2.number_input("線の太さ", 0.5, 5.0, 1.5, step=0.1, key="w_delta")
            
            st.divider()
            st.markdown("**Sample 2 (Lambda体) のスタイル**")
            c_l1, c_l2 = st.columns(2)
            color_lambda_cust = c_l1.color_picker("線の色", DEFAULT_COLOR_LAMBDA, key="c_lambda")
            width_lambda_cust = c_l2.number_input("線の太さ", 0.5, 5.0, 1.5, step=0.1, key="w_lambda")

        # Plot
        plt.rcParams.update({'font.size': font_size})
        fig2, (ax2_vcd, ax2_ir) = plt.subplots(2, 1, sharex=True, figsize=(8, 9), 
                                            gridspec_kw={'height_ratios': [1, 1]})
        plt.subplots_adjust(hspace=0.05)

        ax2_vcd.axhline(0, color='black', linewidth=0.8, linestyle='-')
        for item in delta_data:
            ax2_vcd.plot(item['x'], item['vcd'], color=color_delta_cust, linewidth=width_delta_cust)
        for item in lambda_data:
            ax2_vcd.plot(item['x'], item['vcd'], color=color_lambda_cust, linewidth=width_lambda_cust)
            
        ax2_vcd.set_ylabel("VCD Intensity", fontsize=font_size)
        ax2_vcd.tick_params(direction='in', top=True, right=True, labelsize=font_size)
        if man_t2: ax2_vcd.set_ylim(t2_vcd_min, t2_vcd_max)

        for item in delta_data:
            ax2_ir.plot(item['x'], item['ir'], color=color_delta_cust, linewidth=width_delta_cust)
        for item in lambda_data:
            ax2_ir.plot(item['x'], item['ir'], color=color_lambda_cust, linewidth=width_lambda_cust)

        ax2_ir.set_ylabel("Absorbance", fontsize=font_size)
        ax2_ir.set_xlabel("Wavenumber ($cm^{-1}$)", fontsize=font_size)
        ax2_ir.tick_params(direction='in', top=True, right=True, labelsize=font_size)
        ax2_ir.set_xlim(t2_x_high, t2_x_low)
        if man_t2: ax2_ir.set_ylim(t2_ir_min, t2_ir_max)

        if show_legend:
            legend_elements = [
                Line2D([0], [0], color=color_delta_cust, lw=width_delta_cust, label='Sample 1 (Delta)'),
                Line2D([0], [0], color=color_lambda_cust, lw=width_lambda_cust, label='Sample 2 (Lambda)')
            ]
            ax2_vcd.legend(handles=legend_elements, loc='best', fontsize=font_size)

        st.pyplot(fig2)

        st.markdown("---")
        c1, c2 = st.columns(2)
        buf_png = io.BytesIO()
        fig2.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
        buf_png.seek(0)
        c1.download_button("グラフ画像 (PNG)", buf_png, "vcd_plot_comparison.png", "image/png")
        
        zip_dat = create_gnuplot_package(
            delta_data, lambda_data, 
            (t2_x_high, t2_x_low), (t2_vcd_min, t2_vcd_max), (t2_ir_min, t2_ir_max)
        )
        if zip_dat:
            c2.download_button("Gnuplotデータ (.zip)", zip_dat, "vcd_gnuplot.zip", "application/zip")

if __name__ == "__main__":
    main()