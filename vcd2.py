import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 定数・マッピング
# ---------------------------------------------------------
# 線種の選択肢とMatplotlibの記号のマッピング
LINE_STYLES = {
    "実線 (Solid)": '-',
    "破線 (Dashed)": '--',
    "点線 (Dotted)": ':',
    "一点鎖線 (Dash-dot)": '-.'
}

# ---------------------------------------------------------
# 関数: ダミーデータ生成
# ---------------------------------------------------------
def generate_vcd_dummy(isomer_type='Delta'):
    x = np.linspace(850, 2000, 500)
    
    peaks = [
        (1750, 20, 0.8, +1.0),
        (1650, 25, 0.3, -0.4),
        (1450, 15, 0.4, -0.5),
        (1200, 15, 0.5, +0.8),
        (1050, 10, 0.2, -0.3),
    ]
    
    y_ir = np.zeros_like(x)
    y_vcd = np.zeros_like(x)
    noise = np.random.normal(0, 0.003, len(x))
    
    for center, width, h_ir, sign_vcd in peaks:
        y_ir += h_ir * (width**2 / ((x - center)**2 + width**2))
        actual_sign = sign_vcd if isomer_type == 'Delta' else -sign_vcd
        y_vcd += (h_ir * 0.1 * actual_sign) * (width**2 / ((x - center)**2 + width**2))

    y_ir += np.abs(noise)
    y_vcd += noise * 0.1
    
    return x, y_ir, y_vcd

# ---------------------------------------------------------
# 関数: データ読み込み
# ---------------------------------------------------------
def load_vcd_data(uploaded_file, sep_char, skip_rows, skip_footer, col_indices, encoding_type):
    try:
        df = pd.read_csv(
            uploaded_file, 
            sep=sep_char, 
            skiprows=skip_rows, 
            skipfooter=skip_footer, 
            header=None, 
            engine='python',
            encoding=encoding_type
        )
        
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        max_idx = max(col_indices.values())
        if max_idx >= df.shape[1]:
            st.error(f"{uploaded_file.name}: 指定された列番号 ({max_idx+1}) がデータ列数 ({df.shape[1]}) を超えています。")
            return None

        x = df.iloc[:, col_indices['x']].values
        ir = df.iloc[:, col_indices['ir']].values
        vcd = df.iloc[:, col_indices['vcd']].values
        
        return {'filename': uploaded_file.name, 'x': x, 'ir': ir, 'vcd': vcd}

    except Exception as e:
        st.error(f"読み込みエラー: {uploaded_file.name}\n{e}\n\n※ 文字コード設定を確認してください。")
        return None

# ---------------------------------------------------------
# 関数: Gnuplotパッケージ作成 (スタイル反映版)
# ---------------------------------------------------------
def create_gnuplot_package(delta_list, lambda_list, x_lim, vcd_lim, ir_lim, scale_factor, legend_pos, styles):
    """
    styles: ユーザーが選択した色や設定を含む辞書
    """
    all_x = []
    for d in delta_list + lambda_list: all_x.extend(d['x'])
    if not all_x: return None
    common_x = np.sort(np.unique(all_x))[::-1]
    
    df_out = pd.DataFrame({'Wavenumber': common_x})
    col_names = []

    # Delta
    for i, d in enumerate(delta_list):
        ir_scaled = d['ir'] * scale_factor
        vcd_scaled = d['vcd'] * scale_factor
        ir_i = np.interp(common_x, d['x'][::-1], ir_scaled[::-1])
        vcd_i = np.interp(common_x, d['x'][::-1], vcd_scaled[::-1])
        lbl = f"Delta_{i+1}"
        df_out[f"{lbl}_IR"] = ir_i
        df_out[f"{lbl}_VCD"] = vcd_i
        col_names.append({'type': 'Delta', 'label': d['filename'], 'col_idx': len(df_out.columns)-1})

    # Lambda
    for i, d in enumerate(lambda_list):
        ir_scaled = d['ir'] * scale_factor
        vcd_scaled = d['vcd'] * scale_factor
        ir_i = np.interp(common_x, d['x'][::-1], ir_scaled[::-1])
        vcd_i = np.interp(common_x, d['x'][::-1], vcd_scaled[::-1])
        lbl = f"Lambda_{i+1}"
        df_out[f"{lbl}_IR"] = ir_i
        df_out[f"{lbl}_VCD"] = vcd_i
        col_names.append({'type': 'Lambda', 'label': d['filename'], 'col_idx': len(df_out.columns)-1})

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.5f')

    # Gnuplot legend setting
    key_setting = "set key top left" if "内部" in legend_pos else "set key outside right top"

    plot_cmds = []
    curr = 2
    for item in col_names:
        # ユーザー選択の色を反映
        c = styles['delta_color'] if item['type'] == 'Delta' else styles['lambda_color']
        t = item['label'].replace('_', '\\_')
        
        # NOTE: Gnuplotの線種(dt)は複雑になるため、ここでは標準的な VCD=実線(1), IR=点線(2) としています
        plot_cmds.append(f"'data.dat' u 1:{curr} axes x1y2 w l lc rgb '{c}' dt 2 notitle") 
        plot_cmds.append(f"'data.dat' u 1:{curr+1} axes x1y1 w l lc rgb '{c}' dt 1 title '{t} ({item['type']})'")
        curr += 2

    script = f"""
set terminal pngcairo size 800,600 font "Arial,12"
set output 'vcd_dual_axis.png'

set xrange [{x_lim[0]}:{x_lim[1]}]
set xlabel "Wavenumber (cm^{{-1}})"

set ylabel "VCD Intensity (x{scale_factor})"
set yrange [{vcd_lim[0] if vcd_lim[0] else ":"}:{vcd_lim[1] if vcd_lim[1] else ":"}]
set ytics nomirror

set y2label "Absorbance (x{scale_factor})"
set y2range [{ir_lim[0] if ir_lim[0] else ":"}:{ir_lim[1] if ir_lim[1] else ":"}]
set y2tics

set xzeroaxis lt 1 lc rgb "gray" lw 0.5
set grid ls 1 lc rgb "gray" lw 0.5 dt 2

{key_setting}

plot {', '.join(plot_cmds)}
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
    st.set_page_config(page_title="VCD Dual-Axis Plotter", layout="wide")
    st.title("VCD & IR Dual-Axis Plotter")
    
    if 'delta_data' not in st.session_state: st.session_state['delta_data'] = []
    if 'lambda_data' not in st.session_state: st.session_state['lambda_data'] = []

    # --- 1. データ設定 ---
    st.sidebar.header("1. データ設定")

    file_format = st.sidebar.radio("ファイル拡張子:", ["CSV形式 (.csv)", "テキスト形式 (.txt / .dat)"], index=1)
    sep_char = ',' if "CSV" in file_format else '\t'

    encoding_label = st.sidebar.radio("文字コード:", ["UTF-8", "Shift-JIS (日本語Windows)"], index=1)
    encoding_type = 'utf-8' if "UTF-8" in encoding_label else 'shift_jis'

    c_skip1, c_skip2 = st.sidebar.columns(2)
    skip_row = c_skip1.number_input("ヘッダー (先頭)", min_value=0, value=21)
    skip_footer = c_skip2.number_input("フッター (末尾)", min_value=0, value=47)

    scale_factor = st.sidebar.number_input("シグナル強度倍率 (n倍)", value=1.0, step=0.1, format="%.2f")

    st.sidebar.subheader("列の割り当て")
    c1, c2, c3 = st.sidebar.columns(3)
    col_num_x = c1.number_input("波数 (X)", min_value=1, value=1)
    col_num_ir = c2.number_input("IR (Y2)", min_value=1, value=2)
    col_num_vcd = c3.number_input("VCD (Y1)", min_value=1, value=3)
    col_indices = {'x': col_num_x - 1, 'ir': col_num_ir - 1, 'vcd': col_num_vcd - 1}

    st.sidebar.markdown("---")
    
    # --- 2. データアップロード ---
    st.sidebar.header("2. データアップロード")
    if st.sidebar.button("ダミーデータをロード"):
        dx, dir_, dvcd = generate_vcd_dummy('Delta')
        st.session_state['delta_data'] = [{'filename': 'Dummy_Delta', 'x': dx, 'ir': dir_, 'vcd': dvcd}]
        lx, lir, lvcd = generate_vcd_dummy('Lambda')
        st.session_state['lambda_data'] = [{'filename': 'Dummy_Lambda', 'x': lx, 'ir': lir, 'vcd': lvcd}]
        st.sidebar.success("ロード完了")

    up_delta = st.sidebar.file_uploader("Sample 1 (Delta)", accept_multiple_files=True, type=['csv', 'txt', 'dat'], key="ud")
    if up_delta:
        temp = [load_vcd_data(f, sep_char, skip_row, skip_footer, col_indices, encoding_type) for f in up_delta]
        st.session_state['delta_data'] = [t for t in temp if t]

    up_lambda = st.sidebar.file_uploader("Sample 2 (Lambda)", accept_multiple_files=True, type=['csv', 'txt', 'dat'], key="ul")
    if up_lambda:
        temp = [load_vcd_data(f, sep_char, skip_row, skip_footer, col_indices, encoding_type) for f in up_lambda]
        st.session_state['lambda_data'] = [t for t in temp if t]

    # --- 3. グラフ軸設定 ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. グラフ軸設定")
    
    col_x1, col_x2 = st.sidebar.columns(2)
    x_high = col_x1.number_input("X High (左)", value=2000.0)
    x_low = col_x2.number_input("X Low (右)", value=850.0)

    man_vcd = st.sidebar.checkbox("VCD範囲指定 (左軸)", value=False)
    vcd_min, vcd_max = None, None
    if man_vcd:
        c1, c2 = st.sidebar.columns(2)
        vcd_max = c1.number_input("VCD Max", value=0.1)
        vcd_min = c2.number_input("VCD Min", value=-0.1)

    man_ir = st.sidebar.checkbox("IR範囲指定 (右軸)", value=False)
    ir_min, ir_max = None, None
    if man_ir:
        c1, c2 = st.sidebar.columns(2)
        ir_max = c1.number_input("IR Max", value=1.0)
        ir_min = c2.number_input("IR Min", value=0.0)

    st.sidebar.markdown("#### 凡例 (Legend)")
    legend_pos = st.sidebar.radio("位置を選択:", ["内部 (左上)", "外部 (右側)"], index=0)

    # --- 4. グラフスタイル設定 (New!) ---
    st.sidebar.markdown("---")
    st.sidebar.header("4. グラフスタイル設定")

    # Delta Styles
    with st.sidebar.expander("Sample 1 (Delta) のスタイル", expanded=True):
        c1, c2 = st.columns(2)
        d_color = c1.color_picker("色 (Color)", '#8B0000', key='dc')
        d_width = c2.number_input("線幅", value=1.5, step=0.1, key='dw')
        d_style_vcd_key = st.selectbox("VCD 線種", list(LINE_STYLES.keys()), index=0, key='dsv')
        d_style_ir_key = st.selectbox("IR 線種", list(LINE_STYLES.keys()), index=2, key='dsi') # デフォルト点線

    # Lambda Styles
    with st.sidebar.expander("Sample 2 (Lambda) のスタイル", expanded=True):
        c1, c2 = st.columns(2)
        l_color = c1.color_picker("色 (Color)", '#00008B', key='lc')
        l_width = c2.number_input("線幅", value=1.5, step=0.1, key='lw')
        l_style_vcd_key = st.selectbox("VCD 線種", list(LINE_STYLES.keys()), index=0, key='lsv')
        l_style_ir_key = st.selectbox("IR 線種", list(LINE_STYLES.keys()), index=2, key='lsi')

    # スタイル情報を辞書にまとめる
    user_styles = {
        'delta_color': d_color,
        'delta_width': d_width,
        'delta_vcd_ls': LINE_STYLES[d_style_vcd_key],
        'delta_ir_ls': LINE_STYLES[d_style_ir_key],
        'lambda_color': l_color,
        'lambda_width': l_width,
        'lambda_vcd_ls': LINE_STYLES[l_style_vcd_key],
        'lambda_ir_ls': LINE_STYLES[l_style_ir_key]
    }

    # --- プロット描画 ---
    delta_data = st.session_state['delta_data']
    lambda_data = st.session_state['lambda_data']

    if delta_data or lambda_data:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # ゼロ線 (薄く変更)
        ax1.axhline(0, color='gray', linewidth=0.5, linestyle='-', zorder=1)

        def plot_item(ax_vcd, ax_ir, item, color, width, vcd_ls, ir_ls, scale):
            # VCD
            ax_vcd.plot(item['x'], item['vcd'] * scale, 
                        color=color, linestyle=vcd_ls, linewidth=width, zorder=3)
            # IR
            ax_ir.plot(item['x'], item['ir'] * scale, 
                       color=color, linestyle=ir_ls, linewidth=width, alpha=0.7, zorder=2)

        # Delta Plot
        for item in delta_data:
            plot_item(ax1, ax2, item, 
                      user_styles['delta_color'], user_styles['delta_width'],
                      user_styles['delta_vcd_ls'], user_styles['delta_ir_ls'],
                      scale_factor)
            
        # Lambda Plot
        for item in lambda_data:
            plot_item(ax1, ax2, item, 
                      user_styles['lambda_color'], user_styles['lambda_width'],
                      user_styles['lambda_vcd_ls'], user_styles['lambda_ir_ls'],
                      scale_factor)

        ax1.set_xlabel("Wavenumber ($cm^{-1}$)", fontsize=12)
        ax1.set_ylabel(f"VCD Intensity (x{scale_factor})", fontsize=12)
        ax2.set_ylabel(f"Absorbance (x{scale_factor})", fontsize=12)

        ax1.set_xlim(x_high, x_low)
        if man_vcd: ax1.set_ylim(vcd_min, vcd_max)
        if man_ir: ax2.set_ylim(ir_min, ir_max)

        # 凡例 (動的生成)
        legend_elements = [
            Line2D([0], [0], color=user_styles['delta_color'], lw=user_styles['delta_width'], 
                   linestyle=user_styles['delta_vcd_ls'], label='Sample 1 (Delta) VCD'),
            Line2D([0], [0], color=user_styles['delta_color'], lw=user_styles['delta_width'], 
                   linestyle=user_styles['delta_ir_ls'], label='Sample 1 (Delta) IR'),
            
            Line2D([0], [0], color=user_styles['lambda_color'], lw=user_styles['lambda_width'], 
                   linestyle=user_styles['lambda_vcd_ls'], label='Sample 2 (Lambda) VCD'),
            Line2D([0], [0], color=user_styles['lambda_color'], lw=user_styles['lambda_width'], 
                   linestyle=user_styles['lambda_ir_ls'], label='Sample 2 (Lambda) IR'),
        ]

        # 凡例位置
        if "内部" in legend_pos:
            ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        else:
            ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1.0), borderaxespad=0.)

        st.pyplot(fig)

        st.markdown("---")
        c1, c2 = st.columns(2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        c1.download_button("画像 (PNG)", buf, "vcd_dual.png", "image/png")

        zip_dat = create_gnuplot_package(
            delta_data, lambda_data, 
            (x_high, x_low), (vcd_min, vcd_max), (ir_min, ir_max),
            scale_factor, legend_pos, user_styles
        )
        if zip_dat:
            c2.download_button("Gnuplotデータ (.zip)", zip_dat, "vcd_dual_gnuplot.zip", "application/zip")
            
    else:
        st.info("👈 データの設定を行い、ファイルをアップロードしてください。")

if __name__ == "__main__":
    main()