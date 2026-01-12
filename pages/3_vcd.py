import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 定数設定: 色コード
# ---------------------------------------------------------
COLOR_DELTA = '#8B0000'  # 暗めの赤 (Dark Red)
COLOR_LAMBDA = '#00008B' # 暗めの青 (Dark Blue)

# ---------------------------------------------------------
# 関数: ダミーデータ生成 (Delta/Lambda)
# ---------------------------------------------------------
def generate_vcd_dummy(isomer_type='Delta'):
    """
    Delta体またはLambda体のVCD/IRダミーデータを生成
    isomer_type: 'Delta' or 'Lambda'
    """
    x = np.linspace(800, 3000, 500)
    
    # ピーク定義 (中心波数, 幅, IR高さ, VCD符号基準)
    # VCD符号基準: Delta体を基準に定義し、Lambda体は反転させる
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
        # IRは常に正（共通）
        y_ir += h_ir * (width**2 / ((x - center)**2 + width**2))
        
        # VCD符号の決定
        # Deltaならそのまま、Lambdaなら反転
        actual_sign = sign_vcd if isomer_type == 'Delta' else -sign_vcd
        
        # VCD強度はIRの10%程度と仮定
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
        # [波数, IR, VCD] の3列を想定
        df = pd.read_csv(uploaded_file, sep=sep_char, skiprows=skip_rows, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] < 3:
            st.error(f"{uploaded_file.name}: 列数が不足しています (波数, IR, VCDが必要です)")
            return None

        # 0:波数, 1:IR, 2:VCD と仮定
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
    """
    データとプロットスクリプトをZIP化
    """
    # X軸の統合
    all_x = []
    for d in delta_list + lambda_list:
        all_x.extend(d['x'])
    if not all_x: return None
    
    common_x = np.sort(np.unique(all_x))[::-1] # 降順
    
    # データ結合
    df_out = pd.DataFrame({'Wavenumber': common_x})
    col_names = []

    # Delta体データの補間と格納
    for i, d in enumerate(delta_list):
        # np.interpはxが昇順である必要があるので [::-1] で反転して処理
        ir_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
        vcd_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1])
        
        label = f"Delta_{i+1}"
        df_out[f"{label}_IR"] = ir_interp
        df_out[f"{label}_VCD"] = vcd_interp
        col_names.append({'type': 'Delta', 'label': d['filename'], 'col_idx': len(df_out.columns)-1}) # col_idxはVCDの位置

    # Lambda体データの補間と格納
    for i, d in enumerate(lambda_list):
        ir_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
        vcd_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1])
        
        label = f"Lambda_{i+1}"
        df_out[f"{label}_IR"] = ir_interp
        df_out[f"{label}_VCD"] = vcd_interp
        col_names.append({'type': 'Lambda', 'label': d['filename'], 'col_idx': len(df_out.columns)-1})

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.5f')

    # Gnuplotスクリプト生成
    plot_cmds_vcd = []
    plot_cmds_ir = []
    
    # Gnuplot上のカラム番号 (1:Wavenumber, 2:D1_IR, 3:D1_VCD, ...)
    # df_outの列順: [Wavenumber, D1_IR, D1_VCD, L1_IR, L1_VCD...]
    current_col = 2
    
    for item in col_names:
        color = COLOR_DELTA if item['type'] == 'Delta' else COLOR_LAMBDA
        title = item['label'].replace('_', '\\_')
        
        # IR: current_col, VCD: current_col+1
        plot_cmds_ir.append(f"'data.dat' u 1:{current_col} w l lc rgb '{color}' title '{title} ({item['type']})'")
        plot_cmds_vcd.append(f"'data.dat' u 1:{current_col+1} w l lc rgb '{color}' notitle")
        current_col += 2

    # 範囲設定 (Noneの場合は自動)
    xr = f"[{x_lim[0]}:{x_lim[1]}]" # 高波数 -> 低波数
    yr_vcd = f"[{vcd_lim[0]}:{vcd_lim[1]}]" if vcd_lim[0] is not None else "[:]"
    yr_ir = f"[{ir_lim[0]}:{ir_lim[1]}]" if ir_lim[0] is not None else "[:]"

    script = f"""
set terminal pngcairo size 800,800 font "Arial,12"
set output 'vcd_plot.png'

set multiplot layout 2,1 margins 0.15, 0.95, 0.1, 0.95 spacing 0.05

# 共通設定
set xrange {xr}
set grid ls 1 lc rgb "gray" lw 0.5 dt 2

# 上段: VCD
set ylabel "VCD Intensity"
set yrange {yr_vcd}
set lmargin 12
set bmargin 0
set format x ""
set xzeroaxis lt 1 lc rgb "black" lw 1
plot {', '.join(plot_cmds_vcd)}

# 下段: IR
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
    st.set_page_config(page_title="VCD Plotter (Delta/Lambda)", layout="wide")
    st.title("VCD Spectra Plotter (Delta / Lambda)")
    
    if 'delta_data' not in st.session_state: st.session_state['delta_data'] = []
    if 'lambda_data' not in st.session_state: st.session_state['lambda_data'] = []

    # --- サイドバー: データ入力 ---
    st.sidebar.header("1. データソース")
    
    # ダミーデータ生成
    if st.sidebar.button("ダミーデータをロード (Sample 1 & 2)"):
        # Delta体 (Sample 1)
        d_df, d_x, d_ir, d_vcd = generate_vcd_dummy('Delta')
        st.session_state['delta_data'] = [{'filename': 'Dummy_Delta', 'x': d_x, 'ir': d_ir, 'vcd': d_vcd}]
        
        # Lambda体 (Sample 2)
        l_df, l_x, l_ir, l_vcd = generate_vcd_dummy('Lambda')
        st.session_state['lambda_data'] = [{'filename': 'Dummy_Lambda', 'x': l_x, 'ir': l_ir, 'vcd': l_vcd}]
        
        st.sidebar.success("ダミーデータを生成しました。")

    st.sidebar.markdown("---")
    
    # ファイルアップロード
    st.sidebar.subheader("ファイルから読み込み")
    st.sidebar.caption("形式: 1列目=波数, 2列目=IR, 3列目=VCD")
    
    sep_mode = st.sidebar.radio("区切り文字", ["カンマ (,)", "タブ (TAB)"])
    sep_char = ',' if "カンマ" in sep_mode else '\t'
    skip_row = st.sidebar.number_input("スキップ行数", 0, value=0)

    # Sample 1 (Delta) Upload
    up_delta = st.sidebar.file_uploader("Sample 1 (Delta体) - 暗赤色", accept_multiple_files=True, key="up_d")
    if up_delta:
        d_list = []
        for f in up_delta:
            res = load_vcd_data(f, sep_char, skip_row)
            if res: d_list.append(res)
        st.session_state['delta_data'] = d_list

    # Sample 2 (Lambda) Upload
    up_lambda = st.sidebar.file_uploader("Sample 2 (Lambda体) - 暗青色", accept_multiple_files=True, key="up_l")
    if up_lambda:
        l_list = []
        for f in up_lambda:
            res = load_vcd_data(f, sep_char, skip_row)
            if res: l_list.append(res)
        st.session_state['lambda_data'] = l_list

    # --- サイドバー: グラフ設定 ---
    st.sidebar.header("2. グラフ設定")
    
    # X軸
    st.sidebar.subheader("X軸 (波数)")
    col_x1, col_x2 = st.sidebar.columns(2)
    x_high = col_x1.number_input("High (左)", value=3000.0)
    x_low = col_x2.number_input("Low (右)", value=800.0)
    
    # Y軸 (VCD)
    st.sidebar.subheader("Y1軸 (VCD)")
    man_vcd = st.sidebar.checkbox("VCD範囲指定", value=False)
    vcd_min, vcd_max = None, None
    if man_vcd:
        c1, c2 = st.sidebar.columns(2)
        vcd_max = c1.number_input("VCD Max", value=0.1)
        vcd_min = c2.number_input("VCD Min", value=-0.1)

    # Y軸 (IR)
    st.sidebar.subheader("Y2軸 (IR)")
    man_ir = st.sidebar.checkbox("IR範囲指定", value=False)
    ir_min, ir_max = None, None
    if man_ir:
        c1, c2 = st.sidebar.columns(2)
        ir_max = c1.number_input("IR Max", value=1.0)
        ir_min = c2.number_input("IR Min", value=0.0)

    # --- メイン描画 ---
    delta_data = st.session_state['delta_data']
    lambda_data = st.session_state['lambda_data']
    
    if not delta_data and not lambda_data:
        st.info("👈 サイドバーの「ダミーデータをロード」ボタンを押すか、ファイルをアップロードしてください。")
        return

    # プロット作成
    fig, (ax_vcd, ax_ir) = plt.subplots(2, 1, sharex=True, figsize=(8, 9), 
                                        gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.05)

    # VCDプロット (上段)
    ax_vcd.axhline(0, color='black', linewidth=0.8, linestyle='-')
    
    for item in delta_data:
        ax_vcd.plot(item['x'], item['vcd'], color=COLOR_DELTA, linewidth=1.5, label=f"Delta: {item['filename']}")
    for item in lambda_data:
        ax_vcd.plot(item['x'], item['vcd'], color=COLOR_LAMBDA, linewidth=1.5, label=f"Lambda: {item['filename']}")
        
    ax_vcd.set_ylabel("VCD Intensity", fontsize=12)
    ax_vcd.tick_params(direction='in', top=True, right=True)
    
    if man_vcd:
        ax_vcd.set_ylim(vcd_min, vcd_max)

    # IRプロット (下段)
    for item in delta_data:
        ax_ir.plot(item['x'], item['ir'], color=COLOR_DELTA, linewidth=1.5)
    for item in lambda_data:
        ax_ir.plot(item['x'], item['ir'], color=COLOR_LAMBDA, linewidth=1.5)

    ax_ir.set_ylabel("Absorbance", fontsize=12)
    ax_ir.set_xlabel("Wavenumber ($cm^{-1}$)", fontsize=12)
    ax_ir.tick_params(direction='in', top=True, right=True)
    
    # 軸反転設定
    ax_ir.set_xlim(x_high, x_low)
    
    if man_ir:
        ax_ir.set_ylim(ir_min, ir_max)

    # 凡例 (カスタム凡例を作成してDelta/Lambdaの色を示す)
    legend_elements = [
        Line2D([0], [0], color=COLOR_DELTA, lw=2, label='Sample 1 (Delta)'),
        Line2D([0], [0], color=COLOR_LAMBDA, lw=2, label='Sample 2 (Lambda)')
    ]
    ax_vcd.legend(handles=legend_elements, loc='best')

    st.pyplot(fig)

    # --- ダウンロード ---
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    # PNG保存
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
    buf_png.seek(0)
    c1.download_button("グラフ画像 (PNG)", buf_png, "vcd_plot.png", "image/png")
    
    # Gnuplot保存
    zip_dat = create_gnuplot_package(
        delta_data, lambda_data, 
        (x_high, x_low), (vcd_min, vcd_max), (ir_min, ir_max)
    )
    if zip_dat:
        c2.download_button("Gnuplotデータ (.zip)", zip_dat, "vcd_gnuplot.zip", "application/zip")

if __name__ == "__main__":
    main()
