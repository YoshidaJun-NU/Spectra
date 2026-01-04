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
COLOR_DELTA = '#8B0000'  # Dark Red
COLOR_LAMBDA = '#00008B' # Dark Blue

# ---------------------------------------------------------
# 関数: ダミーデータ生成
# ---------------------------------------------------------
def generate_vcd_dummy(isomer_type='Delta'):
    x = np.linspace(800, 3000, 500)
    
    # ピーク定義 (中心, 幅, IR高さ, VCD符号基準)
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
# 関数: データ読み込み (エンコーディング対応版)
# ---------------------------------------------------------
def load_vcd_data(uploaded_file, sep_char, skip_rows, skip_footer, col_indices, encoding_type):
    """
    encoding_type: 'utf-8' or 'shift_jis' etc.
    """
    try:
        # engine='python' は skipfooter を使うために必須
        df = pd.read_csv(
            uploaded_file, 
            sep=sep_char, 
            skiprows=skip_rows, 
            skipfooter=skip_footer, 
            header=None, 
            engine='python',
            encoding=encoding_type # エンコーディングを指定
        )
        
        # 数値変換 (変換できない文字が含まれる行は削除)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        # 指定列が存在するか確認
        max_idx = max(col_indices.values())
        if max_idx >= df.shape[1]:
            st.error(f"{uploaded_file.name}: 指定された列番号 ({max_idx+1}) がデータ列数 ({df.shape[1]}) を超えています。")
            return None

        # データ抽出
        x = df.iloc[:, col_indices['x']].values
        ir = df.iloc[:, col_indices['ir']].values
        vcd = df.iloc[:, col_indices['vcd']].values
        
        return {'filename': uploaded_file.name, 'x': x, 'ir': ir, 'vcd': vcd}

    except Exception as e:
        st.error(f"読み込みエラー: {uploaded_file.name}\n{e}\n\n※ 文字コードを 'Shift-JIS' に変更して試してください。")
        return None

# ---------------------------------------------------------
# 関数: Gnuplotパッケージ作成
# ---------------------------------------------------------
def create_gnuplot_package(delta_list, lambda_list, x_lim, vcd_lim, ir_lim):
    all_x = []
    for d in delta_list + lambda_list: all_x.extend(d['x'])
    if not all_x: return None
    common_x = np.sort(np.unique(all_x))[::-1]
    
    df_out = pd.DataFrame({'Wavenumber': common_x})
    col_names = []

    # Delta
    for i, d in enumerate(delta_list):
        ir_i = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
        vcd_i = np.interp(common_x, d['x'][::-1], d['vcd'][::-1])
        lbl = f"Delta_{i+1}"
        df_out[f"{lbl}_IR"] = ir_i
        df_out[f"{lbl}_VCD"] = vcd_i
        col_names.append({'type': 'Delta', 'label': d['filename'], 'col_idx': len(df_out.columns)-1})

    # Lambda
    for i, d in enumerate(lambda_list):
        ir_i = np.interp(common_x, d['x'][::-1], d['ir'][::-1])
        vcd_i = np.interp(common_x, d['x'][::-1], d['vcd'][::-1])
        lbl = f"Lambda_{i+1}"
        df_out[f"{lbl}_IR"] = ir_i
        df_out[f"{lbl}_VCD"] = vcd_i
        col_names.append({'type': 'Lambda', 'label': d['filename'], 'col_idx': len(df_out.columns)-1})

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.5f')

    # Gnuplot Script
    plot_cmds = []
    curr = 2
    for item in col_names:
        c = COLOR_DELTA if item['type'] == 'Delta' else COLOR_LAMBDA
        t = item['label'].replace('_', '\\_')
        plot_cmds.append(f"'data.dat' u 1:{curr} axes x1y2 w l lc rgb '{c}' dt 2 notitle") 
        plot_cmds.append(f"'data.dat' u 1:{curr+1} axes x1y1 w l lc rgb '{c}' dt 1 title '{t} ({item['type']})'")
        curr += 2

    script = f"""
set terminal pngcairo size 800,600 font "Arial,12"
set output 'vcd_dual_axis.png'

set xrange [{x_lim[0]}:{x_lim[1]}]
set xlabel "Wavenumber (cm^{{-1}})"

set ylabel "VCD Intensity"
set yrange [{vcd_lim[0] if vcd_lim[0] else ":"}:{vcd_lim[1] if vcd_lim[1] else ":"}]
set ytics nomirror

set y2label "Absorbance"
set y2range [{ir_lim[0] if ir_lim[0] else ":"}:{ir_lim[1] if ir_lim[1] else ":"}]
set y2tics

set xzeroaxis lt 1 lc rgb "black" lw 1
set grid ls 1 lc rgb "gray" lw 0.5 dt 2
set key top right

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

    # --- サイドバー: データ読み込み設定 ---
    st.sidebar.header("1. データ設定")

    # 1. ファイル形式と文字コード
    st.sidebar.subheader("ファイル形式・文字コード")
    
    # 形式選択
    file_format = st.sidebar.radio(
        "ファイル拡張子:",
        ["CSV形式 (.csv)", "テキスト形式 (.txt / .dat)"]
    )
    if "CSV" in file_format:
        sep_char = ','
        st.sidebar.caption("※ カンマ (,) 区切り")
    else:
        sep_char = '\t'
        st.sidebar.caption("※ タブ (TAB) 区切り")

    # 文字コード選択（追加！）
    encoding_label = st.sidebar.radio(
        "文字コード (エラーが出る場合に変更):",
        ["UTF-8 (デフォルト)", "Shift-JIS (日本語Windows)"]
    )
    encoding_type = 'utf-8' if "UTF-8" in encoding_label else 'shift_jis'

    # 2. スキップ行数
    st.sidebar.subheader("行スキップ設定")
    c_skip1, c_skip2 = st.sidebar.columns(2)
    skip_row = c_skip1.number_input("ヘッダー (先頭)", min_value=0, value=0, help="ファイルの先頭から無視する行数")
    skip_footer = c_skip2.number_input("フッター (末尾)", min_value=0, value=0, help="ファイルの末尾にある説明書きなどを無視する行数")

    # 3. 列の割り当て
    st.sidebar.subheader("列の割り当て (列番号: 1始まり)")
    c1, c2, c3 = st.sidebar.columns(3)
    col_num_x = c1.number_input("波数 (X)", min_value=1, value=1)
    col_num_ir = c2.number_input("IR (Y2)", min_value=1, value=2)
    col_num_vcd = c3.number_input("VCD (Y1)", min_value=1, value=3)

    col_indices = {
        'x': col_num_x - 1, 
        'ir': col_num_ir - 1, 
        'vcd': col_num_vcd - 1
    }

    st.sidebar.markdown("---")
    st.sidebar.header("2. データアップロード")

    # ダミーデータ
    if st.sidebar.button("ダミーデータをロード"):
        dx, dir_, dvcd = generate_vcd_dummy('Delta')
        st.session_state['delta_data'] = [{'filename': 'Dummy_Delta', 'x': dx, 'ir': dir_, 'vcd': dvcd}]
        lx, lir, lvcd = generate_vcd_dummy('Lambda')
        st.session_state['lambda_data'] = [{'filename': 'Dummy_Lambda', 'x': lx, 'ir': lir, 'vcd': lvcd}]
        st.sidebar.success("ダミーデータをロードしました")

    # アップローダー
    up_delta = st.sidebar.file_uploader("Sample 1 (Delta) - 赤色", accept_multiple_files=True, type=['csv', 'txt', 'dat'], key="ud")
    if up_delta:
        temp_list = []
        for f in up_delta:
            res = load_vcd_data(f, sep_char, skip_row, skip_footer, col_indices, encoding_type)
            if res: temp_list.append(res)
        if temp_list: st.session_state['delta_data'] = temp_list

    up_lambda = st.sidebar.file_uploader("Sample 2 (Lambda) - 青色", accept_multiple_files=True, type=['csv', 'txt', 'dat'], key="ul")
    if up_lambda:
        temp_list = []
        for f in up_lambda:
            res = load_vcd_data(f, sep_char, skip_row, skip_footer, col_indices, encoding_type)
            if res: temp_list.append(res)
        if temp_list: st.session_state['lambda_data'] = temp_list

    # --- サイドバー: グラフ軸設定 ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. グラフ軸設定")
    
    col_x1, col_x2 = st.sidebar.columns(2)
    x_high = col_x1.number_input("X High (左)", value=3000.0)
    x_low = col_x2.number_input("X Low (右)", value=800.0)

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

    # --- プロット描画 ---
    delta_data = st.session_state['delta_data']
    lambda_data = st.session_state['lambda_data']

    if delta_data or lambda_data:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx() # 右軸

        ax1.axhline(0, color='black', linewidth=0.8, linestyle='-', zorder=1)

        def plot_item(ax_vcd, ax_ir, item, color, label_prefix):
            ax_vcd.plot(item['x'], item['vcd'], color=color, linestyle='-', linewidth=1.5, 
                        label=f"{label_prefix} VCD", zorder=3)
            ax_ir.plot(item['x'], item['ir'], color=color, linestyle=':', linewidth=1.2, alpha=0.7, 
                       label=f"{label_prefix} IR", zorder=2)

        for item in delta_data:
            plot_item(ax1, ax2, item, COLOR_DELTA, "Delta")
        for item in lambda_data:
            plot_item(ax1, ax2, item, COLOR_LAMBDA, "Lambda")

        ax1.set_xlabel("Wavenumber ($cm^{-1}$)", fontsize=12)
        ax1.set_ylabel("VCD Intensity", fontsize=12)
        ax2.set_ylabel("Absorbance", fontsize=12)

        ax1.set_xlim(x_high, x_low)
        if man_vcd: ax1.set_ylim(vcd_min, vcd_max)
        if man_ir: ax2.set_ylim(ir_min, ir_max)

        legend_elements = [
            Line2D([0], [0], color=COLOR_DELTA, lw=2, linestyle='-', label='Sample 1 (Delta) VCD'),
            Line2D([0], [0], color=COLOR_LAMBDA, lw=2, linestyle='-', label='Sample 2 (Lambda) VCD'),
            Line2D([0], [0], color='gray', lw=1.5, linestyle=':', label='IR / Absorbance (Ref)'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        st.pyplot(fig)

        st.markdown("---")
        c1, c2 = st.columns(2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        c1.download_button("画像 (PNG)", buf, "vcd_dual.png", "image/png")

        zip_dat = create_gnuplot_package(delta_data, lambda_data, (x_high, x_low), (vcd_min, vcd_max), (ir_min, ir_max))
        if zip_dat:
            c2.download_button("Gnuplotデータ (.zip)", zip_dat, "vcd_dual_gnuplot.zip", "application/zip")
            
    else:
        st.info("👈 サイドバーで文字コードを「Shift-JIS」にしてからファイルを読み込んでください。")

if __name__ == "__main__":
    main()