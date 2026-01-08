import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import io
from scipy.signal import savgol_filter

# GUIなし環境での動作安定化
matplotlib.use('Agg')

# ---------------------------------------------------------
# 1. 関数定義: データ処理
# ---------------------------------------------------------
@st.cache_data
def generate_cd_dummy_data():
    x = np.linspace(190, 260, 150)
    y1 = 30 * np.exp(-((x - 192)**2) / 50) - 15 * np.exp(-((x - 222)**2) / 100) - 10 * np.exp(-((x - 208)**2) / 100)
    y2 = 10 * np.exp(-((x - 195)**2) / 80) - 12 * np.exp(-((x - 218)**2) / 200)
    return [
        {'label': 'Sample_A', 'x': x, 'y': y1 + np.random.normal(0, 0.2, len(x))},
        {'label': 'Sample_B', 'x': x, 'y': y2 + np.random.normal(0, 0.2, len(x))}
    ]

@st.cache_data
def load_data(uploaded_files, separator, skip_rows, has_header, col_x, col_y):
    data_list = []
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            sep_char = '\t' if separator == 'タブ (tab)' else ','
            # 指定された行をスキップして読み込み
            df = pd.read_csv(uploaded_file, sep=sep_char, skiprows=skip_rows, header=0 if has_header else None)
            
            # 数値以外のデータを除去
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] > max(col_x, col_y):
                x = df.iloc[:, col_x].values
                y = df.iloc[:, col_y].values
                # X軸でソート（波長が降順の場合があるため）
                idx = np.argsort(x)
                data_list.append({'label': uploaded_file.name.split('.')[0], 'x': x[idx], 'y': y[idx]})
        except Exception as e:
            st.error(f"エラー: {uploaded_file.name} の読み込みに失敗しました。{e}")
            continue
    return data_list

def apply_processing(data_list, smooth, use_offset, offset_wl, convert_to_de, params_dict):
    processed = []
    for item in data_list:
        x, y = item['x'].copy(), item['y'].copy()
        if smooth > 1:
            y = savgol_filter(y, window_length=smooth if smooth%2!=0 else smooth+1, polyorder=3)
        if use_offset:
            y -= y[np.abs(x - offset_wl).argmin()]
        if convert_to_de:
            p = params_dict.get(item['label'], {'c': 1e-5, 'l': 0.1})
            y /= (32980 * p['c'] * p['l'])
        processed.append({'label': item['label'], 'x': x, 'y': y})
    return processed

# ---------------------------------------------------------
# 2. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="高度なCDプロッター", layout="wide")
    st.title("🧬 CDスペクトル描画ツール")

    if 'raw_data' not in st.session_state: st.session_state['raw_data'] = []

    # --- サイドバー 1: データ管理 ---
    with st.sidebar:
        st.header("1. データ管理")
        
        files = st.file_uploader("CSV/TXTファイルをアップロード", accept_multiple_files=True)
        
        if files:
            # プレビュー機能：最初のファイルの生データを確認
            with st.expander("ファイルの生データを確認 (プレビュー)"):
                test_file = files[0]
                test_file.seek(0)
                lines = test_file.readlines()[:25] # 最初の25行
                st.code("".join([line.decode('utf-8', errors='ignore') for line in lines]))
                st.caption("※XYDATAの後の数値が何行目から始まっているか確認してください。")

            with st.expander("インポート詳細設定", expanded=True):
                sep = st.radio("区切り文字", ("タブ (tab)", "カンマ (comma)"))
                skip = st.number_input("読み飛ばす行数 (スキップ)", 0, 100, 19)
                head = st.checkbox("ヘッダー(列名)あり", False)
                st.markdown("---")
                col_x = st.number_input("X軸（波長）の列番号", 0, 10, 0, help="0から数えます")
                col_y = st.number_input("Y軸（データ）の列番号", 0, 10, 1, help="0から数えます")
            
            if st.button("設定を反映して読み込む"):
                st.session_state['raw_data'] = load_data(files, sep, skip, head, col_x, col_y)

        st.markdown("---")
        c1, c2 = st.columns(2)
        if c1.button("サンプル読み込み"): 
            st.session_state['raw_data'] = generate_cd_dummy_data()
        if c2.button("データをクリア"): 
            st.session_state['raw_data'] = []; st.rerun()

    if not st.session_state['raw_data']:
        st.info("👈 左側のメニューからファイルをアップロードし、「読み込む」ボタンを押してください。")
        return

    # --- サイドバー 2: 選択と単位変換 ---
    all_labels = [d['label'] for d in st.session_state['raw_data']]
    selected = st.sidebar.multiselect("表示する系列を選択", all_labels, default=all_labels)
    target_data = [d for d in st.session_state['raw_data'] if d['label'] in selected]

    convert_de = st.sidebar.checkbox("Δε (M⁻¹cm⁻¹) に変換")
    unit_params = {}
    if convert_de:
        st.sidebar.caption("濃度(M)と光路長(cm)を指定:")
        for d in target_data:
            with st.sidebar.expander(f"パラメータ: {d['label']}"):
                c = st.number_input("濃度 (M)", value=1.0e-5, format="%.2e", key=f"c_{d['label']}")
                l = st.number_input("光路長 (cm)", value=0.1, key=f"l_{d['label']}")
                unit_params[d['label']] = {'c': c, 'l': l}

    # --- サイドバー 3: プロット設定 ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. グラフのカスタマイズ")
    
    with st.sidebar.expander("軸・共通スタイルの設定"):
        tick_dir = st.radio("目盛りの向き", ["in (内向き)", "out (外向き)", "inout (両側)"], index=0, horizontal=True)
        t_dir = tick_dir.split()[0]
        show_top_right = st.checkbox("枠囲みを表示 (上・右側)", value=True)
        show_legend = st.checkbox("凡例を表示", value=True)
        grid_on = st.checkbox("グリッド線を表示", value=False)
        x_lab = st.text_input("X軸ラベル", "Wavelength (nm)")
        y_lab = st.text_input("Y軸ラベル", r"$\Delta\epsilon$ (M$^{-1}$cm$^{-1}$)" if convert_de else "Ellipticity (mdeg)")

    line_configs = {}
    st.sidebar.subheader("系列別スタイル")
    for i, d in enumerate(target_data):
        with st.sidebar.expander(f"スタイル: {d['label']}"):
            default_hex = mcolors.to_hex(plt.cm.tab10(i % 10))
            col = st.color_picker("線の色", default_hex, key=f"col_{d['label']}")
            width = st.slider("線の太さ", 0.5, 5.0, 2.0, 0.5, key=f"width_{d['label']}")
            style = st.selectbox("線種", ["- (実線)", "-- (破線)", ": (点線)", "-. (一点鎖線)"], key=f"style_{d['label']}")
            line_configs[d['label']] = {'color': col, 'lw': width, 'ls': style.split()[0]}

    # --- 描画実行 ---
    smooth = st.sidebar.slider("平滑化 (Smoothing)", 1, 31, 1, 2)
    processed_data = apply_processing(target_data, smooth, False, 350, convert_de, unit_params)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.tick_params(direction=t_dir, top=show_top_right, right=show_top_right, labelsize=11)
    if not show_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if grid_on: ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    
    for d in processed_data:
        cfg = line_configs[d['label']]
        ax.plot(d['x'], d['y'], label=d['label'], color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['ls'])

    ax.set_xlabel(x_lab, fontsize=13)
    ax.set_ylabel(y_lab, fontsize=13)
    if show_legend: ax.legend(frameon=False)
    
    st.pyplot(fig)

    # --- エクスポート ---
    st.markdown("### 📥 エクスポート")
    c1, c2, c3 = st.columns(3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    c1.download_button("PNG画像 (300dpi) を保存", buf.getvalue(), "plot.png", "image/png")
    tif_buf = io.BytesIO()
    fig.savefig(tif_buf, format="tiff", dpi=300, bbox_inches='tight')
    c2.download_button("TIFF画像を保存", tif_buf.getvalue(), "plot.tiff", "image/tiff")
    csv_data = pd.DataFrame({d['label']: pd.Series(d['y'], index=d['x']) for d in processed_data})
    c3.download_button("処理済みCSVを保存", csv_data.to_csv(), "processed_data.csv", "text/csv")

    plt.close(fig)

if __name__ == "__main__":
    main()