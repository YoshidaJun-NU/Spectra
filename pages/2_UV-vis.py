import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks

# ---------------------------------------------------------
# 定数・設定
# ---------------------------------------------------------
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ---------------------------------------------------------
# 関数：データ読み込み (改良：ヘッダー・フッター指定対応)
# ---------------------------------------------------------
def load_spectrum_data(uploaded_file, sep_type, skip_head, skip_foot):
    try:
        # 区切り文字の設定
        sep = ',' if sep_type == 'CSV' else None  # Noneにするとpandasが自動推論(txt/tab等)
        
        # 読み込み
        df = pd.read_csv(
            uploaded_file,
            sep=sep,
            skiprows=skip_head,
            skipfooter=skip_foot,
            header=None,
            engine='python',
            encoding='utf-8'
        )
        
        # 数値データのみを抽出（文字列混入対策）
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] < 2:
            return None
        
        # 1列目をx(波長)、2列目をy(Abs)とする
        return {"x": df.iloc[:, 0].values, "y": df.iloc[:, 1].values}
    except Exception as e:
        st.error(f"ファイル読み込みエラー ({uploaded_file.name}): {e}")
        return None

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="UV-Vis Spectra Analyzer", layout="wide")
    st.title("🧪 Absorption Specta (episilon)")

    # --- サイドバー：1. ファイルアップロード ---
    st.sidebar.header("1. データ読み込み")
    uploaded_files = st.sidebar.file_uploader(
        "CSV または TXT ファイルを選択", 
        accept_multiple_files=True, 
        type=['txt', 'csv', 'dat']
    )

    st.sidebar.subheader("読み込み設定")
    sep_type = st.sidebar.radio("ファイル形式", ('CSV', 'TXT/TSV/DAT'))
    skip_head = st.sidebar.number_input("ヘッダー (行数)", value=0, min_value=0)
    skip_foot = st.sidebar.number_input("フッター (行数)", value=0, min_value=0)

    # セッション状態の初期化
    if 'data_dict' not in st.session_state:
        st.session_state['data_dict'] = {}

    # アップロードされたファイルを処理
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state['data_dict']:
                res = load_spectrum_data(f, sep_type, skip_head, skip_foot)
                if res:
                    st.session_state['data_dict'][f.name] = {
                        'x': res['x'],
                        'y': res['y'],
                        'conc': 1.0,  # デフォルト濃度
                        'path': 1.0   # デフォルト光路長 (cm)
                    }

    # --- サイドバー：2. 表示選択とパラメータ入力 ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. 表示設定と物理定数")
    
    all_filenames = list(st.session_state['data_dict'].keys())
    selected_files = st.sidebar.multiselect("表示するファイルを選択", all_filenames, default=all_filenames)

    # 縦軸のモード選択
    y_mode = st.sidebar.radio("縦軸の単位", ["Abs.", "ε (Molar extinction coefficient)"])
    
    # 選択されたファイルごとに濃度と光路長を設定
    if y_mode == "ε (Molar extinction coefficient)":
        st.sidebar.info("各サンプルの濃度 C (mol/L) を入力してください。")
        for f_name in selected_files:
            with st.sidebar.expander(f"定数: {f_name}"):
                st.session_state['data_dict'][f_name]['conc'] = st.number_input(
                    f"濃度 C [mol/L]", value=1.0e-4, format="%.2e", key=f"c_{f_name}")
                st.session_state['data_dict'][f_name]['path'] = st.number_input(
                    f"光路長 L [cm]", value=1.0, step=0.1, key=f"l_{f_name}")

    # --- メイン表示エリア ---
    if not selected_files:
        st.info("👈 左側のサイドバーからファイルをアップロードし、表示するデータを選択してください。")
        return

    # グラフの作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 計算式の説明
    if y_mode == "ε (Molar extinction coefficient)":
        st.latex(r"\text{Abs} = \epsilon \cdot C \cdot L \implies \epsilon = \frac{\text{Abs}}{C \cdot L}")

    for i, f_name in enumerate(selected_files):
        data = st.session_state['data_dict'][f_name]
        x = data['x']
        y_abs = data['y']
        
        if y_mode == "ε (Molar extinction coefficient)":
            # ε = Abs / (C * L)
            y_plot = y_abs / (data['conc'] * data['path'])
            y_label_text = "Molar Extinction Coefficient ε / (L·mol⁻¹·cm⁻¹)"
        else:
            y_plot = y_abs
            y_label_text = "Absorbance"

        ax.plot(x, y_plot, label=f_name, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], linewidth=1.5)

    # グラフ装飾
    ax.set_xlabel("Wavelength / nm", fontsize=12)
    ax.set_ylabel(y_label_text, fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)

    # --- 解析：ピーク検出機能の統合 ---
    if st.checkbox("ピーク検出を表示"):
        peak_list = []
        for f_name in selected_files:
            data = st.session_state['data_dict'][f_name]
            y_val = data['y']
            if y_mode == "ε (Molar extinction coefficient)":
                y_val = y_val / (data['conc'] * data['path'])
            
            peaks, _ = find_peaks(y_val, prominence=0.01) # 感度は適宜調整
            for p in peaks:
                peak_list.append({
                    "ファイル名": f_name,
                    "ピーク波長 (nm)": data['x'][p],
                    y_mode: y_val[p]
                })
        
        if peak_list:
            st.dataframe(pd.DataFrame(peak_list))

if __name__ == "__main__":
    main()