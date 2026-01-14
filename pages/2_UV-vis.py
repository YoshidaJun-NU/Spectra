import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.signal import find_peaks
import plotly.graph_objects as go

# ---------------------------------------------------------
# データ読み込み関数
# ---------------------------------------------------------
def load_spectrum_data(uploaded_file, sep_type, skip_head, skip_foot):
    try:
        sep = ',' if sep_type == 'CSV' else None
        file_content = uploaded_file.getvalue().decode('utf-8')
        
        df = pd.read_csv(
            io.StringIO(file_content),
            sep=sep,
            skiprows=skip_head,
            skipfooter=skip_foot,
            header=None,
            engine='python',
            encoding='utf-8'
        )
        
        # 数値データのみを抽出
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] < 2:
            return None
        
        return {"x": df.iloc[:, 0].values, "y": df.iloc[:, 1].values}
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Analyzer", layout="wide")
    st.title("🧪 UV-Vis Spectra Analyzer")

    # 1. サイドバー：データ読み込み
    st.sidebar.header("1. データの読み込み")
    uploaded_files = st.sidebar.file_uploader(
        "CSV/TXTファイルを選択", 
        accept_multiple_files=True, 
        type=['txt', 'csv', 'dat']
    )

    sep_type = st.sidebar.radio("ファイル形式", ('CSV', 'TXT/TSV/DAT'))
    skip_head = st.sidebar.number_input("ヘッダー (行数)", value=0, min_value=0)
    skip_foot = st.sidebar.number_input("フッター (行数)", value=0, min_value=0)

    # セッション状態の初期化
    if 'data_dict' not in st.session_state:
        st.session_state['data_dict'] = {}

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state['data_dict']:
                res = load_spectrum_data(f, sep_type, skip_head, skip_foot)
                if res:
                    st.session_state['data_dict'][f.name] = {
                        'x': res['x'], 'y': res['y'],
                        'conc': 1.0e-4, 'path': 1.0
                    }

    # 選択されていないファイルの削除
    current_names = [f.name for f in uploaded_files] if uploaded_files else []
    st.session_state['data_dict'] = {k: v for k, v in st.session_state['data_dict'].items() if k in current_names}

    if not st.session_state['data_dict']:
        st.info("👈 左側のサイドバーからファイルをアップロードしてください。")
        return

    # 2. サイドバー：表示設定
    st.sidebar.markdown("---")
    st.sidebar.header("2. 表示設定")
    selected_files = st.sidebar.multiselect("表示するファイル", list(st.session_state['data_dict'].keys()), default=list(st.session_state['data_dict'].keys()))
    y_mode = st.sidebar.radio("縦軸の単位", ["Abs.", "ε (Molar extinction coefficient)"])
    
    do_baseline = st.sidebar.checkbox("ベースライン補正 (最長波長を0とする)")

    if y_mode == "ε (Molar extinction coefficient)":
        for f_name in selected_files:
            with st.sidebar.expander(f"定数入力: {f_name}"):
                st.session_state['data_dict'][f_name]['conc'] = st.number_input(f"濃度 C [mol/L]", value=st.session_state['data_dict'][f_name]['conc'], format="%.2e", key=f"c_{f_name}")
                st.session_state['data_dict'][f_name]['path'] = st.number_input(f"光路長 L [cm]", value=st.session_state['data_dict'][f_name]['path'], key=f"l_{f_name}")

    # 3. メイン：グラフ表示
    if y_mode == "ε (Molar extinction coefficient)":
        st.subheader("Beer-Lambert Law")
        st.latex(r"\epsilon = \frac{A}{C \cdot L}")
        
    fig = go.Figure()
    all_processed_data = []

    for f_name in selected_files:
        data = st.session_state['data_dict'][f_name]
        x, y = data['x'], data['y'].copy()

        if do_baseline:
            y = y - y[-1]

        y_label = "Absorbance"
        if y_mode == "ε (Molar extinction coefficient)":
            y = y / (data['conc'] * data['path'])
            y_label = "ε / (L·mol⁻¹·cm⁻¹)"

        fig.add_trace(go.Scatter(x=x, y=y, name=f_name, mode='lines'))
        all_processed_data.append(pd.DataFrame({"Wavelength (nm)": x, f"{f_name} ({y_label})": y}))

    fig.update_layout(xaxis_title="Wavelength / nm", yaxis_title=y_mode, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 4. 解析：ピーク検出
    st.markdown("---")
    if st.checkbox("ピーク検出を実行"):
        prominence = st.slider("感度 (Prominence)", 0.0, 1.0, 0.05, step=0.01)
        peak_results = []
        for f_name in selected_files:
            data = st.session_state['data_dict'][f_name]
            # 現在表示中のy軸データを再計算
            y_val = data['y'].copy()
            if do_baseline: y_val -= y_val[-1]
            if y_mode == "ε (Molar extinction coefficient)": y_val /= (data['conc'] * data['path'])
            
            peaks, _ = find_peaks(y_val, prominence=prominence)
            for p in peaks:
                peak_results.append({"File": f_name, "Wavelength (nm)": data['x'][p], "Value": y_val[p]})
        
        if peak_results:
            df_peaks = pd.DataFrame(peak_results)
            st.dataframe(df_peaks)
            st.download_button("ピークデータをダウンロード", df_peaks.to_csv(index=False).encode('utf-8'), "peaks.csv", "text/csv")

    # 5. データダウンロード
    if all_processed_data:
        df_final = all_processed_data[0]
        for next_df in all_processed_data[1:]:
            df_final = pd.merge(df_final, next_df, on="Wavelength (nm)", how="outer")
        
        st.download_button("処理済み全データをダウンロード", df_final.to_csv(index=False).encode('utf-8'), "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()