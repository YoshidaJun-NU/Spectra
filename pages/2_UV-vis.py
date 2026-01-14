import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.signal import find_peaks
import plotly.graph_objects as go

# ---------------------------------------------------------
# データ読み込み関数 (JASCO形式/CSV/TXT対応)
# ---------------------------------------------------------
def load_spectrum_data(uploaded_file):
    try:
        # ファイルの内容を読み込み
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        lines = stringio.readlines()
        
        # 1. データ開始行 (XYDATA) を自動検索
        start_line = 0
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                start_line = i + 1
                break
        
        # 2. 区切り文字の判定 (CSVならカンマ、TXTならタブまたはスペース)
        if uploaded_file.name.lower().endswith('.csv'):
            sep = ','
        else:
            sep = None # Noneにするとpandasが自動推論（タブや複数スペースに対応）

        # 3. データの読み込み
        stringio.seek(0) # ポインタを最初に戻す
        df = pd.read_csv(
            stringio,
            sep=sep,
            skiprows=start_line,
            header=None,
            engine='python'
        )
        
        # 数値データのみ抽出し、欠損値を削除
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] >= 2:
            return {"x": df.iloc[:, 0].values, "y": df.iloc[:, 1].values}
        else:
            return None
    except Exception as e:
        st.error(f"読み込みエラー ({uploaded_file.name}): {e}")
        return None

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="JASCO Spectra Analyzer", layout="wide")
    st.title("🧪 Absorption Spectra")
    st.markdown("JASCO形式の **CSV** および **TXT** ファイルを直接アップロードできます。")

    # 1. サイドバー：ファイルアップロード
    st.sidebar.header("1. データの読み込み")
    uploaded_files = st.sidebar.file_uploader(
        "ファイルをドロップしてください", 
        accept_multiple_files=True, 
        type=['txt', 'csv', 'dat']
    )

    if 'data_dict' not in st.session_state:
        st.session_state['data_dict'] = {}

    # アップロード時の処理
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state['data_dict']:
                res = load_spectrum_data(f)
                if res:
                    st.session_state['data_dict'][f.name] = {
                        'x': res['x'], 'y': res['y'],
                        'conc': 1.0e-4, 'path': 1.0
                    }

    # リストから消されたファイルをセッションからも消去
    current_names = [f.name for f in uploaded_files] if uploaded_files else []
    st.session_state['data_dict'] = {k: v for k, v in st.session_state['data_dict'].items() if k in current_names}

    if not st.session_state['data_dict']:
        st.info("👈 左側のサイドバーから .csv または .txt ファイルをアップロードしてください。")
        return

    # 2. サイドバー：表示・物理定数設定
    st.sidebar.markdown("---")
    st.sidebar.header("2. 表示設定")
    selected_files = st.sidebar.multiselect("表示するファイル", list(st.session_state['data_dict'].keys()), default=list(st.session_state['data_dict'].keys()))
    
    y_mode = st.sidebar.radio("縦軸の単位", ["Abs. (吸光度)", "ε (モル吸光係数)"])
    do_baseline = st.sidebar.checkbox("ベースライン補正 (長波長側を0にする)")

    if y_mode == "ε (モル吸光係数)":
        for f_name in selected_files:
            with st.sidebar.expander(f"定数: {f_name}"):
                st.session_state['data_dict'][f_name]['conc'] = st.number_input(f"濃度 C [mol/L]", value=st.session_state['data_dict'][f_name]['conc'], format="%.2e", key=f"c_{f_name}")
                st.session_state['data_dict'][f_name]['path'] = st.number_input(f"光路長 L [cm]", value=st.session_state['data_dict'][f_name]['path'], key=f"l_{f_name}")

    # 3. グラフ描画
    if y_mode == "ε (モル吸光係数)":
        st.subheader("Beer-Lambert Law")
        st.latex(r"\epsilon = \frac{A}{C \cdot L}")
    
    fig = go.Figure()
    all_processed_df = []

    for f_name in selected_files:
        data = st.session_state['data_dict'][f_name]
        x = data['x']
        y = data['y'].copy()

        # 前処理：ベースライン補正
        if do_baseline and len(y) > 0:
            y = y - y[0] if x[0] > x[-1] else y - y[-1] # 波長の並び順に関わらず長波長側を0に

        # 単位変換
        y_label = "Absorbance"
        if y_mode == "ε (モル吸光係数)":
            y = y / (data['conc'] * data['path'])
            y_label = "ε / (L·mol⁻¹·cm⁻¹)"

        fig.add_trace(go.Scatter(x=x, y=y, name=f_name, mode='lines'))
        all_processed_df.append(pd.DataFrame({"Wavelength (nm)": x, f"{f_name}": y}))

    fig.update_layout(
        xaxis_title="Wavelength / nm",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. 解析・ダウンロード
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("ピーク検出を実行"):
            prom = st.slider("ピーク感度 (Prominence)", 0.0, 1.0, 0.05, step=0.01)
            p_list = []
            for f_name in selected_files:
                # グラフと同じ処理を適用したy値を取得
                d = st.session_state['data_dict'][f_name]
                y_p = d['y'].copy()
                if do_baseline: y_p -= (y_p[0] if d['x'][0] > d['x'][-1] else y_p[-1])
                if y_mode == "ε (モル吸光係数)": y_p /= (d['conc'] * d['path'])
                
                peaks, _ = find_peaks(y_p, prominence=prom)
                for p in peaks:
                    p_list.append({"File": f_name, "Wavelength (nm)": d['x'][p], "Value": y_p[p]})
            if p_list:
                df_p = pd.DataFrame(p_list)
                st.dataframe(df_p)
    
    with col2:
        if all_processed_df:
            st.write("データ出力")
            # データの結合
            df_final = all_processed_df[0]
            for next_df in all_processed_df[1:]:
                df_final = pd.merge(df_final, next_df, on="Wavelength (nm)", how="outer")
            st.download_button("CSVをダウンロード", df_final.to_csv(index=False).encode('utf-8'), "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()