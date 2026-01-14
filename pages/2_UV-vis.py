import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.signal import find_peaks
import plotly.graph_objects as go

# ---------------------------------------------------------
# データ読み込み関数
# ---------------------------------------------------------
def load_spectrum_data(uploaded_file):
    encodings = ['utf-8', 'cp932', 'shift_jis', 'latin1']
    content = None
    for enc in encodings:
        try:
            content = uploaded_file.getvalue().decode(enc)
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        st.error(f"文字コードエラー: {uploaded_file.name}")
        return None

    try:
        lines = content.splitlines()
        start_line = 0
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                start_line = i + 1
                break
        
        sep = ',' if uploaded_file.name.lower().endswith('.csv') else None
        data_content = "\n".join(lines[start_line:])
        df = pd.read_csv(io.StringIO(data_content), sep=sep, header=None, engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] >= 2:
            return {"x": df.iloc[:, 0].values, "y": df.iloc[:, 1].values}
        return None
    except Exception as e:
        st.error(f"解析エラー ({uploaded_file.name}): {e}")
        return None

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Analyzer Pro", layout="wide")
    st.title("🧪 Advanced Spectra Analyzer")

    # --- サイドバー：1. ファイル読み込み ---
    st.sidebar.header("1. データの読み込み")
    uploaded_files = st.sidebar.file_uploader("CSV / TXT ファイル", accept_multiple_files=True, type=['txt', 'csv', 'dat'])

    if 'data_dict' not in st.session_state:
        st.session_state['data_dict'] = {}

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state['data_dict']:
                res = load_spectrum_data(f)
                if res:
                    st.session_state['data_dict'][f.name] = {
                        'x': res['x'], 'y': res['y'],
                        'conc': 1.0e-4, 'path': 1.0,
                        'color': None, 'width': 2, 'dash': 'solid' # 初期デザイン
                    }

    current_names = [f.name for f in uploaded_files] if uploaded_files else []
    st.session_state['data_dict'] = {k: v for k, v in st.session_state['data_dict'].items() if k in current_names}

    if not st.session_state['data_dict']:
        st.info("👈 左側のサイドバーからファイルをアップロードしてください。")
        return

    # --- サイドバー：2. グラフ全体のデザイン設定 ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. グラフのデザイン")
    show_legend = st.sidebar.checkbox("凡例を表示する", value=True)
    show_grid = st.sidebar.checkbox("目盛り線（グリッド）を表示する", value=True)
    y_mode = st.sidebar.radio("縦軸の単位", ["Abs.", "ε (Molar extinction coefficient)"])
    do_baseline = st.sidebar.checkbox("ベースライン補正")

    # --- サイドバー：3. 各線の個別設定 ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. 各線の個別設定")
    selected_files = st.sidebar.multiselect("表示するファイル", list(st.session_state['data_dict'].keys()), default=list(st.session_state['data_dict'].keys()))
    
    line_styles = {'実線': 'solid', '破線': 'dash', '点線': 'dot', '一点鎖線': 'dashdot'}

    for f_name in selected_files:
        with st.sidebar.expander(f"🎨 設定: {f_name}"):
            # 色・太さ・線種の入力
            st.session_state['data_dict'][f_name]['color'] = st.color_picker(f"線の色", key=f"col_{f_name}")
            st.session_state['data_dict'][f_name]['width'] = st.slider(f"線の太さ", 1, 10, 2, key=f"wid_{f_name}")
            st.session_state['data_dict'][f_name]['dash'] = st.selectbox(f"線種", list(line_styles.keys()), key=f"dash_{f_name}")
            
            if y_mode == "ε (Molar extinction coefficient)":
                st.session_state['data_dict'][f_name]['conc'] = st.number_input(f"濃度 C [mol/L]", value=st.session_state['data_dict'][f_name]['conc'], format="%.2e", key=f"c_{f_name}")
                st.session_state['data_dict'][f_name]['path'] = st.number_input(f"光路長 L [cm]", value=st.session_state['data_dict'][f_name]['path'], key=f"l_{f_name}")

    # --- メインエリア：グラフ描画 ---
    fig = go.Figure()
    y_label = "Absorbance"
    
    for f_name in selected_files:
        d = st.session_state['data_dict'][f_name]
        x, y = d['x'], d['y'].copy()

        if do_baseline:
            y -= (y[0] if x[0] > x[-1] else y[-1])
        
        if y_mode == "ε (Molar extinction coefficient)":
            y /= (d['conc'] * d['path'])
            y_label = "ε / (L·mol⁻¹·cm⁻¹)"

        fig.add_trace(go.Scatter(
            x=x, y=y, name=f_name,
            mode='lines',
            line=dict(
                color=d['color'],
                width=d['width'],
                dash=line_styles[d['dash']]
            )
        ))

    fig.update_layout(
        xaxis_title="Wavelength / nm",
        yaxis_title=y_label,
        showlegend=show_legend,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # グリッド表示設定
    fig.update_xaxes(showgrid=show_grid, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=show_grid, gridwidth=1, gridcolor='LightGray')

    st.plotly_chart(fig, use_container_width=True)

    st.info("📸 **画像を保存する場合**: グラフ右上のカメラアイコン『Download plot as a png』をクリックしてください。")

    # --- 解析・エクスポート ---
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("ピーク検出"):
            prom = st.slider("感度", 0.0, 1.0, 0.05, step=0.01)
            p_list = []
            for f_name in selected_files:
                d = st.session_state['data_dict'][f_name]
                y_p = d['y'].copy()
                if do_baseline: y_p -= (y_p[0] if d['x'][0] > d['x'][-1] else y_p[-1])
                if y_mode == "ε (Molar extinction coefficient)": y_p /= (d['conc'] * d['path'])
                peaks, _ = find_peaks(y_p, prominence=prom)
                for p in peaks:
                    p_list.append({"File": f_name, "Wavelength (nm)": d['x'][p], "Value": y_p[p]})
            if p_list:
                st.dataframe(pd.DataFrame(p_list))

    with col2:
        st.write("データダウンロード")
        # 簡易CSVエクスポート（表示データのみ）
        if selected_files:
            csv_data = pd.DataFrame({"Wavelength (nm)": st.session_state['data_dict'][selected_files[0]]['x']})
            st.download_button("CSVを保存", csv_data.to_csv(index=False).encode('utf-8'), "spectra_export.csv")

if __name__ == "__main__":
    main()