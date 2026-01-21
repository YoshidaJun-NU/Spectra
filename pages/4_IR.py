import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.signal import find_peaks
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. 解析・計算用関数
# ---------------------------------------------------------
def trans_to_abs(y_trans):
    """透過率(%)を吸光度(Abs)に変換"""
    # 0や負の値の対数計算エラーを防ぐためにクリップ
    y_clamped = np.clip(y_trans, 1e-5, 150.0) 
    return 2.0 - np.log10(y_clamped)

def detect_peaks(x, y, mode="Transmittance (%)", prominence=1.0, distance=10):
    """ピーク検出関数"""
    # 透過率の場合は「谷」を探すため、データを反転させて「山」として検出する
    if mode == "Transmittance (%)":
        search_y = -1 * y
    else:
        search_y = y
    
    # ピーク検出実行
    peaks, properties = find_peaks(search_y, prominence=prominence, distance=distance)
    
    return x[peaks], y[peaks]

# ---------------------------------------------------------
# 2. データ読み込み
# ---------------------------------------------------------
def load_data(uploaded_files):
    data_list = []
    for f in uploaded_files:
        try:
            content = f.getvalue()
            # 文字コード判定（日本語が含まれる場合などの対策）
            for enc in ['utf-8', 'cp932', 'shift_jis', 'latin1']:
                try: 
                    text = content.decode(enc)
                    break
                except: 
                    continue
            
            lines = text.splitlines()
            x_unit, y_unit = "Wavenumber (cm⁻¹)", "Transmittance (%)"
            use_skip = 0
            
            # ヘッダー解析 (JASCO形式などを想定)
            for i, line in enumerate(lines):
                if 'XUNITS' in line:
                    val = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                    if val: x_unit = val
                if 'YUNITS' in line:
                    val = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                    if val: y_unit = val
                if 'XYDATA' in line:
                    use_skip = i + 1
                    break
            
            # CSV読み込み
            sep = ',' if f.name.lower().endswith('.csv') else None
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=use_skip, header=None, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] >= 2:
                data_list.append({
                    'label': f.name.rsplit('.', 1)[0],
                    'x': df.iloc[:, 0].values,
                    'y': df.iloc[:, 1].values, # 元データはそのまま保持
                    'x_unit': x_unit,
                    'y_unit': y_unit # 元データの単位
                })
        except Exception as e:
            st.error(f"{f.name} の読み込み失敗: {e}")
    return data_list

# ---------------------------------------------------------
# 3. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="IR Spectra Pro", layout="wide")
    st.title("IR Spectra Analyzer 🧪")

    # セッション状態の初期化
    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：共通設定 ---
    st.sidebar.header("📂 データ読み込み")
    files = st.sidebar.file_uploader("CSV/TXTファイルをアップロード", accept_multiple_files=True)
    if files:
        if st.sidebar.button("データを読み込む / リセット"):
            st.session_state['data_list'] = load_data(files)

    if not st.session_state['data_list']:
        st.info("👈 左側のサイドバーからスペクトルデータをアップロードしてください。")
        return

    # 全データのラベルリスト
    all_labels = [d['label'] for d in st.session_state['data_list']]

    # --- タブ構成 ---
    tab1, tab2 = st.tabs(["📊 データ解析 (Analysis)", "📈 重ね書き (Comparison)"])

    # =========================================================
    # タブ1: 個別解析モード (ピークサーチなど)
    # =========================================================
    with tab1:
        st.header("Single Spectrum Analysis")
        
        col_ctrl, col_plot = st.columns([1, 3])
        
        with col_ctrl:
            st.subheader("設定")
            # 対象データ選択
            target_label = st.selectbox("解析するデータ", all_labels)
            target_data = next((d for d in st.session_state['data_list'] if d['label'] == target_label), None)
            
            # 縦軸変換
            y_mode = st.radio("縦軸モード", ["Transmittance (%)", "Absorbance"], key="t1_mode")
            
            st.divider()
            st.markdown("**ピーク検出設定**")
            do_peak_search = st.checkbox("ピーク検出を有効にする", value=True)
            prominence = st.number_input("感度 (Prominence)", value=1.0, step=0.1, help="ピークの突出度。値を小さくすると細かいピークも拾います。")
            distance = st.number_input("最小間隔 (Distance)", value=10, min_value=1, help="検出するピーク同士の最小データ点間隔")

        if target_data:
            x = target_data['x']
            raw_y = target_data['y']
            
            # データ変換処理
            # 元データがAbsorbanceで、表示モードがTransmittanceの場合などの考慮が必要ですが、
            # ここでは「元データはTransmittanceである」と仮定して簡易実装します。
            # もし元データがAbsの場合は逆変換が必要ですが、IR機器の出力はT%が多い前提です。
            
            if y_mode == "Absorbance":
                # 元がT%なら変換、元がAbsならそのまま (簡易判定: 最大値が20以下なら元々Absかも?)
                if np.max(raw_y) > 20: 
                    y = trans_to_abs(raw_y)
                else:
                    y = raw_y
            else:
                # Transmittanceモード
                y = raw_y

            # Plotlyグラフ作成
            fig = go.Figure()

            # スペクトルプロット
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='lines', 
                name=target_label,
                line=dict(color='blue', width=1.5)
            ))

            # ピーク検出とプロット
            peak_x, peak_y = [], []
            if do_peak_search:
                peak_x, peak_y = detect_peaks(x, y, mode=y_mode, prominence=prominence, distance=int(distance))
                
                # ピークマーカー
                fig.add_trace(go.Scatter(
                    x=peak_x, y=peak_y,
                    mode='markers',
                    name='Peaks',
                    marker=dict(color='red', size=8, symbol='x'),
                    text=[f"{px:.1f} cm⁻¹" for px in peak_x],
                    hovertemplate='Wavenumber: %{x:.1f}<br>Value: %{y:.2f}'
                ))

            # レイアウト設定
            fig.update_layout(
                title=f"{target_label} ({y_mode})",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title=y_mode,
                xaxis=dict(autorange="reversed"), # IRスペクトルは通常 高波数->低波数
                hovermode="closest",
                height=600,
                template="simple_white"
            )

            with col_plot:
                st.plotly_chart(fig, use_container_width=True)

                # ピークリストの表示
                if do_peak_search and len(peak_x) > 0:
                    with st.expander("検出されたピーク一覧リスト"):
                        df_peaks = pd.DataFrame({
                            "Wavenumber (cm⁻¹)": peak_x,
                            f"Value ({y_mode})": peak_y
                        })
                        st.dataframe(df_peaks.style.format("{:.2f}"))


    # =========================================================
    # タブ2: 重ね書きモード (一括オフセット)
    # =========================================================
    with tab2:
        st.header("Multi-Spectra Comparison")
        
        col_c2, col_p2 = st.columns([1, 3])
        
        with col_c2:
            st.subheader("重ね書き設定")
            selected_labels = st.multiselect("表示するデータ", all_labels, default=all_labels)
            y_mode_comp = st.radio("縦軸モード", ["Transmittance (%)", "Absorbance"], key="t2_mode")
            
            st.divider()
            st.markdown("**オフセット設定**")
            offset_step = st.number_input("一括オフセット間隔", value=0.0, step=0.1, help="各スペクトルを指定した値ずつずらして表示します")
            reverse_stack = st.checkbox("積み上げ順を逆にする", value=False)

        with col_p2:
            if selected_labels:
                fig_comp = go.Figure()
                
                # 選択されたデータをループ処理
                plot_data_list = [d for d in st.session_state['data_list'] if d['label'] in selected_labels]
                
                if reverse_stack:
                    plot_data_list = plot_data_list[::-1]

                for i, item in enumerate(plot_data_list):
                    x_c = item['x']
                    raw_y_c = item['y']
                    
                    # 縦軸変換
                    if y_mode_comp == "Absorbance":
                        if np.max(raw_y_c) > 20: 
                            y_c = trans_to_abs(raw_y_c)
                        else:
                            y_c = raw_y_c
                    else:
                        y_c = raw_y_c
                    
                    # オフセット適用
                    # i=0 (1つ目) はオフセットなし、i=1 は offset_step * 1 ...
                    current_offset = i * offset_step
                    y_plotted = y_c + current_offset
                    
                    fig_comp.add_trace(go.Scatter(
                        x=x_c, y=y_plotted,
                        mode='lines',
                        name=f"{item['label']} (+{current_offset:.1f})",
                        hovertemplate=f"<b>{item['label']}</b><br>X: %{{x:.1f}}<br>Y: %{{y:.2f}}<extra></extra>"
                    ))

                # レイアウト設定
                fig_comp.update_layout(
                    title=f"Comparison ({y_mode_comp})",
                    xaxis_title="Wavenumber (cm⁻¹)",
                    yaxis_title=f"{y_mode_comp} (Offset applied)",
                    xaxis=dict(autorange="reversed"),
                    hovermode="x unified", # X座標を揃えて値を比較しやすくする
                    height=700,
                    template="simple_white"
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("表示するデータを選択してください。")

if __name__ == "__main__":
    main()