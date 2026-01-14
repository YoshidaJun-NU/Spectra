import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
import plotly.graph_objects as go # Plotlyを追加

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
        sep = ',' if sep_type == 'CSV' else None # Noneにするとpandasが自動推論(txt/tab等)
        
        # 読み込み
        # ファイルの内容を一度メモリに読み込み、バイト文字列として渡す
        # これにより、skipfooterがエンジン='python'で動作する
        file_content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(
            io.StringIO(file_content), # バイト文字列をStringIOでラップ
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
    st.title("🧪 Absorption Spectra (ε)")

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
            # 新しいファイルか、設定変更があった場合のみ再読み込み
            file_id = f"{f.name}_{f.size}_{skip_head}_{skip_foot}_{sep_type}"
            if file_id not in st.session_state.get('loaded_file_ids', {}):
                res = load_spectrum_data(f, sep_type, skip_head, skip_foot)
                if res:
                    st.session_state['data_dict'][f.name] = {
                        'x': res['x'],
                        'y': res['y'],
                        'conc': 1.0e-4,  # デフォルト濃度
                        'path': 1.0    # デフォルト光路長 (cm)
                    }
                    if 'loaded_file_ids' not in st.session_state:
                        st.session_state['loaded_file_ids'] = {}
                    st.session_state['loaded_file_ids'][file_id] = f.name # 読み込み済みIDを記録
    
    # 削除されたファイルをdata_dictから除去
    current_uploaded_names = {f.name for f in uploaded_files}
    keys_to_delete = [key for key in st.session_state['data_dict'] if key not in current_uploaded_names]
    for key in keys_to_delete:
        del st.session_state['data_dict'][key]
    
    # --- サイドバー：2. 表示選択とパラメータ入力 ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. 表示設定と物理定数")
    
    all_filenames = list(st.session_state['data_dict'].keys())
    selected_files = st.sidebar.multiselect("表示するファイルを選択", all_filenames, default=all_filenames)

    # 縦軸のモード選択
    y_mode = st.sidebar.radio("縦軸の単位", ["Abs. (吸光度)", "ε (モル吸光係数)"])
    
    # 選択されたファイルごとに濃度と光路長を設定
    if y_mode == "ε (モル吸光係数)":
        st.sidebar.info("各サンプルの濃度 C (mol/L) と光路長 L (cm) を入力してください。")
        for f_name in selected_files:
            with st.sidebar.expander(f"定数: {f_name}"):
                # デフォルト値をセッション状態から取得または初期設定
                current_conc = st.session_state['data_dict'][f_name].get('conc', 1.0e-4)
                current_path = st.session_state['data_dict'][f_name].get('path', 1.0)

                st.session_state['data_dict'][f_name]['conc'] = st.number_input(
                    f"濃度 C [mol/L]", value=current_conc, format="%.2e", key=f"c_{f_name}")
                st.session_state['data_dict'][f_name]['path'] = st.number_input(
                    f"光路長 L [cm]", value=current_path, step=0.1, key=f"l_{f_name}")

    # --- サイドバー：3. 前処理 ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. 前処理")
    do_baseline = st.sidebar.checkbox("ベースライン補正 (最長波長を0とする)", value=False)
    st.sidebar.info("スペクトルの長波長側の吸収がない領域をゼロに補正します。")

    # --- メイン表示エリア ---
    if not selected_files:
        st.info("👈 左側のサイドバーからファイルをアップロードし、表示するデータを選択してください。")
        return

    # 計算式の説明
    if y_mode == "ε (モル吸光係数)":
        st.subheader("Beer-Lambertの法則")
        st.latex(r"\text{Abs} = \epsilon \cdot C \cdot L \implies \epsilon = \frac{\text{Abs}}{C \cdot L}")
        st.markdown(
            "Beer-Lambertの法則は、物質の吸光度（Abs）がその濃度（C）と光路長（L）に比例することを示します。\n"
            "ここで、ε（イプシロン）はモル吸光係数と呼ばれ、物質固有の定数です。"
        )
        st.markdown("光が溶液を透過する概念図:")
        # Beer-Lambert則の概念図を生成
        
        st.markdown("http://googleusercontent.com/image_generation_content/1

")


    # グラフの作成 (Plotlyを使用)
    fig = go.Figure()
    
    y_label_text = "" # 初期化
    all_processed_data = [] # ダウンロード用データフレーム格納リスト

    for i, f_name in enumerate(selected_files):
        data = st.session_state['data_dict'][f_name]
        x = data['x']
        y_abs = data['y']
        
        y_plot = y_abs
        if do_baseline:
            # 最長波長側の値をベースラインとして差し引く
            if len(y_plot) > 0:
                y_plot = y_plot - y_plot[-1] 
            
        if y_mode == "ε (モル吸光係数)":
            # ε = Abs / (C * L)
            # 濃度または光路長が0の場合のゼロ除算を防ぐ
            conc = data['conc'] if data['conc'] != 0 else 1e-9 
            path = data['path'] if data['path'] != 0 else 1e-9
            y_plot = y_plot / (conc * path)
            y_label_text = "Molar Extinction Coefficient ε / (L·mol⁻¹·cm⁻¹)"
        else:
            y_label_text = "Absorbance"

        fig.add_trace(go.Scatter(x=x, y=y_plot, mode='lines', 
                                 name=f_name, 
                                 line=dict(color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])))
        
        # ダウンロード用のデータに追加
        all_processed_data.append(pd.DataFrame({
            "Wavelength (nm)": x,
            f"{f_name}_{y_label_text}": y_plot
        }))

    # Plotlyグラフ装飾
    fig.update_layout(
        xaxis_title="Wavelength / nm",
        yaxis_title=y_label_text,
        hovermode="x unified", # ホバーでX軸を共有する表示
        height=600 # グラフの高さ調整
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 解析：ピーク検出機能の統合 ---
    st.subheader("4. スペクトル解析")
    if st.checkbox("ピーク検出を表示"):
        st.markdown("---")
        st.subheader("ピーク検出")
        # ピーク検出感度をスライダーで調整
        peak_prominence = st.slider("ピーク感度 (Prominence): ピークの相対的な高さの閾値", 0.0, 1.0, 0.05, step=0.01)
        st.info("Prominenceを上げると、小さなノイズによるピークが除去されます。")

        peak_list = []
        for f_name in selected_files:
            data = st.session_state['data_dict'][f_name]
            y_val = data['y']
            
            if do_baseline:
                if len(y_val) > 0:
                    y_val = y_val - y_val[-1]

            if y_mode == "ε (モル吸光係数)":
                conc = data['conc'] if data['conc'] != 0 else 1e-9 
                path = data['path'] if data['path'] != 0 else 1e-9
                y_val = y_val / (conc * path)
            
            # find_peaksにprominenceを渡す
            peaks, properties = find_peaks(y_val, prominence=peak_prominence) 
            
            for p in peaks:
                peak_list.append({
                    "ファイル名": f_name,
                    "ピーク波長 (nm)": data['x'][p],
                    y_label_text: y_val[p]
                })
        
        if peak_list:
            df_peaks = pd.DataFrame(peak_list)
            st.dataframe(df_peaks)
            
            # ピークデータをCSVでダウンロード
            csv_peaks = df_peaks.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="ピークデータをCSVでダウンロード",
                data=csv_peaks,
                file_name="detected_peaks.csv",
                mime="text/csv",
            )
        else:
            st.info("指定された感度ではピークが検出されませんでした。感度を調整してみてください。")

    # --- ダウンロード機能 ---
    st.subheader("5. データダウンロード")
    if all_processed_data:
        # すべての処理済みデータをマージしてダウンロード用に準備
        # 波長が完全に一致しない場合を考慮して、最初のdfを基準にマージ
        if len(all_processed_data) > 1:
            df_merged = all_processed_data[0]
            for i in range(1, len(all_processed_data)):
                df_merged = pd.merge(df_merged, all_processed_data[i], on="Wavelength (nm)", how="outer")
        else:
            df_merged = all_processed_data[0]

        csv_export = df_merged.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"現在の表示データ ({y_label_text}) をCSVでダウンロード",
            data=csv_export,
            file_name="processed_spectra.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()