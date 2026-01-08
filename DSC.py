import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def detect_header_row(file_path_or_buffer, encoding):
    header_row = 0
    try:
        if isinstance(file_path_or_buffer, str):
            with open(file_path_or_buffer, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
        else:
            file_path_or_buffer.seek(0)
            content = file_path_or_buffer.read().decode(encoding, errors='ignore')
            lines = content.splitlines()
            file_path_or_buffer.seek(0)

        for i, line in enumerate(lines):
            if '[Data]' in line:
                header_row = i + 1
                break
    except Exception:
        pass 
    return header_row

def load_data_robust(file_path_or_buffer, sep, header, encoding):
    encodings_to_try = [encoding, 'utf-8', 'cp932', 'shift_jis', 'utf-8-sig']
    encodings_to_try = sorted(set(encodings_to_try), key=encodings_to_try.index)
    last_error = None
    for enc in encodings_to_try:
        try:
            if isinstance(file_path_or_buffer, str):
                df = pd.read_csv(file_path_or_buffer, sep=sep, header=header, encoding=enc, engine='python')
            else:
                file_path_or_buffer.seek(0)
                df = pd.read_csv(file_path_or_buffer, sep=sep, header=header, encoding=enc, engine='python')
            return df
        except Exception as e:
            last_error = e
            continue
    raise last_error

# ---------------------------------------------------------
# アプリ設定
# ---------------------------------------------------------
st.set_page_config(page_title="DSC Style Plotter", layout="wide")
st.title("Scientific Graph Plotter (DSC)")

# ---------------------------------------------------------
# サイドバー：1. データ読み込み設定
# ---------------------------------------------------------
st.sidebar.header("1. データ読み込み設定")
uploaded_file = st.sidebar.file_uploader("ファイルを選択 (CSV or TXT)", type=['csv', 'txt'])

demo_file_path = "demoDSC.txt"
target_file = None

if uploaded_file is not None:
    target_file = uploaded_file
elif os.path.exists(demo_file_path):
    target_file = demo_file_path
else:
    st.sidebar.warning(f"ファイルをロードするか、{demo_file_path} を配置してください。")

if target_file:
    encoding_option = st.sidebar.selectbox("文字コード", ["utf-8", "cp932", "shift_jis"], index=0)
    delimiter = st.sidebar.radio("区切り文字", [", (CSV)", "\\t (Tab)", "Space"], index=1)
    sep = "," if delimiter == ", (CSV)" else "\t" if delimiter == "\\t (Tab)" else r"\s+"

    default_header_row = detect_header_row(target_file, encoding_option)
    header_arg = st.sidebar.number_input("ヘッダーの行番号", min_value=0, value=default_header_row)

    try:
        df = load_data_robust(target_file, sep, header_arg, encoding_option)
        if len(df) > 0:
            df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
            df = df_numeric.dropna().reset_index(drop=True)

        columns = df.columns.tolist()
        st.sidebar.subheader("2. 列の選択")
        idx_x = 1 if len(columns) > 1 else 0
        idx_y = 2 if len(columns) > 2 else (1 if len(columns) > 1 else 0)
        x_col = st.sidebar.selectbox("X軸のデータ列", columns, index=idx_x)
        y_col = st.sidebar.selectbox("Y軸のデータ列", columns, index=idx_y)
        
        # ---------------------------------------------------------
        # サイドバー：3. グラフ詳細設定（追加機能）
        # ---------------------------------------------------------
        st.sidebar.subheader("3. グラフのスタイル設定")
        
        # 追加：目盛の向き、線の太さ、文字の大きさ
        tick_dir = st.sidebar.radio("目盛の向き", ["in (内向き)", "out (外向き)"], index=0).split()[0]
        global_lw = st.sidebar.slider("線の太さ", 0.5, 5.0, 1.5, 0.5)
        global_font_size = st.sidebar.slider("文字の大きさ", 8, 24, 12, 1)

        st.sidebar.markdown("---")
        y_label = st.sidebar.text_input("Y軸ラベル", "DSC (mW)")
        x_label = st.sidebar.text_input("X軸ラベル", "Temperature (℃)")
        
        st.sidebar.markdown("**表示範囲設定**")
        c_x1, c_x2 = st.sidebar.columns(2)
        x_min = c_x1.number_input("最小値 (X)", value=float(df[x_col].min()))
        x_max = c_x2.number_input("最大値 (X)", value=float(df[x_col].max()))
        
        use_manual_y = st.sidebar.checkbox("Y軸の範囲を手動指定", value=False)
        y_min, y_max = None, None
        if use_manual_y:
            c_y1, c_y2 = st.sidebar.columns(2)
            y_min = c_y1.number_input("最小値 (Y)", value=float(df[y_col].min()))
            y_max = c_y2.number_input("最大値 (Y)", value=float(df[y_col].max()))

        # ---------------------------------------------------------
        # メインコンテンツレイアウト
        # ---------------------------------------------------------
        graph_container = st.container()
        st.markdown("---") 
        settings_container = st.container()

        plot_configs = []
        with settings_container:
            st.subheader("プロット設定")
            num_plots = st.number_input("プロットするDSC Curveの数", min_value=1, max_value=10, value=2)
            set_cols = st.columns(2) 
            for i in range(num_plots):
                with set_cols[i % 2]:
                    with st.expander(f"DSC Curve {i+1} の範囲・オフセット", expanded=True):
                        total_rows = len(df)
                        start_def = [30, 800][i] if i < 2 else 0
                        end_def = [700, 1750][i] if i < 2 else total_rows
                        
                        c1, c2 = st.columns(2)
                        s_val = c1.number_input(f"開始行 (No.{i+1})", 0, total_rows, start_def, key=f"s_{i}")
                        e_val = c2.number_input(f"終了行 (No.{i+1})", 0, total_rows, end_def, key=f"e_{i}")
                        
                        c3, c4 = st.columns(2)
                        c_val = c3.color_picker(f"色 (No.{i+1})", ["#FF0000", "#0000FF"][i] if i < 2 else "#000000", key=f"c_{i}")
                        o_val = c4.number_input(f"Y軸オフセット (No.{i+1})", value=0.0 if i < 2 else -0.5*(i-1), step=0.1, key=f"o_{i}")

                        plot_configs.append({"label": f"Curve {i+1}", "start": s_val, "end": e_val, "color": c_val, "offset": o_val})

        # ---------------------------------------------------------
        # グラフ描画（中央8割）
        # ---------------------------------------------------------
        with graph_container:
            st.subheader("プレビュー")
            spacer_l, main_col, spacer_r = st.columns([0.1, 0.8, 0.1])
            
            with main_col:
                # 動的なスタイル適用
                plt.rcParams['font.size'] = global_font_size
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # 目盛の向きと枠線の設定
                ax.tick_params(direction=tick_dir, top=True, right=True, width=1.2)
                
                has_data = False
                for config in plot_configs:
                    subset = df.iloc[config["start"]:config["end"]]
                    if not subset.empty:
                        ax.plot(
                            subset[x_col], subset[y_col] + config["offset"], 
                            color=config["color"], label=config["label"], linewidth=global_lw
                        )
                        has_data = True

                ax.set_xlim(x_min, x_max)
                if use_manual_y: ax.set_ylim(y_min, y_max)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                
                if has_data:
                    st.pyplot(fig)
                    
                    # Gnuplotスクリプト生成（省略・維持）
                    st.download_button(label="Gnuplotスクリプトを保存", data="...", file_name="plot.plt")
                else:
                    st.warning("データ範囲が空です。")

    except Exception as e:
        st.error(f"エラー: {e}")

# ---------------------------------------------------------
# 使い方（画面最下部に配置）
# ---------------------------------------------------------
st.divider()
with st.expander("📖 使い方とヒント", expanded=False):
    st.markdown("""
    ### 1. データ読み込み
    - CSVまたはタブ区切りのテキストをアップロードしてください。
    - `[Data]` 行を自動検出し、その次から数値を読み込みます。
    
    ### 2. プロットの分割
    - 昇温(Heating)と降温(Cooling)が混ざったファイルの場合、「プロットする数」を2以上にし、それぞれの「開始行・終了行」を指定することで別々の線として描画できます。
    
    ### 3. スタイルの調整
    - **目盛の向き**: 論文用には 'in'（内向き）が一般的です。
    - **オフセット**: 複数の曲線を上下にずらして比較したい場合に使用します。
    - **文字サイズ**: プレゼン用なら大きめ(16pt〜)、論文用なら(12pt〜)がおすすめです。
    """)

plt.close('all')