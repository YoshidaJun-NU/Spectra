import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 関数定義: ヘッダー位置の自動検出
# ---------------------------------------------------------
def detect_header_row(file_path_or_buffer, encoding):
    """
    ファイル内の '[Data]' という行を探し、その次の行をヘッダー行として返す。
    見つからない場合は 0 を返す。
    """
    header_row = 0
    try:
        # ファイルパスの場合
        if isinstance(file_path_or_buffer, str):
            with open(file_path_or_buffer, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
        # アップロードされたファイル(BytesIO)の場合
        else:
            file_path_or_buffer.seek(0)
            content = file_path_or_buffer.read().decode(encoding, errors='ignore')
            lines = content.splitlines()
            file_path_or_buffer.seek(0) # リセット

        for i, line in enumerate(lines):
            if '[Data]' in line:
                header_row = i + 1
                break
    except Exception:
        pass # エラー時はデフォルト0
    return header_row

def load_data_robust(file_path_or_buffer, sep, header, encoding):
    """
    指定されたエンコーディングで読み込みを試み、失敗したらUTF-8/cp932を切り替えて再試行する関数
    """
    encodings_to_try = [encoding, 'utf-8', 'cp932', 'shift_jis', 'utf-8-sig']
    # 重複を除去しつつ順序を保持
    encodings_to_try = sorted(set(encodings_to_try), key=encodings_to_try.index)
    
    last_error = None
    
    for enc in encodings_to_try:
        try:
            if isinstance(file_path_or_buffer, str):
                df = pd.read_csv(file_path_or_buffer, sep=sep, header=header, encoding=enc, engine='python')
            else:
                file_path_or_buffer.seek(0)
                df = pd.read_csv(file_path_or_buffer, sep=sep, header=header, encoding=enc, engine='python')
            
            # 成功したら、もし設定と違うエンコーディングだったら通知する
            if enc != encoding:
                st.sidebar.warning(f"設定された {encoding} で読み込めなかったため、{enc} で読み込みました。")
            
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            # その他のエラー（パースエラーなど）はそのまま再試行せず返すか、エラーとして扱う
            raise e
            
    # 全て失敗した場合
    raise last_error

# ---------------------------------------------------------
# 設定とスタイル
# ---------------------------------------------------------
st.set_page_config(page_title="DSC Style Plotter", layout="wide")

st.title("Scientific Graph Plotter (DSC Style)")
st.markdown("""
CSVやTXTデータを読み込み、行範囲を指定して複数の曲線をプロットします。
ファイルがアップロードされていない場合は、デモデータ(`demoDSC.txt`)を表示します。
""")

# Matplotlibのスタイル設定
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['font.size'] = 12

# ---------------------------------------------------------
# サイドバー：データ読み込み設定
# ---------------------------------------------------------
st.sidebar.header("1. データ読み込み設定")

# ファイルアップロード
uploaded_file = st.sidebar.file_uploader("ファイルを選択 (CSV or TXT)", type=['csv', 'txt'])

# デモファイルの確認
demo_file_path = "demoDSC.txt"
target_file = None

if uploaded_file is not None:
    target_file = uploaded_file
    st.sidebar.success("アップロードされたファイルを使用します。")
elif os.path.exists(demo_file_path):
    target_file = demo_file_path
    st.sidebar.info("デモファイル (demoDSC.txt) を読み込んでいます。")
else:
    st.sidebar.warning("ファイルをアップロードしてください。")

if target_file:
    # --- 読み込みオプション ---
    # デフォルトのエンコーディング選択
    encoding_option = st.sidebar.selectbox("優先する文字コード", ["cp932", "utf-8", "shift_jis"], index=0)

    # 区切り文字設定
    delimiter = st.sidebar.radio("区切り文字", [", (CSV)", "\\t (Tab)", "Space"], index=1)
    if delimiter == ", (CSV)":
        sep = ","
    elif delimiter == "\\t (Tab)":
        sep = "\t"
    else:
        sep = r"\s+"

    # ヘッダー行の自動検出
    default_header_row = detect_header_row(target_file, encoding_option)
    
    st.sidebar.subheader("ヘッダー設定")
    header_arg = st.sidebar.number_input(
        "ヘッダー(列名)がある行番号", 
        min_value=0, 
        value=default_header_row, 
        help="指定した行番号より前の行は無視されます。自動検出された場合は自動入力されます。"
    )

    # ---------------------------------------------------------
    # データ処理
    # ---------------------------------------------------------
    try:
        # ロバストな読み込み関数を使用
        df = load_data_robust(target_file, sep, header_arg, encoding_option)
        
        # 単位行（数値変換できない行）の削除
        if len(df) > 0:
            cols_to_check = df.columns
            # 念のためコピーを作成
            df_numeric = df.copy()
            for col in cols_to_check:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
            
            # 全ての列がNaNになるような行（ヘッダーの残りカスなど）や、
            # 重要な列がNaNになる行を削除
            df_numeric = df_numeric.dropna(how='all') # 全てNaNの行を削除
            
            # もとのdfを、数値変換できた行だけに絞る（こうすることで文字列列が消えるのを防ぐが、今回はプロット用なので数値化してしまって良い）
            df = df_numeric.dropna().reset_index(drop=True)

        st.sidebar.success(f"読み込み成功: {df.shape[0]}行, {df.shape[1]}列")
        
        if df.empty:
            st.error("有効な数値データが見つかりませんでした。区切り文字などを確認してください。")
            st.stop()

        # --- 列の選択 ---
        columns = df.columns.tolist()
        st.sidebar.subheader("2. 列の選択")
        
        # デフォルト選択の推測
        idx_x, idx_y = 0, 1
        for i, col in enumerate(columns):
            c_str = str(col).lower()
            if "temp" in c_str: idx_x = i
            if "dsc" in c_str or "heat" in c_str or "mw" in c_str: idx_y = i
        
        if idx_y == idx_x and len(columns) > 1:
             idx_y = 1 if idx_x == 0 else 0

        x_col = st.sidebar.selectbox("X軸のデータ列", columns, index=idx_x)
        y_col = st.sidebar.selectbox("Y軸のデータ列", columns, index=idx_y)

        # --- グラフ全体の設定 ---
        st.sidebar.subheader("グラフ全体の設定")
        
        x_min_def = float(df[x_col].min())
        x_max_def = float(df[x_col].max())
        
        x_range = st.sidebar.slider(
            "X軸の範囲",
            min_value=x_min_def,
            max_value=x_max_def,
            value=(x_min_def, x_max_def)
        )
        
        y_label = st.sidebar.text_input("Y軸ラベル", str(y_col))
        x_label = st.sidebar.text_input("X軸ラベル", str(x_col))

        # ---------------------------------------------------------
        # メインエリア：プロット詳細設定
        # ---------------------------------------------------------
        col_main, col_plot = st.columns([1, 2])

        with col_main:
            st.subheader("プロット設定")
            num_plots = st.number_input("プロットする曲線の数", min_value=1, max_value=10, value=2)

            plot_configs = []

            for i in range(num_plots):
                with st.expander(f"曲線 {i+1} の設定", expanded=(i < 2)):
                    default_color = "#FF0000" if i % 2 == 0 else "#0000FF"
                    if i >= 2: default_color = "#000000"
                    
                    offset_def = 0.0
                    if i >= 2: offset_def = -0.5 * (i - 1)

                    c1, c2 = st.columns(2)
                    start_row = c1.number_input(f"開始行 (No.{i+1})", 0, len(df), 0, key=f"s_{i}")
                    end_row = c2.number_input(f"終了行 (No.{i+1})", 0, len(df), len(df), key=f"e_{i}")
                    
                    c3, c4 = st.columns(2)
                    color = c3.color_picker(f"色 (No.{i+1})", default_color, key=f"c_{i}")
                    offset = c4.number_input(f"Y軸オフセット (No.{i+1})", value=offset_def, step=0.1, format="%.2f", key=f"o_{i}")

                    plot_configs.append({
                        "label": f"Plot {i+1}",
                        "start": start_row,
                        "end": end_row,
                        "color": color,
                        "offset": offset
                    })

        # ---------------------------------------------------------
        # 描画処理
        # ---------------------------------------------------------
        with col_plot:
            st.subheader("プレビュー")
            fig, ax = plt.subplots(figsize=(8, 6))

            has_data = False
            for config in plot_configs:
                subset = df.iloc[config["start"]:config["end"]]
                
                if not subset.empty:
                    ax.plot(
                        subset[x_col], 
                        subset[y_col] + config["offset"], 
                        color=config["color"], 
                        label=config["label"]
                    )
                    has_data = True

            ax.set_xlim(x_range)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if has_data:
                st.pyplot(fig)

                # ---------------------------------------------------------
                # Gnuplotスクリプト生成
                # ---------------------------------------------------------
                st.subheader("ダウンロード")
                
                gnuplot_script = []
                gnuplot_script.append(f"# Generated by Streamlit DSC Plotter")
                source_name = uploaded_file.name if uploaded_file else 'demoDSC.txt'
                gnuplot_script.append(f"# Source: {source_name}")
                gnuplot_script.append(f"set terminal pngcairo size 800,600 enhanced font 'Arial,12'")
                gnuplot_script.append(f"set output 'graph.png'")
                gnuplot_script.append(f"set border 15 linewidth 1.5")
                gnuplot_script.append(f"set tics scale 1.5")
                gnuplot_script.append(f"set xtics out nomirror")
                gnuplot_script.append(f"set ytics out nomirror")
                gnuplot_script.append(f"set xlabel '{x_label}'")
                gnuplot_script.append(f"set ylabel '{y_label}'")
                gnuplot_script.append(f"set xrange [{x_range[0]}:{x_range[1]}]")
                
                plot_cmds = []
                for i, config in enumerate(plot_configs):
                    c = config['color']
                    title = config['label']
                    plot_cmds.append(f"'-' using 1:2 with lines lc rgb '{c}' title '{title}' lw 2")
                
                if plot_cmds:
                    gnuplot_script.append("plot " + ", \\\n     ".join(plot_cmds))
                    
                    for config in plot_configs:
                        subset = df.iloc[config["start"]:config["end"]]
                        for _, row in subset.iterrows():
                            val_x = row[x_col]
                            val_y = row[y_col] + config["offset"]
                            gnuplot_script.append(f"{val_x} {val_y}")
                        gnuplot_script.append("e")

                    final_script = "\n".join(gnuplot_script)
                    
                    st.download_button(
                        label="Gnuplotスクリプト(.plt)をダウンロード",
                        data=final_script,
                        file_name="plot_script.plt",
                        mime="text/plain"
                    )
            else:
                st.warning("表示できるデータがありません。")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.info("設定を確認してください。")

else:
    st.info("ファイルをアップロードするか、demoDSC.txtを配置してください。")