import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def detect_header_row(file_path_or_buffer, encoding):
    """
    ファイル内の '[Data]' という行を探し、その次の行をヘッダー行として返す。
    見つからない場合は 0 を返す。
    """
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
    """
    指定されたエンコーディングで読み込みを試み、失敗したら他を試す
    """
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
            
            if enc != encoding:
                st.sidebar.warning(f"指定された {encoding} で読み込めなかったため、{enc} で読み込みました。")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            raise e
            
    raise last_error

# ---------------------------------------------------------
# アプリ設定
# ---------------------------------------------------------
st.set_page_config(page_title="DSC Style Plotter", layout="wide")

st.title("DSC Curve plotter")
st.markdown(f"""
使い方  (1) ファイルをアップロード（最初の表示はデモ用ファイル）　(2) 複数回のscanをした場合はプロット数を増やして（最大10個），分割プロット。(heating, coolingに分けるなど))
""")

# Matplotlibスタイル
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['font.size'] = 12

# ---------------------------------------------------------
# サイドバー：読み込み設定
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
    st.sidebar.warning(f"ファイルをアップロードするか、実行フォルダに {demo_file_path} を配置してください。")

if target_file:
    encoding_option = st.sidebar.selectbox("優先する文字コード", ["utf-8", "cp932", "shift_jis"], index=0)

    delimiter = st.sidebar.radio("区切り文字", [", (CSV)", "\\t (Tab)", "Space"], index=1)
    sep = "," if delimiter == ", (CSV)" else "\t" if delimiter == "\\t (Tab)" else r"\s+"

    default_header_row = detect_header_row(target_file, encoding_option)
    
    st.sidebar.subheader("ヘッダー設定")
    header_arg = st.sidebar.number_input(
        "ヘッダー(列名)がある行番号", 
        min_value=0, 
        value=default_header_row, 
        help="指定した行番号より前の行は無視されます。"
    )

    try:
        # データ読み込み
        df = load_data_robust(target_file, sep, header_arg, encoding_option)
        
        # 数値化処理
        if len(df) > 0:
            df_numeric = df.copy()
            for col in df_numeric.columns:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
            df_numeric = df_numeric.dropna(how='all')
            df = df_numeric.dropna().reset_index(drop=True)

        if df.empty:
            st.error("有効な数値データが見つかりませんでした。")
            st.stop()

        # 列選択
        columns = df.columns.tolist()
        st.sidebar.subheader("2. 列の選択")
        
        idx_x = 1 if len(columns) > 1 else 0
        idx_y = 2 if len(columns) > 2 else (1 if len(columns) > 1 else 0)

        x_col = st.sidebar.selectbox("X軸のデータ列", columns, index=idx_x)
        y_col = st.sidebar.selectbox("Y軸のデータ列", columns, index=idx_y)
        
        st.sidebar.subheader("3. グラフ全体の設定 (軸・ラベル)")
        y_label = st.sidebar.text_input("Y軸ラベル", "DSC (mW)")
        x_label = st.sidebar.text_input("X軸ラベル", "Temperature (℃)")

        st.sidebar.markdown("---")
        
        st.sidebar.markdown("**X軸の範囲 (Temperature)**")
        x_min_data = float(df[x_col].min())
        x_max_data = float(df[x_col].max())
        
        c_x1, c_x2 = st.sidebar.columns(2)
        x_min = c_x1.number_input("最小値 (X)", value=x_min_data, format="%.2f")
        x_max = c_x2.number_input("最大値 (X)", value=x_max_data, format="%.2f")
        
        st.sidebar.markdown("**Y軸の範囲 (DSC)**")
        use_manual_y = st.sidebar.checkbox("Y軸の範囲を手動で指定する", value=False)
        
        y_min, y_max = None, None
        if use_manual_y:
            y_min_data = float(df[y_col].min())
            y_max_data = float(df[y_col].max())
            
            c_y1, c_y2 = st.sidebar.columns(2)
            y_min = c_y1.number_input("最小値 (Y)", value=y_min_data, format="%.2f")
            y_max = c_y2.number_input("最大値 (Y)", value=y_max_data, format="%.2f")

        # ---------------------------------------------------------
        # レイアウト定義 (プレビュー画面を中央7割に制限)
        # ---------------------------------------------------------
        graph_container = st.container()
        
        st.markdown("---") 
        settings_container = st.container()

        # プロット設定用リスト
        plot_configs = []
        
        with settings_container:
            st.subheader("プロット設定")
            num_plots = st.number_input("プロットするDSC Curveの数", min_value=1, max_value=10, value=2)
            
            set_cols = st.columns(2) 
            for i in range(num_plots):
                with set_cols[i % 2]:
                    with st.expander(f"DSC Curve {i+1} の設定", expanded=True):
                        total_rows = len(df)
                        start_def = 0
                        end_def = total_rows
                        
                        if i == 0:
                            start_def, end_def = 30, 700
                        elif i == 1:
                            start_def, end_def = 800, 1750
                        
                        start_def = max(0, min(start_def, total_rows - 1))
                        end_def = max(0, min(end_def, total_rows))

                        default_color = "#FF0000" if i == 0 else "#0000FF" if i == 1 else "#000000"
                        offset_def = 0.0 if i < 2 else -0.5 * (i - 1)

                        c1, c2 = st.columns(2)
                        s_val = c1.number_input(f"開始行 (No.{i+1})", 0, total_rows, start_def, key=f"s_{i}")
                        e_val = c2.number_input(f"終了行 (No.{i+1})", 0, total_rows, end_def, key=f"e_{i}")
                        
                        c3, c4 = st.columns(2)
                        c_val = c3.color_picker(f"色 (No.{i+1})", default_color, key=f"c_{i}")
                        o_val = c4.number_input(f"Y軸オフセット (No.{i+1})", value=offset_def, step=0.1, format="%.2f", key=f"o_{i}")

                        plot_configs.append({
                            "label": f"DSC Curve {i+1}", "start": s_val, "end": e_val, "color": c_val, "offset": o_val
                        })

        # ---------------------------------------------------------
        # グラフ描画 (プレビューの幅を8割に調整)
        # ---------------------------------------------------------
        with graph_container:
            #st.subheader("プレビュー")
            
            # カラムを使って左右に1,2割ずつの余白を作り、中央に7割のメインカラムを配置
            spacer_left, main_col, spacer_right = st.columns([0.1, 0.7, 0.2])
            
            with main_col:
                # figsizeを(10, 6)から(8, 5)に変更して、見た目のバランスを調整
                fig, ax = plt.subplots(figsize=(8, 5))

                has_data = False
                for config in plot_configs:
                    subset = df.iloc[config["start"]:config["end"]]
                    if not subset.empty:
                        ax.plot(
                            subset[x_col], 
                            subset[y_col] + config["offset"], 
                            color=config["color"], 
                            label=config["label"],
                            linewidth=1.5
                        )
                        has_data = True

                ax.set_xlim(x_min, x_max)
                if use_manual_y and y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)
                
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                
                if has_data:
                    st.pyplot(fig)

                    # Gnuplotスクリプト作成ロジック（維持）
                    gnuplot_script = [f"# Generated by Streamlit DSC Plotter"]
                    source_name = uploaded_file.name if uploaded_file else 'demoDSC.txt'
                    gnuplot_script.append(f"# Source: {source_name}")
                    gnuplot_script.append("set terminal pngcairo size 800,600 enhanced font 'Arial,12'")
                    gnuplot_script.append("set output 'graph.png'")
                    gnuplot_script.append("set border 15 linewidth 1.5\nset tics scale 1.5\nset xtics out nomirror\nset ytics out nomirror")
                    gnuplot_script.append(f"set xlabel '{x_label}'\nset ylabel '{y_label}'")
                    gnuplot_script.append(f"set xrange [{x_min}:{x_max}]")
                    if use_manual_y and y_min is not None and y_max is not None:
                        gnuplot_script.append(f"set yrange [{y_min}:{y_max}]")
                    
                    plot_cmds = [f"'-' using 1:2 with lines lc rgb '{c['color']}' title '{c['label']}' lw 2" for c in plot_configs]
                    if plot_cmds:
                        gnuplot_script.append("plot " + ", \\\n     ".join(plot_cmds))
                        for config in plot_configs:
                            subset = df.iloc[config["start"]:config["end"]]
                            for _, row in subset.iterrows():
                                gnuplot_script.append(f"{row[x_col]} {row[y_col] + config['offset']}")
                            gnuplot_script.append("e")

                        st.download_button(
                            label="Gnuplotスクリプト(.plt)をダウンロード",
                            data="\n".join(gnuplot_script),
                            file_name="plot_script.plt",
                            mime="text/plain"
                        )
                else:
                    st.warning("表示できるデータがありません。")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

else:
    st.info(f"ファイルをアップロードするか、実行フォルダに {demo_file_path} を配置してください。")