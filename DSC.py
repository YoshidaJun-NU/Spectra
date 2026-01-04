import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------
# 設定とスタイル
# ---------------------------------------------------------
st.set_page_config(page_title="DSC Style Plotter", layout="wide")

st.title("Scientific Graph Plotter (DSC Style)")
st.markdown("""
CSVやTXTデータを読み込み、行範囲を指定して複数の曲線をプロットします。
作成したグラフはGnuplot形式でダウンロード可能です。
""")

# Matplotlibのスタイル設定（論文調のきれいなグラフにする）
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['font.size'] = 12

# ---------------------------------------------------------
# サイドバー：データ読み込み
# ---------------------------------------------------------
st.sidebar.header("1. データ読み込み")
uploaded_file = st.sidebar.file_uploader("ファイルを選択 (CSV or TXT)", type=['csv', 'txt'])

delimiter = st.sidebar.radio("区切り文字を選択", [", (CSV)", "\\t (Tab)", "Space"], index=0)
if delimiter == ", (CSV)":
    sep = ","
elif delimiter == "\\t (Tab)":
    sep = "\t"
else:
    sep = r"\s+"

if uploaded_file is not None:
    try:
        # データの読み込み
        df = pd.read_csv(uploaded_file, sep=sep, engine='python')
        st.sidebar.success(f"読み込み成功: {df.shape[0]}行, {df.shape[1]}列")
        
        # 列の選択
        columns = df.columns.tolist()
        st.sidebar.subheader("2. 列の選択")
        x_col = st.sidebar.selectbox("X軸のデータ列", columns, index=0)
        y_col = st.sidebar.selectbox("Y軸のデータ列", columns, index=1 if len(columns) > 1 else 0)

        # 全体の設定
        st.sidebar.subheader("グラフ全体の設定")
        x_min_def = float(df[x_col].min())
        x_max_def = float(df[x_col].max())
        
        x_range = st.sidebar.slider(
            "X軸の範囲",
            min_value=x_min_def,
            max_value=x_max_def,
            value=(x_min_def, x_max_def)
        )
        
        y_label = st.sidebar.text_input("Y軸ラベル", "Heat Flow / mW")
        x_label = st.sidebar.text_input("X軸ラベル", "Temperature / °C")

        # ---------------------------------------------------------
        # メインエリア：プロット設定
        # ---------------------------------------------------------
        col_main, col_plot = st.columns([1, 2])

        with col_main:
            st.subheader("プロット設定")
            num_plots = st.number_input("プロットする曲線の数 (最大10)", min_value=1, max_value=10, value=2)

            plot_configs = []

            for i in range(num_plots):
                with st.expander(f"曲線 {i+1} の設定", expanded=(i < 2)):
                    # デフォルト値の設定ロジック
                    default_color = "#FF0000" # 赤
                    if i == 1:
                        default_color = "#0000FF" # 青
                    
                    # 行範囲のデフォルト
                    total_rows = len(df)
                    start_def = 0
                    end_def = total_rows
                    
                    # オフセットのデフォルト（3つ目以降は少しずらす提案）
                    offset_def = 0.0
                    if i >= 2:
                        offset_def = -1.0 * (i - 1) # 適当な初期値

                    c1, c2 = st.columns(2)
                    start_row = c1.number_input(f"開始行 (No.{i+1})", 0, total_rows, start_def, key=f"s_{i}")
                    end_row = c2.number_input(f"終了行 (No.{i+1})", 0, total_rows, end_def, key=f"e_{i}")
                    
                    c3, c4 = st.columns(2)
                    color = c3.color_picker(f"色 (No.{i+1})", default_color, key=f"c_{i}")
                    offset = c4.number_input(f"Y軸オフセット (No.{i+1})", value=offset_def, step=0.1, format="%.2f", key=f"o_{i}")

                    # 設定を保存
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

            # プロットループ
            for config in plot_configs:
                # データのスライス
                # ユーザー入力は行番号（0始まり）を想定
                subset = df.iloc[config["start"]:config["end"]]
                
                if not subset.empty:
                    ax.plot(
                        subset[x_col], 
                        subset[y_col] + config["offset"], 
                        color=config["color"], 
                        label=config["label"]
                    )

            # グラフの体裁
            ax.set_xlim(x_range)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            # 凡例は邪魔なら消せるようにしても良いが、一旦非表示またはシンプルに
            # ax.legend(frameon=False) 
            
            st.pyplot(fig)

            # ---------------------------------------------------------
            # Gnuplotスクリプト生成
            # ---------------------------------------------------------
            st.subheader("ダウンロード")
            
            # Gnuplotスクリプトの作成（データを埋め込む形式 "inline data"）
            gnuplot_script = []
            gnuplot_script.append(f"# Generated by Streamlit DSC Plotter")
            gnuplot_script.append(f"set terminal pngcairo size 800,600 enhanced font 'Arial,12'")
            gnuplot_script.append(f"set output 'graph.png'")
            gnuplot_script.append(f"set border 15 linewidth 1.5") # 枠線
            gnuplot_script.append(f"set tics scale 1.5") # 目盛りの長さ
            gnuplot_script.append(f"set xtics out nomirror") # もしくは in
            gnuplot_script.append(f"set ytics out nomirror") 
            gnuplot_script.append(f"set xlabel '{x_label}'")
            gnuplot_script.append(f"set ylabel '{y_label}'")
            gnuplot_script.append(f"set xrange [{x_range[0]}:{x_range[1]}]")
            
            # plotコマンドの構築
            plot_cmds = []
            for i, config in enumerate(plot_configs):
                # hexカラーをgnuplotの形式に
                c = config['color']
                title = config['label']
                # inline data '-' を使用
                plot_cmds.append(f"'-' using 1:2 with lines lc rgb '{c}' title '{title}' lw 2")
            
            gnuplot_script.append("plot " + ", \\\n     ".join(plot_cmds))
            
            # データブロックの追記
            for config in plot_configs:
                subset = df.iloc[config["start"]:config["end"]]
                # Xと (Y + Offset) を書き出す
                for _, row in subset.iterrows():
                    val_x = row[x_col]
                    val_y = row[y_col] + config["offset"]
                    gnuplot_script.append(f"{val_x} {val_y}")
                gnuplot_script.append("e") # エンドマーカー

            final_script = "\n".join(gnuplot_script)
            
            st.download_button(
                label="Gnuplotスクリプト(.plt)をダウンロード",
                data=final_script,
                file_name="plot_script.plt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.info("データ形式を確認してください。数値データが含まれている必要があります。")

else:
    st.info("サイドバーからファイルをアップロードしてください。")