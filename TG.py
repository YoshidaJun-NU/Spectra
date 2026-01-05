import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ページ設定
st.set_page_config(page_title="TG/DTA Advanced Plotter", layout="wide")

st.title("📈 高機能 TG/DTA 解析・プロットツール")

# --- 1. データ読み込みセクション ---
st.sidebar.header("1. データファイルのアップロード")
uploaded_files = st.sidebar.file_uploader(
    "CSVまたはテキストファイルをアップロード (複数可)", 
    type=['csv', 'txt'], 
    accept_multiple_files=True
)

# データのキャッシュと整形
data_store = {}

if uploaded_files:
    st.sidebar.markdown("---")
    st.sidebar.subheader("データ列の指定")
    st.sidebar.info("ファイル内のどの列が温度、TG、DTAに対応するか指定してください（列番号：0始まり）。")
    
    # ユーザーに入力させる（デフォルト値は一般的な機器を想定）
    col_temp = st.sidebar.number_input("温度 (Temp) の列番号", value=0, min_value=0, step=1)
    col_tg = st.sidebar.number_input("重量 (TG %) の列番号", value=1, min_value=0, step=1)
    col_dta = st.sidebar.number_input("DTA (uV or deg) の列番号", value=2, min_value=0, step=1)
    
    # データ読み込み処理
    for uploaded_file in uploaded_files:
        try:
            # 読み込み (ヘッダーがある場合とない場合を簡易判定)
            df = pd.read_csv(uploaded_file, header=None, skiprows=1) # 1行目をスキップする設定（適宜調整）
            
            # 必要なデータを抽出・リネーム
            temp = df.iloc[:, col_temp].values
            tg = df.iloc[:, col_tg].values
            dta = df.iloc[:, col_dta].values
            
            # ソート（温度順）
            sort_idx = np.argsort(temp)
            temp = temp[sort_idx]
            tg = tg[sort_idx]
            dta = dta[sort_idx]

            # 微分の計算 (Central Difference)
            dtg = np.gradient(tg, temp)
            ddta = np.gradient(dta, temp)
            
            data_store[uploaded_file.name] = pd.DataFrame({
                "Temp": temp,
                "TG": tg,
                "DTA": dta,
                "DTG": dtg,
                "DDTA": ddta
            })
            
        except Exception as e:
            st.error(f"エラー: {uploaded_file.name} を読み込めませんでした。\n詳細: {e}")

# --- 2. 重量減少計算セクション ---
if data_store:
    st.header("📊 重量減少量の計算 (Delta Weight)")
    with st.expander("計算ツールを開く", expanded=True):
        c1, c2, c3 = st.columns(3)
        t_start = c1.number_input("開始温度 (T1)", value=100.0)
        t_end = c2.number_input("終了温度 (T2)", value=500.0)
        
        results = []
        for name, df in data_store.items():
            # 線形補間で指定温度の重量を取得
            w1 = np.interp(t_start, df["Temp"], df["TG"])
            w2 = np.interp(t_end, df["Temp"], df["TG"])
            diff = w1 - w2
            results.append({"File": name, f"TG at {t_start}°C": w1, f"TG at {t_end}°C": w2, "ΔWeight (%)": diff})
        
        st.table(pd.DataFrame(results))

# --- 3. プロット設定セクション ---
if data_store:
    st.header("🎨 グラフの作成とカスタマイズ")
    
    # グラフ設定用のコンテナ
    col_settings, col_plot = st.columns([1, 2])
    
    with col_settings:
        st.subheader("表示データの選択")
        
        # プロットリストの作成
        plot_configs = []
        
        for name in data_store.keys():
            st.markdown(f"**{name}**")
            options = ["TG", "DTA", "DTG", "DDTA"]
            selected_types = st.multiselect(f"{name} の表示項目", options, default=["TG"], key=f"sel_{name}")
            
            for curve_type in selected_types:
                with st.expander(f"設定: {name} - {curve_type}"):
                    color = st.color_picker("色", value="#1f77b4" if "TG" in curve_type else "#ff7f0e", key=f"col_{name}_{curve_type}")
                    linestyle = st.selectbox("線種", ["- (実線)", "-- (破線)", "-. (一点鎖線)", ": (点線)"], key=f"ls_{name}_{curve_type}")
                    linewidth = st.slider("太さ", 0.5, 5.0, 1.5, key=f"lw_{name}_{curve_type}")
                    axis_sel = st.radio("Y軸", ["左軸 (Weight/DTG)", "右軸 (DTA/DDTA)"], index=0 if curve_type in ["TG", "DTG"] else 1, key=f"ax_{name}_{curve_type}")
                    
                    plot_configs.append({
                        "filename": name,
                        "type": curve_type,
                        "color": color,
                        "linestyle": linestyle.split()[0],
                        "linewidth": linewidth,
                        "axis": "left" if "左軸" in axis_sel else "right"
                    })

    # --- 4. プロット描画 ---
    with col_plot:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()
        
        has_left = False
        has_right = False
        
        for config in plot_configs:
            df = data_store[config["filename"]]
            x = df["Temp"]
            y = df[config["type"]]
            
            label = f"{config['filename']} ({config['type']})"
            target_ax = ax1 if config["axis"] == "left" else ax2
            
            if config["axis"] == "left": has_left = True
            if config["axis"] == "right": has_right = True
            
            target_ax.plot(x, y, label=label, 
                           color=config["color"], 
                           linestyle=config["linestyle"], 
                           linewidth=config["linewidth"])

        ax1.set_xlabel("Temperature ($^\circ$C)")
        if has_left: ax1.set_ylabel("Weight % / Derivative")
        if has_right: ax2.set_ylabel("DTA / Derivative")
        
        # グリッドと凡例
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 凡例をまとめて表示
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        st.pyplot(fig)

    # --- 5. ダウンロードセクション ---
    st.header("💾 エクスポート")
    d_col1, d_col2, d_col3 = st.columns(3)
    
    # PNG保存
    fn = "plot_export"
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    d_col1.download_button("PNGでダウンロード", data=img_buffer.getvalue(), file_name=f"{fn}.png", mime="image/png")
    
    # TIFF保存
    tiff_buffer = io.BytesIO()
    fig.savefig(tiff_buffer, format='tiff', dpi=300, bbox_inches='tight')
    d_col2.download_button("TIFFでダウンロード", data=tiff_buffer.getvalue(), file_name=f"{fn}.tiff", mime="image/tiff")
    
    # Gnuplot形式
    # Gnuplot用のデータとスクリプトを作成
    gnuplot_script = "set terminal pngcairo size 800,600 enhanced font 'Arial,10'\n"
    gnuplot_script += "set output 'plot.png'\n"
    gnuplot_script += "set xlabel 'Temperature (C)'\n"
    gnuplot_script += "set ylabel 'Weight %'\n"
    gnuplot_script += "set y2label 'DTA'\n"
    gnuplot_script += "set y2tics\n"
    gnuplot_script += "set grid\n"
    gnuplot_script += "plot "
    
    plot_cmds = []
    # ※Gnuplotエクスポートは簡易的な実装として、現在メモリにあるデータを結合CSVとしてダウンロードさせ、それを参照するスクリプトを作成します
    combined_df = pd.DataFrame()
    for name, df in data_store.items():
        # 列名にファイル名をつけて結合
        temp_df = df.copy()
        temp_df.columns = [f"{name}_{c}" for c in temp_df.columns]
        if combined_df.empty:
            combined_df = temp_df
        else:
            # 温度軸が違う可能性があるため、単純結合は難しいが、ここでは行ごとのマージを試みる（または単に横に結合）
            combined_df = pd.concat([combined_df, temp_df], axis=1)
            
    csv_data = combined_df.to_csv(index=False, sep='\t')
    
    # Gnuplotのplotコマンド生成
    for i, config in enumerate(plot_configs):
        # 列名を検索
        col_name = f"{config['filename']}_{config['type']}"
        # 列インデックスを探す (Gnuplotは1始まり)
        try:
            col_idx = combined_df.columns.get_loc(col_name) + 1
            temp_col_idx = combined_df.columns.get_loc(f"{config['filename']}_Temp") + 1
            axis_str = "x1y1" if config["axis"] == "left" else "x1y2"
            plot_cmds.append(f"'data.dat' using {temp_col_idx}:{col_idx} with lines lw {config['linewidth']} dt 1 title '{col_name}' axes {axis_str}")
        except:
            pass

    gnuplot_script += ", ".join(plot_cmds)
    
    # Zipでまとめる代わりに、テキストエリアに表示＋データダウンロードを提供
    with d_col3:
        with st.popover("Gnuplot形式を取得"):
            st.markdown("以下のデータを `data.dat` として保存し、スクリプトを実行してください。")
            st.download_button("データファイル (data.dat)", data=csv_data, file_name="data.dat")
            st.code(gnuplot_script, language="gnuplot")

else:
    st.info("👈 サイドバーからデータファイルをアップロードしてください。")