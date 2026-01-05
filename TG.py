import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ページ設定
st.set_page_config(page_title="TG/DTA Pro (Rigaku Compatible)", layout="wide")

st.title("📈 TG/DTA 解析ツール Pro (リガク対応版)")

# --- 関数: データ読み込みロジックの強化 ---
def load_data(file_obj, col_indices):
    """
    ファイルを読み込み、ヘッダー位置を自動検索してDataFrame化する関数
    Rigaku形式 (Shift-JIS, Tab区切り, [Data]タグ または Temp/TG 列名) に対応
    """
    encodings = ['shift_jis', 'utf-8', 'cp932', 'latin1']
    delimiters = ['\t', ',', '\s+'] # タブ、カンマ、スペース
    
    # ファイルの内容をバイト列として読み込む
    bytes_data = file_obj.read()
    
    # 1. エンコーディング判定と行リスト化
    lines = []
    decoded_str = ""
    for enc in encodings:
        try:
            decoded_str = bytes_data.decode(enc)
            lines = decoded_str.splitlines()
            break
        except UnicodeDecodeError:
            continue
            
    if not lines:
        raise ValueError("エンコーディングを判別できませんでした。")

    # 2. ヘッダー行（データ開始位置）の探索
    # 戦略: "Temp" と "TG" または "Temperature" が含まれる行を探す
    header_row_idx = -1
    keywords = ["Temp", "TEMP", "Temperature", "TG", "DTA", "Time", "min"]
    
    # [Data]タグがある場合はその直後を優先探索するロジックも一般的だが、
    # ここでは列名が含まれる行をヘッダーとみなす汎用的な方法をとる
    for i, line in enumerate(lines):
        # 少なくとも2つのキーワードが含まれていればヘッダー行とみなす
        hit_count = sum(1 for k in keywords if k in line)
        if hit_count >= 2:
            header_row_idx = i
            break
    
    # ヘッダーが見つからない場合は、単純にスキップなし(0)とするか、あるいは先頭
    if header_row_idx == -1:
        # Rigakuの場合、ヘッダーなしでデータが始まることは稀だが、
        # 見つからない場合はユーザー指定のスキップ数（デフォルト0）を使うなどの対策が必要
        # ここではとりあえず0行目と仮定
        header_row_idx = 0
    
    # 3. Pandasで読み込み
    # ヘッダー行が見つかったので、その行をheader=0として読み込むため、それ以前をスキップする形ではなく
    # io.StringIOでその部分だけ渡す
    data_str = "\n".join(lines[header_row_idx:])
    
    df = None
    # 区切り文字を変えてトライ
    for sep in delimiters:
        try:
            # 単位行（[deg], [mg]など）がヘッダーの直下にある場合、型変換エラーになるため
            # ヘッダーの次の行が数値でない場合はスキップする処理が必要かもしれないが
            # pd.read_csvは数値変換できない行をNaNにするかエラーにする。
            # Rigakuは ヘッダー行 -> 単位行 -> データ行 のパターンが多い。
            
            temp_df = pd.read_csv(io.StringIO(data_str), sep=sep, header=0)
            
            # 2行目（インデックス0）が単位行("deg", "mg"等)の場合、数値変換できないので削除を試みる
            # 簡易チェック: 指定された温度列が数値変換できるか？
            try:
                pd.to_numeric(temp_df.iloc[:, col_indices['temp']], errors='raise')
                df = temp_df
                break # 成功
            except:
                # 1行目が単位行かもしれないので、header=0 (列名), skiprows=[1] (単位行飛ばし) で再トライ
                temp_df = pd.read_csv(io.StringIO(data_str), sep=sep, header=0, skiprows=[1])
                pd.to_numeric(temp_df.iloc[:, col_indices['temp']], errors='raise') # チェック
                df = temp_df
                break
        except:
            continue
            
    if df is None:
        raise ValueError("データの解析に失敗しました。区切り文字や形式を確認してください。")
        
    return df

# --- サイドバー設定 ---
st.sidebar.header("1. データ読み込み")
uploaded_files = st.sidebar.file_uploader(
    "CSV / TXT (Rigaku等) をアップロード", 
    type=['csv', 'txt'], 
    accept_multiple_files=True
)

# 列番号のデフォルト設定（Rigakuは Temp, TG, DTA の順でない場合もあるためユーザー確認用）
st.sidebar.markdown("---")
st.sidebar.subheader("列の定義 (0始まり)")
st.sidebar.info("データ読み込み後、エラーが出る場合はここを調整してください。")
col_temp = st.sidebar.number_input("温度列 (Temp)", value=0, min_value=0)
col_tg = st.sidebar.number_input("重量列 (TG)", value=1, min_value=0)
col_dta = st.sidebar.number_input("DTA列", value=2, min_value=0)

col_indices = {'temp': col_temp, 'tg': col_tg, 'dta': col_dta}
data_store = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # 独自関数で読み込み
            df = load_data(uploaded_file, col_indices)
            
            # 数値データへの変換（念のため強制変換）
            temp = pd.to_numeric(df.iloc[:, col_temp], errors='coerce').values
            tg = pd.to_numeric(df.iloc[:, col_tg], errors='coerce').values
            dta = pd.to_numeric(df.iloc[:, col_dta], errors='coerce').values
            
            # NaN（単位行などが混ざって変換できなかった行）を除去
            valid_mask = ~np.isnan(temp) & ~np.isnan(tg) & ~np.isnan(dta)
            temp = temp[valid_mask]
            tg = tg[valid_mask]
            dta = dta[valid_mask]

            # ソート
            sort_idx = np.argsort(temp)
            temp = temp[sort_idx]
            tg = tg[sort_idx]
            dta = dta[sort_idx]

            # 微分計算
            dtg = np.gradient(tg, temp)
            ddta = np.gradient(dta, temp)
            
            data_store[uploaded_file.name] = {
                "Temp": temp,
                "TG": tg,
                "DTA": dta,
                "DTG": dtg,
                "DDTA": ddta,
                "DTA (Corrected)": dta.copy()
            }
            
        except Exception as e:
            st.error(f"エラー: {uploaded_file.name} を読み込めませんでした。\n{e}")

# --- 2. DTAベースライン補正 ---
if data_store:
    st.sidebar.markdown("---")
    st.sidebar.header("2. DTA補正")
    use_correction = st.sidebar.checkbox("ベースライン補正 ON", value=False)
    
    if use_correction:
        bl_t1 = st.sidebar.number_input("基準温度 1 (°C)", value=100.0)
        bl_t2 = st.sidebar.number_input("基準温度 2 (°C)", value=600.0)
        
        for name, data in data_store.items():
            temp = data["Temp"]
            dta = data["DTA"]
            
            # 範囲外エラー防止
            if min(temp) <= bl_t1 <= max(temp) and min(temp) <= bl_t2 <= max(temp):
                y1 = np.interp(bl_t1, temp, dta)
                y2 = np.interp(bl_t2, temp, dta)
                
                if bl_t2 != bl_t1:
                    slope = (y2 - y1) / (bl_t2 - bl_t1)
                    intercept = y1 - slope * bl_t1
                    baseline = slope * temp + intercept
                    data_store[name]["DTA (Corrected)"] = dta - baseline
            else:
                st.sidebar.warning(f"{name}: 指定温度が範囲外のため補正スキップ")

# --- 3. 重量減少量 計算 ---
if data_store:
    st.header("📊 重量減少量 (Delta Weight)")
    with st.expander("計算パネル", expanded=False):
        c1, c2 = st.columns(2)
        calc_t1 = c1.number_input("開始 T1 (°C)", value=100.0)
        calc_t2 = c2.number_input("終了 T2 (°C)", value=500.0)
        
        res_list = []
        for name, data in data_store.items():
            w1 = np.interp(calc_t1, data["Temp"], data["TG"])
            w2 = np.interp(calc_t2, data["Temp"], data["TG"])
            res_list.append({
                "File": name,
                f"TG@{calc_t1}": f"{w1:.2f}",
                f"TG@{calc_t2}": f"{w2:.2f}",
                "ΔWt (%)": f"{w1 - w2:.3f}"
            })
        st.table(pd.DataFrame(res_list))

# --- 4. プロット設定 ---
if data_store:
    st.header("🎨 グラフ設定")
    
    col_set, col_fig = st.columns([1, 2.5])
    plot_items = []
    
    with col_set:
        st.subheader("表示データ")
        for name in data_store.keys():
            st.markdown(f"**{name}**")
            opts = ["TG", "DTA", "DTA (Corrected)", "DTG", "DDTA"]
            def_sel = ["TG", "DTA (Corrected)"] if use_correction else ["TG", "DTA"]
            
            sels = st.multiselect(f"{name}", opts, default=def_sel, key=f"s_{name}")
            
            for item in sels:
                with st.expander(f"{item} スタイル"):
                    # 自動色設定
                    c_def = "#1f77b4"
                    if "DTA" in item: c_def = "#ff7f0e"
                    if "DTG" in item: c_def = "#2ca02c"
                    
                    color = st.color_picker("色", value=c_def, key=f"c_{name}_{item}")
                    ls = st.selectbox("線種", ["-", "--", "-.", ":"], key=f"l_{name}_{item}")
                    lw = st.slider("太さ", 0.5, 4.0, 1.5, key=f"w_{name}_{item}")
                    ax = st.radio("軸", ["左 (TG)", "右 (DTA)"], 
                                  index=1 if "DTA" in item else 0, key=f"a_{name}_{item}")
                    
                    plot_items.append({
                        "name": name, "type": item, "color": color, 
                        "ls": ls, "lw": lw, "axis": 0 if "左" in ax else 1
                    })

    # --- プロット描画 ---
    with col_fig:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        axes = [ax1, ax2]
        
        for p in plot_items:
            d = data_store[p["name"]]
            axes[p["axis"]].plot(d["Temp"], d[p["type"]], 
                                 label=f"{p['name']} {p['type']}",
                                 color=p['color'], linestyle=p['ls'], linewidth=p['lw'])
            
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Weight % / DTG")
        ax2.set_ylabel("DTA (uV) / DDTA")
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax1.legend(h1+h2, l1+l2, loc='upper right')
            
        st.pyplot(fig)

    # --- 5. エクスポート ---
    st.header("💾 保存")
    c_dl1, c_dl2, c_dl3 = st.columns(3)
    
    # PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    c_dl1.download_button("PNG Download", buf.getvalue(), "plot.png", "image/png")
    
    # TIFF
    buf_tiff = io.BytesIO()
    fig.savefig(buf_tiff, format='tiff', dpi=300, bbox_inches='tight')
    c_dl2.download_button("TIFF Download", buf_tiff.getvalue(), "plot.tiff", "image/tiff")
    
    # Gnuplot
    # データ結合処理
    m_df = pd.DataFrame()
    for name, data in data_store.items():
        _d = pd.DataFrame(data)
        _d.columns = [f"{name}:{c}" for c in _d.columns]
        m_df = pd.concat([m_df, _d], axis=1) if not m_df.empty else _d
        
    csv_str = m_df.to_csv(index=False, sep='\t')
    
    gp_script = "set terminal pngcairo enhanced\nset output 'plot.png'\nset grid\nplot "
    cmds = []
    for p in plot_items:
        try:
            col = f"{p['name']}:{p['type']}"
            tmp = f"{p['name']}:Temp"
            idx_c = m_df.columns.get_loc(col) + 1
            idx_t = m_df.columns.get_loc(tmp) + 1
            ax_s = "x1y2" if p["axis"]==1 else "x1y1"
            cmds.append(f"'data.dat' u {idx_t}:{idx_c} w l lw {p['lw']} t '{col}' axes {ax_s}")
        except: pass
    gp_script += ", ".join(cmds)
    
    with c_dl3:
        with st.popover("Gnuplot Data"):
            st.download_button("data.dat", csv_str, "data.dat")
            st.download_button("plot.gp", gp_script, "plot.gp")

else:
    st.info("👈 サイドバーからCSVまたはリガクTXTファイルをアップロードしてください。")