import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re

# ページ設定
st.set_page_config(page_title="TG/DTA Pro (Rigaku Enhanced)", layout="wide")
st.title("📈 TG/DTA 解析ツール Pro (リガク完全対応版)")

# --- 関数: 高度なデータ読み込みロジック ---
def load_data_enhanced(file_obj, col_indices, manual_skip=None):
    """
    Rigaku形式(#GDタグ)や一般的なCSV/TXTを柔軟に読み込む関数
    """
    # 1. バイト列として読み込み、デコードを試行
    bytes_data = file_obj.read()
    encodings = ['shift_jis', 'cp932', 'utf-8', 'latin1']
    text_data = ""
    
    for enc in encodings:
        try:
            text_data = bytes_data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
            
    if not text_data:
        raise ValueError("エンコーディングの判定に失敗しました。")

    lines = text_data.splitlines()
    
    # --- パターンA: Rigaku (#GD タグ) 形式の検出 ---
    # 行の先頭が #GD で始まる行を探す
    gd_lines = [line for line in lines if line.strip().startswith('#GD')]
    
    df = None
    
    if len(gd_lines) > 10:  # #GD行がある程度あればRigaku形式とみなす
        # #GD を削除して数値部分だけにする
        # Rigaku形式は "#GD (タブ) Time (タブ) Temp..." となっていることが多い
        processed_lines = []
        for line in gd_lines:
            # "#GD" を削除し、前後の空白を除去
            clean_line = line.replace('#GD', '').strip()
            processed_lines.append(clean_line)
            
        # データ結合してDataFrame化 (タブ区切りまたはスペース区切り)
        data_str = "\n".join(processed_lines)
        try:
            df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=None)
        except:
            # 失敗したらタブ区切り固定で試行
            df = pd.read_csv(io.StringIO(data_str), sep='\t', header=None)
            
    else:
        # --- パターンB: 通常のテキスト/CSV形式 ---
        # ユーザー指定のスキップ行数がある場合
        if manual_skip is not None and manual_skip > 0:
            # manual_skip は "データの開始行番号(1始まり)" を想定しているため、
            # skiprows には manual_skip - 1 を渡す (0始まりインデックスのため)
            # ただし、ヘッダー行を含めるなら調整が必要。
            # ここでは「指定行からデータが始まる（ヘッダーなし）」として扱う
            data_str = "\n".join(lines[manual_skip-1:])
            df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=None)
        else:
            # 自動検出ロジック (Temp/TG などのキーワード探索)
            header_idx = -1
            keywords = ["Temp", "Temperature", "TG", "DTA", "Time", "min"]
            
            for i, line in enumerate(lines):
                hit = sum(1 for k in keywords if k in line)
                if hit >= 2:
                    header_idx = i
                    break
            
            if header_idx != -1:
                # ヘッダーが見つかった位置から読み込み
                # 直下の行が単位行([mg]など)の場合は数値変換エラーになるので除去する処理が必要
                data_str = "\n".join(lines[header_idx:])
                try:
                    # まずヘッダーありで読み込む
                    temp_df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=0)
                    # 1行目が数値かチェック (単位行判定)
                    try:
                        pd.to_numeric(temp_df.iloc[0, col_indices['temp']])
                        df = temp_df
                    except:
                        # 数値でなければ1行目(単位行)をスキップ
                        df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=0, skiprows=[1])
                except:
                    pass
            
            if df is None:
                # 何も見つからなければ単純読み込み
                df = pd.read_csv(io.StringIO(text_data), sep=None, engine='python', header=None)

    if df is None or df.empty:
        raise ValueError("データを読み取れませんでした。手動設定を試してください。")

    return df

# --- サイドバーUI ---
st.sidebar.header("1. データ読み込み")

# 読み込みオプション
with st.sidebar.expander("詳細設定 (読み込めない場合)", expanded=False):
    manual_row_start = st.number_input("データの開始行番号 (指定時のみ有効)", value=0, min_value=0, help="例: 49行目からデータがある場合は49と入力。0の場合は自動検出します。")

uploaded_files = st.sidebar.file_uploader(
    "ファイルをアップロード", 
    type=['csv', 'txt', 'asc'], 
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("列の定義 (0始まり)")
# リガク形式(#GD)の場合、#GD除去後の列順は概ね: 0:Time, 1:Temp, 3:TG, 5:DTA のことが多いがファイルによる
# ファイル読み込み後にプレビューを表示して確認できるようにする
col_temp = st.sidebar.number_input("温度列 (Temp)", value=1, min_value=0)
col_tg = st.sidebar.number_input("重量列 (TG)", value=3, min_value=0) # リガクに合わせてデフォルトを3に変更
col_dta = st.sidebar.number_input("DTA列", value=5, min_value=0)   # リガクに合わせてデフォルトを5に変更

col_indices = {'temp': col_temp, 'tg': col_tg, 'dta': col_dta}
data_store = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # 独自ローダーで読み込み
            skip_val = manual_row_start if manual_row_start > 0 else None
            df = load_data_enhanced(uploaded_file, col_indices, manual_skip=skip_val)
            
            # --- プレビュー機能（デバッグ用）---
            # 最初の数行を表示して列番号を確認しやすくする
            if len(data_store) == 0: # 最初のファイルだけ表示
                with st.expander(f"データプレビュー: {uploaded_file.name}", expanded=False):
                    st.dataframe(df.head())
                    st.info(f"現在の列指定 -> 温度:{col_temp}, TG:{col_tg}, DTA:{col_dta}")
            
            # 数値変換と抽出
            # 列番号が範囲外でないかチェック
            max_col = df.shape[1] - 1
            if col_temp > max_col or col_tg > max_col or col_dta > max_col:
                st.error(f"{uploaded_file.name}: 列番号が大きすぎます。データは全{max_col+1}列です。")
                continue

            temp = pd.to_numeric(df.iloc[:, col_temp], errors='coerce').values
            tg = pd.to_numeric(df.iloc[:, col_tg], errors='coerce').values
            dta = pd.to_numeric(df.iloc[:, col_dta], errors='coerce').values
            
            # NaN除去
            mask = ~np.isnan(temp) & ~np.isnan(tg) & ~np.isnan(dta)
            temp = temp[mask]
            tg = tg[mask]
            dta = dta[mask]

            if len(temp) == 0:
                st.error(f"{uploaded_file.name}: 有効な数値データが見つかりませんでした。列指定を確認してください。")
                continue

            # ソート
            sort_idx = np.argsort(temp)
            temp = temp[sort_idx]
            tg = tg[sort_idx]
            dta = dta[sort_idx]

            # 微分計算
            dtg = np.gradient(tg, temp)
            ddta = np.gradient(dta, temp)
            
            data_store[uploaded_file.name] = {
                "Temp": temp, "TG": tg, "DTA": dta, 
                "DTG": dtg, "DDTA": ddta, "DTA (Corrected)": dta.copy()
            }
            
        except Exception as e:
            st.error(f"読み込みエラー ({uploaded_file.name}): {e}")

# --- 2. DTAベースライン補正 ---
if data_store:
    st.sidebar.markdown("---")
    st.sidebar.header("2. DTA補正")
    use_correction = st.sidebar.checkbox("補正 ON", value=False)
    
    if use_correction:
        bl_t1 = st.sidebar.number_input("基準温度1", value=100.0)
        bl_t2 = st.sidebar.number_input("基準温度2", value=600.0)
        
        for name, data in data_store.items():
            t, d = data["Temp"], data["DTA"]
            # 範囲内チェック
            if t.min() <= bl_t1 <= t.max() and t.min() <= bl_t2 <= t.max():
                y1 = np.interp(bl_t1, t, d)
                y2 = np.interp(bl_t2, t, d)
                if bl_t2 != bl_t1:
                    m = (y2 - y1) / (bl_t2 - bl_t1)
                    c = y1 - m * bl_t1
                    baseline = m * t + c
                    data_store[name]["DTA (Corrected)"] = d - baseline

# --- 3. 重量減少量 ---
if data_store:
    st.header("📊 重量減少 (Delta Weight)")
    with st.expander("計算パネル"):
        c1, c2 = st.columns(2)
        wt1 = c1.number_input("開始温度 T1", value=100.0)
        wt2 = c2.number_input("終了温度 T2", value=500.0)
        
        res = []
        for name, data in data_store.items():
            w1 = np.interp(wt1, data["Temp"], data["TG"])
            w2 = np.interp(wt2, data["Temp"], data["TG"])
            res.append({"File": name, f"TG@{wt1}": f"{w1:.2f}", f"TG@{wt2}": f"{w2:.2f}", "ΔWt": f"{w1-w2:.3f}"})
        st.table(pd.DataFrame(res))

# --- 4. プロット ---
if data_store:
    st.header("🎨 グラフ")
    c_set, c_plt = st.columns([1, 2.5])
    plots = []
    
    with c_set:
        st.subheader("表示設定")
        for name in data_store.keys():
            st.markdown(f"**{name}**")
            # デフォルト選択
            def_items = ["TG", "DTA (Corrected)"] if use_correction else ["TG", "DTA"]
            sels = st.multiselect(f"項目 ({name})", ["TG", "DTA", "DTA (Corrected)", "DTG", "DDTA"], default=def_items, key=f"ms_{name}")
            
            for item in sels:
                with st.expander(f"{item} 詳細"):
                    col_def = "#1f77b4"
                    if "DTA" in item: col_def = "#ff7f0e"
                    elif "DTG" in item: col_def = "#2ca02c"
                    
                    c = st.color_picker("色", col_def, key=f"c_{name}_{item}")
                    ls = st.selectbox("線種", ["-", "--", "-.", ":"], key=f"ls_{name}_{item}")
                    lw = st.slider("太さ", 0.5, 4.0, 1.5, key=f"lw_{name}_{item}")
                    ax = st.radio("軸", ["左(TG)", "右(DTA)"], index=1 if "DTA" in item else 0, key=f"ax_{name}_{item}")
                    plots.append({"name": name, "type": item, "c": c, "ls": ls, "lw": lw, "ax": 0 if "左" in ax else 1})

    with c_plt:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        axs = [ax1, ax2]
        
        for p in plots:
            d = data_store[p["name"]]
            axs[p["ax"]].plot(d["Temp"], d[p["type"]], label=f"{p['name']} {p['type']}", color=p["c"], ls=p["ls"], lw=p["lw"])
            
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Weight % / DTG")
        ax2.set_ylabel("DTA (uV) / DDTA")
        ax1.grid(True, ls=':', alpha=0.6)
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2: ax1.legend(h1+h2, l1+l2, loc='best')
        st.pyplot(fig)

    # --- 5. 保存 ---
    st.header("💾 保存")
    d1, d2, d3 = st.columns(3)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    d1.download_button("PNG保存", buf.getvalue(), "plot.png", "image/png")
    
    buf_t = io.BytesIO()
    fig.savefig(buf_t, format='tiff', dpi=300, bbox_inches='tight')
    d2.download_button("TIFF保存", buf_t.getvalue(), "plot.tiff", "image/tiff")
    
    # Gnuplot
    m_df = pd.DataFrame()
    for name, data in data_store.items():
        _d = pd.DataFrame(data)
        _d.columns = [f"{name}:{c}" for c in _d.columns]
        m_df = pd.concat([m_df, _d], axis=1) if not m_df.empty else _d
        
    csv_txt = m_df.to_csv(index=False, sep='\t')
    gp = "set term pngcairo\nset out 'plot.png'\nplot "
    g_cmds = []
    for p in plots:
        try:
            cn = f"{p['name']}:{p['type']}"
            tn = f"{p['name']}:Temp"
            ci = m_df.columns.get_loc(cn)+1
            ti = m_df.columns.get_loc(tn)+1
            ax = "x1y2" if p["ax"]==1 else "x1y1"
            g_cmds.append(f"'data.dat' u {ti}:{ci} w l lw {p['lw']} lc rgb '{p['c']}' t '{cn}' axes {ax}")
        except: pass
    gp += ", ".join(g_cmds)
    
    with d3:
        with st.popover("Gnuplot出力"):
            st.download_button("data.dat", csv_txt, "data.dat")
            st.download_button("plot.gp", gp, "plot.gp")

else:
    st.info("👈 サイドバーからファイルをアップロードしてください。")