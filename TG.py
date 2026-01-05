import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ページ設定
st.set_page_config(page_title="TG/DTA Pro (Legend Control)", layout="wide")
st.title("📈 TG/DTA 解析ツール Pro (凡例位置調整版)")

# --- 関数: 高度なデータ読み込みロジック (Rigaku対応) ---
def load_data_enhanced(file_obj, col_indices, manual_skip=None):
    """
    Rigaku形式(#GDタグ)や一般的なCSV/TXTを柔軟に読み込む関数
    """
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
    
    # --- パターンA: Rigaku (#GD タグ) 形式 ---
    gd_lines = [line for line in lines if line.strip().startswith('#GD')]
    
    df = None
    if len(gd_lines) > 10:
        processed_lines = []
        for line in gd_lines:
            clean_line = line.replace('#GD', '').strip()
            processed_lines.append(clean_line)
        data_str = "\n".join(processed_lines)
        try:
            df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=None)
        except:
            df = pd.read_csv(io.StringIO(data_str), sep='\t', header=None)
            
    else:
        # --- パターンB: 通常形式 ---
        if manual_skip is not None and manual_skip > 0:
            data_str = "\n".join(lines[manual_skip-1:])
            df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=None)
        else:
            header_idx = -1
            keywords = ["Temp", "Temperature", "TG", "DTA", "Time", "min"]
            for i, line in enumerate(lines):
                if sum(1 for k in keywords if k in line) >= 2:
                    header_idx = i
                    break
            
            if header_idx != -1:
                data_str = "\n".join(lines[header_idx:])
                try:
                    temp_df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=0)
                    try:
                        pd.to_numeric(temp_df.iloc[0, col_indices['temp']])
                        df = temp_df
                    except:
                        df = pd.read_csv(io.StringIO(data_str), sep=None, engine='python', header=0, skiprows=[1])
                except:
                    pass
            
            if df is None:
                df = pd.read_csv(io.StringIO(text_data), sep=None, engine='python', header=None)

    if df is None or df.empty:
        raise ValueError("データを読み取れませんでした。")

    return df

# --- サイドバーUI ---
st.sidebar.header("1. データ読み込み")
with st.sidebar.expander("詳細設定 (読み込めない場合)", expanded=False):
    manual_row_start = st.number_input("データの開始行番号", value=0, min_value=0)

uploaded_files = st.sidebar.file_uploader(
    "ファイルをアップロード", 
    type=['csv', 'txt', 'asc'], 
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("列の定義 (0始まり)")
col_temp = st.sidebar.number_input("温度列 (Temp)", value=1, min_value=0)
col_tg = st.sidebar.number_input("重量列 (TG)", value=3, min_value=0)
col_dta = st.sidebar.number_input("DTA列", value=5, min_value=0)

col_indices = {'temp': col_temp, 'tg': col_tg, 'dta': col_dta}
data_store = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            skip_val = manual_row_start if manual_row_start > 0 else None
            df = load_data_enhanced(uploaded_file, col_indices, manual_skip=skip_val)
            
            # 数値変換と抽出
            max_col = df.shape[1] - 1
            if col_temp > max_col or col_tg > max_col or col_dta > max_col:
                st.error(f"{uploaded_file.name}: 列番号指定が範囲外です。")
                continue

            temp = pd.to_numeric(df.iloc[:, col_temp], errors='coerce').values
            tg_raw = pd.to_numeric(df.iloc[:, col_tg], errors='coerce').values
            dta = pd.to_numeric(df.iloc[:, col_dta], errors='coerce').values
            
            # NaN除去
            mask = ~np.isnan(temp) & ~np.isnan(tg_raw) & ~np.isnan(dta)
            temp = temp[mask]
            tg_raw = tg_raw[mask]
            dta = dta[mask]

            if len(temp) == 0:
                continue

            # ソート
            sort_idx = np.argsort(temp)
            temp = temp[sort_idx]
            tg_raw = tg_raw[sort_idx]
            dta = dta[sort_idx]

            # Weight % 変換
            initial_weight = tg_raw[0]
            if initial_weight != 0:
                tg_percent = (tg_raw / initial_weight) * 100.0
            else:
                tg_percent = tg_raw

            # 微分計算
            dtg = np.gradient(tg_percent, temp)
            ddta = np.gradient(dta, temp)
            
            data_store[uploaded_file.name] = {
                "Temp": temp, 
                "TG": tg_percent, 
                "TG_raw": tg_raw,
                "DTA": dta, 
                "DTG": dtg, 
                "DDTA": ddta, 
                "DTA (Corrected)": dta.copy()
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
    st.header("📊 重量減少率 (Delta Weight %)")
    with st.expander("計算パネル"):
        c1, c2 = st.columns(2)
        wt1 = c1.number_input("開始温度 T1", value=100.0)
        wt2 = c2.number_input("終了温度 T2", value=500.0)
        
        res = []
        for name, data in data_store.items():
            w1 = np.interp(wt1, data["Temp"], data["TG"])
            w2 = np.interp(wt2, data["Temp"], data["TG"])
            res.append({
                "File": name, 
                f"Wt%@{wt1:.0f}C": f"{w1:.2f}%", 
                f"Wt%@{wt2:.0f}C": f"{w2:.2f}%", 
                "ΔWt (%)": f"{w1-w2:.3f}%"
            })
        st.table(pd.DataFrame(res))

# --- 4. プロット ---
if data_store:
    st.header("🎨 グラフ")
    c_set, c_plt = st.columns([1, 2.5])
    plots = []
    
    with c_set:
        st.subheader("表示設定")
        
        # --- 凡例設定 (NEW) ---
        legend_pos = st.radio("凡例 (Legend) の位置", ["内部 (自動配置)", "外側 (右)"], index=1)
        st.markdown("---")

        for name in data_store.keys():
            st.markdown(f"**{name}**")
            
            def_items = ["TG", "DTA (Corrected)"] if use_correction else ["TG", "DTA"]
            opts = ["TG", "DTA", "DTA (Corrected)", "DTG", "DDTA"]
            sels = st.multiselect(f"項目 ({name})", opts, default=def_items, key=f"ms_{name}")
            
            for item in sels:
                with st.expander(f"{item} 詳細"):
                    col_def = "#1f77b4"
                    if "DTA" in item: col_def = "#ff7f0e"
                    elif "DTG" in item: col_def = "#2ca02c"
                    elif "DDTA" in item: col_def = "#d62728"
                    
                    c = st.color_picker("色", col_def, key=f"c_{name}_{item}")
                    ls = st.selectbox("線種", ["-", "--", "-.", ":"], key=f"ls_{name}_{item}")
                    lw = st.slider("太さ", 0.5, 4.0, 1.5, key=f"lw_{name}_{item}")
                    
                    default_axis = 0 if item == "TG" else 1
                    ax = st.radio("軸", ["左(TG %)", "右(微小/DTA)"], index=default_axis, key=f"ax_{name}_{item}")
                    plots.append({"name": name, "type": item, "c": c, "ls": ls, "lw": lw, "ax": 0 if "左" in ax else 1})

    with c_plt:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        axs = [ax1, ax2]
        
        has_left = False
        has_right = False
        
        for p in plots:
            d = data_store[p["name"]]
            axs[p["ax"]].plot(d["Temp"], d[p["type"]], label=f"{p['name']} {p['type']}", color=p["c"], ls=p["ls"], lw=p["lw"])
            if p["ax"] == 0: has_left = True
            if p["ax"] == 1: has_right = True
            
        ax1.set_xlabel("Temperature (°C)")
        
        if has_left: ax1.set_ylabel("Weight (%)")
        if has_right: ax2.set_ylabel("DTA / Derivative")
        
        ax1.grid(True, ls=':', alpha=0.6)
        
        # --- 凡例描画ロジック ---
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        
        if h1 or h2:
            handles = h1 + h2
            labels = l1 + l2
            
            if legend_pos == "内部 (自動配置)":
                # Matplotlibに最適な位置を判断させる
                ax1.legend(handles, labels, loc='best')
            else:
                # グラフの外側（右）に配置
                ax1.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.15, 1.0))

        st.pyplot(fig)

    # --- 5. 保存 ---
    st.header("💾 保存")
    d1, d2, d3 = st.columns(3)
    
    # PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    d1.download_button("PNG保存", buf.getvalue(), "plot.png", "image/png")
    
    # TIFF
    buf_t = io.BytesIO()
    fig.savefig(buf_t, format='tiff', dpi=300, bbox_inches='tight')
    d2.download_button("TIFF保存", buf_t.getvalue(), "plot.tiff", "image/tiff")
    
    # Gnuplot
    m_df = pd.DataFrame()
    for name, data in data_store.items():
        _d = pd.DataFrame(data)
        sel_keys = [k for k in _d.columns if k != "TG_raw"]
        _d = _d[sel_keys]
        _d.columns = [f"{name}:{c}" for c in _d.columns]
        m_df = pd.concat([m_df, _d], axis=1) if not m_df.empty else _d
        
    csv_txt = m_df.to_csv(index=False, sep='\t')
    
    # Gnuplotの凡例設定
    gp_key = "set key best" if "内部" in legend_pos else "set key outside right top"
    
    gp = f"set term pngcairo enhanced font 'Arial,12'\nset out 'plot.png'\nset xlabel 'Temp (C)'\nset ylabel 'Weight %'\nset y2label 'DTA / Derivative'\nset y2tics\nset grid\n{gp_key}\nplot "
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
    gp += ", \\\n     ".join(g_cmds)
    
    with d3:
        with st.popover("Gnuplot出力"):
            st.download_button("data.dat", csv_txt, "data.dat")
            st.download_button("plot.gp", gp, "plot.gp")

else:
    st.info("👈 サイドバーからファイルをアップロードしてください。")