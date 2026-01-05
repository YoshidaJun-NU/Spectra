import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io

# ---------------------------------------------------------
# ヘッダー自動検出関数
# ---------------------------------------------------------
def find_header_row(file_content, encodings=['shift_jis', 'cp932', 'utf-8', 'latin1']):
    """
    ファイルの中身を走査して、データ開始行（ヘッダー）と思われる行番号を返す
    """
    for encoding in encodings:
        try:
            # バイト列を文字列にデコードして行ごとにリスト化
            # (大きなファイルの場合は先頭数KBだけ読むのが効率的ですが、テキストファイルなら全読みでもOK)
            content = file_content.decode(encoding).splitlines()
            
            for i, line in enumerate(content[:300]):  # 先頭300行までをチェック
                # Rigaku形式の特徴: "Time", "Temp", "TG" などが含まれる行を探す
                # タブ区切りであることが多いため、splitしてみる
                parts = [p.strip() for p in line.split('\t') if p.strip()]
                
                # キーワード判定 (TempとTGまたはTimeが含まれていればヘッダーとみなす)
                line_str = line.lower()
                if ('temp' in line_str and 'tg' in line_str) or \
                   ('time' in line_str and 'temp' in line_str and 'min' in line_str):
                    return i, encoding, content
            
            # 見つからなかった場合
            return None, encoding, content
            
        except UnicodeDecodeError:
            continue
            
    return None, None, None

# ---------------------------------------------------------
# データ読み込みと前処理
# ---------------------------------------------------------
def load_and_normalize_data(uploaded_file, manual_skip_rows, use_auto_detect):
    """
    ファイルを読み込み、正規化を行う
    """
    file_bytes = uploaded_file.getvalue()
    
    skip_rows = manual_skip_rows
    encoding = 'shift_jis' # デフォルト
    header_found = False
    
    # --- 自動検出ロジック ---
    if use_auto_detect:
        detected_row, detected_enc, lines = find_header_row(file_bytes)
        if detected_row is not None:
            skip_rows = detected_row
            encoding = detected_enc
            header_found = True
            # st.info(f"{uploaded_file.name}: ヘッダーを {skip_rows + 1} 行目に検出しました (Encoding: {encoding})")
        else:
            # 自動検出失敗時は手動設定を使用し、エンコーディングは総当たり
            pass

    # --- 読み込み実行 ---
    df = None
    encodings_to_try = [encoding] if header_found else ['shift_jis', 'cp932', 'utf-8', 'latin1']
    
    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            # Rigakuのファイルはタブ区切りが多いが、sep=Noneで自動判定させる
            # ヘッダー行を特定して読み込む
            df = pd.read_csv(
                uploaded_file, 
                sep=None, 
                engine='python', 
                encoding=enc, 
                skiprows=skip_rows
            )
            
            # 読み込み成功したらループを抜ける
            break 
        except Exception:
            continue
            
    if df is None:
        return None, "読み込みに失敗しました。文字コードまたは形式が不明です。"

    # --- カラム名のクリーニング ---
    # Rigakuデータは先頭に #GD という列が入ったり、カラム名がずれたりすることがあるため調整
    
    # 1. カラム名を文字列にする
    df.columns = df.columns.astype(str)
    
    # 2. 不要な空白を除去
    df.columns = [c.strip() for c in df.columns]
    
    # 3. #GD列 (データ行識別子) が混入している場合の処理
    # データ行が "#GD  Time  Temp..." のようになっていると、
    # pandasは "#GD" を最初の列名、"Time" を2番目... と解釈してズレることがある
    
    # "Temp" や "Weight/TG" カラムを探す
    temp_col = None
    weight_col = None
    
    for col in df.columns:
        c_lower = col.lower()
        if 'temp' in c_lower and 'dtemp' not in c_lower: # 微分(DTemp)は除外
            temp_col = col
        if 'tg' in c_lower and 'dtg' not in c_lower: # 微分(DTG)は除外
            weight_col = col
        elif 'weight' in c_lower:
            weight_col = col
            
    # カラムが見つからない場合、列の位置で強制割り当てを試みる
    # Rigaku形式: 0:タグ, 1:Time, 2:Temp, ... TG ... のパターンが多い
    if temp_col is None or weight_col is None:
        if df.shape[1] >= 4:
            # 一般的なRigaku形式の並びを仮定
            # [Tag, Time, Temp, DTemp, TG, DTG, DTA...]
            # カラム名が正しくパースされていない場合、中身で判断するのは難しいので
            # ユーザーに列選択させるのがベストだが、ここでは簡易的に列番で推定
            
            # もし1列目がすべて "#GD" なら、それはタグ列
            if df.iloc[:, 0].astype(str).str.contains('#GD').all():
                # 2列目(Time), 3列目(Temp), 5列目(TG) あたりを候補に
                if df.shape[1] > 2: temp_col = df.columns[2]
                if df.shape[1] > 4: weight_col = df.columns[4] # TGは後ろの方にあることが多い
            else:
                # タグ列がない場合 (CSV等)
                temp_col = df.columns[0] # 仮
                weight_col = df.columns[1] # 仮
    
    if temp_col is None or weight_col is None:
        return None, f"温度または重量の列を特定できませんでした。(検出列: {list(df.columns)})"

    # 数値変換 (エラー値はNaNに)
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
    
    # NaN行を削除
    df = df.dropna(subset=[temp_col, weight_col])
    
    # 解析用にカラム名を統一
    df_out = pd.DataFrame({
        'Temp': df[temp_col],
        'Weight': df[weight_col]
    })
    
    # --- 正規化 ---
    if df_out.empty:
        return None, "有効なデータ行がありません。"
        
    w0 = df_out['Weight'].iloc[0]
    if w0 == 0:
        df_out['Weight_Norm'] = 0
    else:
        df_out['Weight_Norm'] = (df_out['Weight'] / w0) * 100
        
    return df_out, None

# ---------------------------------------------------------
# メインアプリケーション
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="TGA Analysis Tool", layout="wide")
    st.title("TGA Analysis Tool")
    st.markdown("Rigaku形式 (.txt) や CSV/Excel の熱重量分析データをプロット・解析します。")

    # --- サイドバー設定 ---
    st.sidebar.header("読み込み設定")
    
    use_auto_header = st.sidebar.checkbox("ヘッダー位置を自動検出", value=True)
    
    manual_skip = 0
    if not use_auto_header:
        manual_skip = st.sidebar.number_input(
            "ヘッダーまでのスキップ行数", 
            min_value=0, 
            max_value=300, # 上限を拡大
            value=0,
            step=1,
            help="ファイルの先頭からデータ開始位置までの行数を指定します。"
        )

    st.sidebar.markdown("---")
    uploaded_files = st.sidebar.file_uploader(
        "ファイルをアップロード", 
        type=['csv', 'txt', 'xlsx'], 
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("サイドバーからファイルをアップロードしてください。")
        return

    # データ処理ループ
    data_dict = {}
    
    for file in uploaded_files:
        df, error = load_and_normalize_data(file, manual_skip, use_auto_header)
        if df is not None:
            data_dict[file.name] = df
        else:
            st.error(f"{file.name}: {error}")

    if not data_dict:
        return

    # --- 1. プロット ---
    st.subheader("1. データ比較プロット")
    
    palette = px.colors.qualitative.Plotly
    styles = {name: palette[i % len(palette)] for i, name in enumerate(data_dict.keys())}

    fig = go.Figure()
    for filename, df in data_dict.items():
        fig.add_trace(go.Scatter(
            x=df['Temp'],
            y=df['Weight_Norm'],
            mode='lines',
            name=filename,
            line=dict(color=styles[filename])
        ))

    fig.update_layout(
        xaxis_title="Temperature (°C)",
        yaxis_title="Weight (%)",
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 2. 解析 ---
    st.subheader("2. 数値解析 (線形補間)")
    
    c1, c2 = st.columns(2)
    
    # 指定温度 -> 重量
    with c1:
        st.markdown("##### 指定温度における重量残存率 (%)")
        t_target = st.number_input("温度 (°C)", value=500.0, step=10.0)
        res_a = []
        for name, df in data_dict.items():
            if t_target < df['Temp'].min() or t_target > df['Temp'].max():
                val = "範囲外"
            else:
                val = f"{np.interp(t_target, df['Temp'], df['Weight_Norm']):.2f}"
            res_a.append({"File": name, "Weight (%)": val})
        st.table(pd.DataFrame(res_a))

    # 指定重量 -> 温度
    with c2:
        st.markdown("##### 指定重量残存率における温度 (°C)")
        w_target = st.number_input("重量 (%)", value=95.0, step=1.0)
        res_b = []
        for name, df in data_dict.items():
            # 重量でソートしないとinterpできない
            df_s = df.sort_values('Weight_Norm')
            if w_target < df_s['Weight_Norm'].min() or w_target > df_s['Weight_Norm'].max():
                val = "範囲外"
            else:
                val = f"{np.interp(w_target, df_s['Weight_Norm'], df_s['Temp']):.2f}"
            res_b.append({"File": name, "Temp (°C)": val})
        st.table(pd.DataFrame(res_b))

if __name__ == "__main__":
    main()