import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ---------------------------------------------------------
# 1. データ読み込みと前処理 (Normalization & Encoding Fix)
# ---------------------------------------------------------
def load_and_normalize_data(uploaded_file, skip_rows):
    """
    ファイルを読み込み、初期重量を基準に正規化を行う関数
    Weight(%) = (Wt / W0) * 100
    対応: 複数のエンコーディング試行、ヘッダー行スキップ
    """
    file_ext = uploaded_file.name.split('.')[-1].lower()
    df = None
    
    # 試行するエンコーディングのリスト
    # 日本の装置は 'shift_jis' や 'cp932' が多い。海外製は 'latin1' など。
    encodings_to_try = ['utf-8', 'shift_jis', 'cp932', 'latin1']
    
    try:
        if file_ext == 'xlsx':
            # Excelはエンコーディング指定不要だが、skiprowsは必要
            df = pd.read_excel(uploaded_file, skiprows=skip_rows)
        else:
            # CSV / TXT の場合、エンコーディングを順番に試す
            for enc in encodings_to_try:
                try:
                    # Streamlitの特性上、読み込み直前にポインタを先頭に戻す必要がある
                    uploaded_file.seek(0)
                    
                    # 読み込み試行
                    df = pd.read_csv(
                        uploaded_file, 
                        sep=None,          # 区切り文字自動判定
                        engine='python', 
                        encoding=enc,      # エンコーディング指定
                        skiprows=skip_rows # ヘッダー行スキップ
                    )
                    # 成功したらループを抜ける
                    break
                except UnicodeDecodeError:
                    # このエンコーディングで失敗したら次へ
                    continue
                except pd.errors.ParserError:
                    # パースエラーの場合も次へ（区切り文字等の問題の可能性）
                    continue
            
            # 全てのエンコーディングで失敗した場合
            if df is None:
                st.error(f"Error: ファイル {uploaded_file.name} を読み込めませんでした。文字コードまたは形式を確認してください。")
                return None
            
    except Exception as e:
        st.error(f"予期せぬエラー ({uploaded_file.name}): {e}")
        return None
    
    # --- カラム名の簡易クレンジングと特定 ---
    # 読み込んだデータの列名を確認し、Temp/Weightを特定する
    # ユーザーのファイルに合わせてロジックを調整してください
    
    # すべての列名を文字列型に変換（混在回避）
    df.columns = df.columns.astype(str)
    
    # 必須カラムが含まれているかチェック (大文字小文字を無視して探す)
    # ここでは仮に、列名に "Temp" や "Weight" (または "mg", "%") が含まれているかを探す簡易ロジック
    # 実際のデータに合わせて、より厳密に指定することも可能です
    
    # もしカラム名が見つからない、またはデータ構造が単純な場合
    # 「1列目を温度、2列目を重量」として強制的にリネームする処理
    if df.shape[1] >= 2:
        # 元の列名を保持しつつ、内部処理用にリネーム
        # 実際の解析では、どの列が温度でどの列が重量かを選択させるUIがあるとベスト
        df.columns.values[0] = 'Temp'
        df.columns.values[1] = 'Weight'
        # 残りの列はそのまま
    else:
        st.warning(f"{uploaded_file.name}: 列数が不足しています（2列以上必要）。skiprowsの設定を確認してください。")
        return None

    # 数値型への変換 (文字列として読み込まれている場合への対処)
    # エラー('errors="coerce"')が発生した値はNaNになる
    df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    
    # NaNを含む行（ヘッダーの残りなど）を削除
    df = df.dropna(subset=['Temp', 'Weight'])

    # --- [機能: データの正規化] ---
    if df.empty:
        st.warning(f"{uploaded_file.name}: 有効なデータ行がありません。")
        return None

    # 初期値 (W0) を取得
    w0 = df['Weight'].iloc[0]
    
    if w0 == 0:
        df['Weight_Norm'] = 0
    else:
        df['Weight_Norm'] = (df['Weight'] / w0) * 100
    
    return df

# ---------------------------------------------------------
# メインアプリケーション
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="TGA Analysis Tool", layout="wide")
    st.title("TGA Analysis: Visualization & Interpolation")

    # サイドバー設定
    st.sidebar.header("Settings")
    
    # 1. ヘッダー行のスキップ設定 (データ読み込み前に必要)
    # テキストファイルの上部に測定条件などが書かれている場合、これを調整してデータ開始行を指定します
    skip_rows = st.sidebar.number_input(
        "ヘッダーの行数をスキップ (Skip Rows)", 
        min_value=0, 
        max_value=100, 
        value=0,
        help="データの開始行がずれている場合、この数値を増やしてください。"
    )

    # 2. ファイルアップロード
    uploaded_files = st.sidebar.file_uploader(
        "データファイルをアップロード (csv, txt, xlsx)", 
        type=['csv', 'txt', 'xlsx'], 
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("左のサイドバーからTGAデータファイルをアップロードしてください。")
        st.markdown("""
        **ヒント:** - テキストファイルの読み込みエラーが出る場合は、**「ヘッダーの行数をスキップ」** の値を増やしてみてください。
        - 日本語(Shift-JIS)のファイルも自動判定して読み込みます。
        """)
        return

    # データを格納する辞書
    data_dict = {}
    
    # ファイル読み込みループ
    for file in uploaded_files:
        # ここで skip_rows を渡す
        df = load_and_normalize_data(file, skip_rows)
        if df is not None:
            data_dict[file.name] = df

    if not data_dict:
        return

    # ---------------------------------------------------------
    # 2. 柔軟なスタイル設定
    # ---------------------------------------------------------
    st.subheader("1. Comparison Plot (Normalized)")
    
    palette = px.colors.qualitative.Plotly
    styles = {}
    for i, filename in enumerate(data_dict.keys()):
        color = palette[i % len(palette)]
        styles[filename] = {'color': color}

    fig = go.Figure()

    for filename, df in data_dict.items():
        fig.add_trace(go.Scatter(
            x=df['Temp'],
            y=df['Weight_Norm'],
            mode='lines',
            name=filename,
            line=dict(color=styles[filename]['color'])
        ))

    fig.update_layout(
        xaxis_title="Temperature (°C)",
        yaxis_title="Weight (%)",
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 3. 2点間解析
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Precise Analysis (Linear Interpolation)")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### A. 指定温度 → 重量(%)")
        target_temp = st.number_input("温度を入力 (°C)", value=300.0, step=10.0)
        
        results_a = []
        for filename, df in data_dict.items():
            if target_temp < df['Temp'].min() or target_temp > df['Temp'].max():
                calc_weight_str = "Out of Range"
            else:
                calc_weight = np.interp(target_temp, df['Temp'], df['Weight_Norm'])
                calc_weight_str = f"{calc_weight:.2f}"
            
            results_a.append({"File": filename, "Weight (%)": calc_weight_str})
        
        st.table(pd.DataFrame(results_a))

    with col2:
        st.markdown("#### B. 指定重量(%) → 温度")
        target_weight = st.number_input("重量(%)を入力", value=95.0, step=1.0)
        
        results_b = []
        for filename, df in data_dict.items():
            # 重量の昇順ソート（補間のため）
            df_sorted = df.sort_values(by='Weight_Norm')
            
            # 範囲外チェック
            if target_weight < df_sorted['Weight_Norm'].min() or target_weight > df_sorted['Weight_Norm'].max():
                calc_temp_str = "Out of Range"
            else:
                calc_temp = np.interp(target_weight, df_sorted['Weight_Norm'], df_sorted['Temp'])
                calc_temp_str = f"{calc_temp:.2f}"
                
            results_b.append({"File": filename, "Temp (°C)": calc_temp_str})
            
        st.table(pd.DataFrame(results_b))

if __name__ == "__main__":
    main()