import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ---------------------------------------------------------
# 1. データ読み込みと前処理 (Normalization)
# ---------------------------------------------------------
def load_and_normalize_data(uploaded_file):
    """
    ファイルを読み込み、初期重量を基準に正規化を行う関数
    Weight(%) = (Wt / W0) * 100
    """
    # 拡張子の判定
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'xlsx':
            # Excelファイルの場合
            df = pd.read_excel(uploaded_file)
        else:
            # CSV または TXT ファイルの場合
            # sep=None, engine='python' を指定すると、カンマ/タブ/スペース等を自動判定して読み込みます
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
            
    except Exception as e:
        st.error(f"ファイル {uploaded_file.name} の読み込みに失敗しました: {e}")
        return None
    
    # --- カラム名の簡易クレンジング ---
    # 空白を除去し、すべて小文字にして扱いやすくする等の処理を入れるとより堅牢になりますが、
    # ここでは既存ロジック通り、必須カラムがない場合の救済措置のみ行います。
    
    if 'Temp' not in df.columns or 'Weight' not in df.columns:
        # カラム名が一致しない場合、1列目をTemp, 2列目をWeightとみなす
        # (装置出力の生データなどはヘッダー行が多い場合があるため、ここを調整する必要があるかもしれません)
        if df.shape[1] >= 2:
            df.columns = ['Temp', 'Weight'] + list(df.columns[2:])
        else:
            st.warning(f"{uploaded_file.name}: データの列数が不足しています。")
            return None

    # --- [機能: データの正規化] ---
    # データが空でないか確認
    if df.empty:
        return None

    # 初期値 (W0) を取得
    w0 = df['Weight'].iloc[0]
    
    # 0除算回避
    if w0 == 0:
        st.warning(f"{uploaded_file.name}: 初期重量が0のため正規化できません。")
        df['Weight_Norm'] = 0
    else:
        # 正規化計算
        df['Weight_Norm'] = (df['Weight'] / w0) * 100
    
    return df

# ---------------------------------------------------------
# メインアプリケーション
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="TGA Analysis Tool", layout="wide")
    st.title("TGA Analysis: Visualization & Interpolation")
    st.markdown("### 機能概要: 正規化プロットと高精度な数値解析")

    # サイドバー: ファイルアップロード
    st.sidebar.header("Data Upload")
    
    # --- [変更点] typeに 'txt' を追加 ---
    uploaded_files = st.sidebar.file_uploader(
        "データファイルをアップロード (csv, txt, xlsx)", 
        type=['csv', 'txt', 'xlsx'], 
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("左のサイドバーからTGAデータファイルをアップロードしてください。")
        return

    # データを格納する辞書
    data_dict = {}
    
    # ファイル読み込みループ
    for file in uploaded_files:
        df = load_and_normalize_data(file)
        if df is not None:
            data_dict[file.name] = df

    if not data_dict:
        st.error("有効なデータが読み込めませんでした。")
        return

    # ---------------------------------------------------------
    # 2. 柔軟なスタイル設定 (Flexible Style Settings)
    # ---------------------------------------------------------
    st.subheader("1. Comparison Plot (Normalized)")
    
    # Plotlyのデフォルトカラーパレットを取得
    palette = px.colors.qualitative.Plotly
    
    # ファイル名ごとに色を割り当てる辞書を作成
    styles = {}
    for i, filename in enumerate(data_dict.keys()):
        color = palette[i % len(palette)]
        styles[filename] = {'color': color}

    # プロット作成
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
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 3. 2点間解析: 線形補間 (Interpolation Logic)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Precise Analysis (Linear Interpolation)")
    
    col1, col2 = st.columns(2)

    # --- 解析モードA: 温度から重量を算出 ---
    with col1:
        st.markdown("#### A. 指定温度における重量(%)")
        target_temp = st.number_input("温度を入力 (°C)", value=300.0, step=10.0)
        
        st.write(f"**Target Temperature: {target_temp} °C**")
        results_a = []
        
        for filename, df in data_dict.items():
            calc_weight = np.interp(target_temp, df['Temp'], df['Weight_Norm'])
            results_a.append({"File": filename, "Weight (%)": f"{calc_weight:.2f}"})
        
        st.table(pd.DataFrame(results_a))

    # --- 解析モードB: 重量から温度を算出 ---
    with col2:
        st.markdown("#### B. 指定重量(%)における温度")
        target_weight = st.number_input("重量(%)を入力", value=95.0, step=1.0)
        
        st.write(f"**Target Weight: {target_weight} %**")
        results_b = []
        
        for filename, df in data_dict.items():
            # 重量の昇順にソートして補間
            df_sorted = df.sort_values(by='Weight_Norm')
            calc_temp = np.interp(target_weight, df_sorted['Weight_Norm'], df_sorted['Temp'])
            results_b.append({"File": filename, "Temp (°C)": f"{calc_temp:.2f}"})
            
        st.table(pd.DataFrame(results_b))

if __name__ == "__main__":
    main()