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
    # 拡張子に応じた読み込み
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # カラム名の簡易クレンジング (実運用に合わせて調整してください)
    # ここでは仮に 'Temp' と 'Weight' というカラムが含まれていると想定します
    # 実際にはユーザーにカラムを選択させるUIを追加するのがベストです
    
    # 必須カラムの確認 (デモ用にカラム名が存在するかチェック)
    if 'Temp' not in df.columns or 'Weight' not in df.columns:
        # カラムがない場合、1列目をTemp, 2列目をWeightとみなす救済措置
        df.columns = ['Temp', 'Weight'] + list(df.columns[2:])

    # --- [機能追加: データの正規化] ---
    # 初期値 (W0) を取得 (通常は測定開始時の重量)
    w0 = df['Weight'].iloc[0]
    
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
    uploaded_files = st.sidebar.file_uploader(
        "CSV/Excelファイルをアップロード (複数可)", 
        type=['csv', 'xlsx'], 
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
        data_dict[file.name] = df

    # ---------------------------------------------------------
    # 2. 柔軟なスタイル設定 (Flexible Style Settings)
    # ---------------------------------------------------------
    st.subheader("1. Comparison Plot (Normalized)")
    
    # --- [機能追加: 動的なスタイル辞書の作成] ---
    # Plotlyのデフォルトカラーパレットを取得
    palette = px.colors.qualitative.Plotly
    
    # ファイル名ごとに色を割り当てる辞書を作成
    styles = {}
    for i, filename in enumerate(data_dict.keys()):
        color = palette[i % len(palette)]  # ファイル数が増えても色を循環させる
        styles[filename] = {'color': color, 'line_dash': 'solid'}

    # プロット作成
    fig = go.Figure()

    for filename, df in data_dict.items():
        fig.add_trace(go.Scatter(
            x=df['Temp'],
            y=df['Weight_Norm'],
            mode='lines',
            name=filename,
            line=dict(color=styles[filename]['color']) # 動的スタイル適用
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
    st.markdown(
        """
        グラフ上のクリックではなく、**数値計算(np.interp)** によって値を算出します。
        - **Temp → Weight**: 指定した温度での重量残存率を計算
        - **Weight → Temp**: 指定した重量減少率(例: 95% = 5%減)になる温度を計算
        """
    )

    col1, col2 = st.columns(2)

    # --- 解析モードA: 温度から重量を算出 ---
    with col1:
        st.markdown("#### A. 指定温度における重量(%)")
        target_temp = st.number_input("温度を入力 (°C)", value=300.0, step=10.0)
        
        st.write(f"**Target Temperature: {target_temp} °C**")
        results_a = []
        
        for filename, df in data_dict.items():
            # --- [機能追加: 線形補間 np.interp] ---
            # x: Temp, y: Weight_Norm
            calc_weight = np.interp(target_temp, df['Temp'], df['Weight_Norm'])
            results_a.append({"File": filename, "Weight (%)": f"{calc_weight:.2f}"})
        
        st.table(pd.DataFrame(results_a))

    # --- 解析モードB: 重量から温度を算出 (例: 5%分解温度) ---
    with col2:
        st.markdown("#### B. 指定重量(%)における温度")
        target_weight = st.number_input("重量(%)を入力 (例: 95% = 5%減量)", value=95.0, step=1.0)
        
        st.write(f"**Target Weight: {target_weight} %**")
        results_b = []
        
        for filename, df in data_dict.items():
            # --- [機能追加: 線形補間 np.interp (逆引き)] ---
            # 重量から温度を求める場合、x(Weight)は単調増加である必要があるためソートして処理
            # TGAでは重量は減少していくため、データを逆順にするかソートが必要
            
            df_sorted = df.sort_values(by='Weight_Norm') # 重量の昇順にソート
            
            # x: Weight_Norm (Sorted), y: Temp
            calc_temp = np.interp(target_weight, df_sorted['Weight_Norm'], df_sorted['Temp'])
            results_b.append({"File": filename, "Temp (°C)": f"{calc_temp:.2f}"})
            
        st.table(pd.DataFrame(results_b))

if __name__ == "__main__":
    main()