import streamlit as st
import pandas as pd
import plotly.express as px

# ページ設定（ワイド表示）
st.set_page_config(page_title="Scientific Data Plotter", layout="wide")

st.title("📊 Scientific Data Plotter")
st.markdown("CSVファイルをアップロードしてプロットを表示します。")

# --- サイドバー設定 ---
st.sidebar.header("設定")

# ファイルアップロード機能
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

# 表示オプション
st.sidebar.subheader("グラフオプション")
xlabel = st.sidebar.text_input("X軸ラベル", value="Chemical Shift / Potential / Time")
ylabel = st.sidebar.text_input("Y軸ラベル", value="Intensity / Current")
invert_x = st.sidebar.checkbox("X軸を反転する (例: NMR)", value=False)
skip_rows = st.sidebar.number_input("スキップする行数 (ヘッダー等)", min_value=0, value=1)

# --- メイン処理 ---
if uploaded_file is not None:
    try:
        # データの読み込み
        # ユーザー指定の行数をスキップし、ヘッダーなしとして読み込む
        df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None)
        
        # 2列以上のデータがあるか確認
        if df.shape[1] >= 2:
            # 1列目をX, 2列目をYとする
            df.columns = ['X', 'Y'] + [f'Col_{i}' for i in range(2, df.shape[1])]
            
            # --- プロット作成 (Plotly) ---
            fig = px.line(df, x='X', y='Y', title=uploaded_file.name)
            
            # 軸ラベルの設定
            fig.update_layout(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                hovermode="x unified"
            )
            
            # X軸反転処理
            if invert_x:
                fig.update_xaxes(autorange="reversed")
            
            # グラフの表示
            st.plotly_chart(fig, use_container_width=True)
            
            # 生データの表示（折りたたみ）
            with st.expander("生データを確認する"):
                st.dataframe(df)
                
        else:
            st.error("エラー: データが2列以上必要です。")
            
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
else:
    # ファイルがない場合の案内
    st.info("👈 サイドバーからCSVファイルをアップロードしてください。")
    st.markdown("""
    **テスト用ファイルの仕様:**
    - 1列目: X軸データ
    - 2列目: Y軸データ
    - 1行目にメタデータがある場合は設定でスキップ可能です。
    """)