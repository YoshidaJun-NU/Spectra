import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

# ページ設定
st.set_page_config(page_title="TGA Plotter", layout="wide")

st.title("TGA データプロッター & 解析ツール")
st.markdown("""
複数のTGAデータを読み込み、プロット・比較・解析を行うツールです。
CSVまたはExcelファイルをアップロードしてください。
""")

# --- サイドバー：データ読み込みと設定 ---
st.sidebar.header("1. データアップロード")
uploaded_files = st.sidebar.file_uploader(
    "ファイルをアップロード (複数可)", 
    type=['csv', 'xlsx', 'txt'], 
    accept_multiple_files=True
)

# データの格納用辞書
data_dict = {}

# デフォルトの色リスト（Plotlyのデフォルトなど）
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
LINE_STYLES = ['solid', 'dash', 'dot', 'dashdot']

if uploaded_files:
    st.sidebar.markdown("---")
    st.sidebar.header("2. データ読み込み設定")
    
    # 共通のカラム名設定（簡易化のため、全ファイル共通と仮定して入力を促す）
    # 実際の運用ではファイルごとに推定ロジックを入れると親切です
    with st.sidebar.expander("カラム設定 (全ファイル共通)", expanded=True):
        st.info("データに含まれる列名を指定してください")
        temp_col_name = st.text_input("温度の列名 (Temperature)", value="Temperature")
        weight_col_name = st.text_input("重量の列名 (Weight)", value="Weight")
        
        # エンコーディング選択 (日本語データ対策)
        encoding_option = st.selectbox("文字コード", ["utf-8", "shift_jis", "cp932"], index=0)

    # ファイル読み込み処理
    for uploaded_file in uploaded_files:
        try:
            # 拡張子で判定
            if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, encoding=encoding_option)
            else:
                df = pd.read_excel(uploaded_file)
            
            # 列名のクリーニング（空白削除など）
            df.columns = [c.strip() for c in df.columns]
            
            # 指定された列があるか確認
            if temp_col_name in df.columns and weight_col_name in df.columns:
                # データをソート（温度順）
                df = df.sort_values(by=temp_col_name)
                data_dict[uploaded_file.name] = df
            else:
                st.sidebar.error(f"{uploaded_file.name}: 指定された列名が見つかりません。")
                st.sidebar.write(f"検出された列名: {list(df.columns)}")
                
        except Exception as e:
            st.sidebar.error(f"{uploaded_file.name} の読み込みに失敗しました: {e}")

# --- メインエリア：プロット設定と表示 ---

if data_dict:
    # 3. プロット対象と正規化の選択
    st.subheader("設定 & プロット")
    
    col_control1, col_control2 = st.columns([1, 2])
    
    with col_control1:
        st.markdown("#### 表示データの選択")
        selected_files = st.multiselect(
            "プロットするファイルを選択",
            options=list(data_dict.keys()),
            default=list(data_dict.keys())
        )
        
        st.markdown("#### データ処理")
        normalize_option = st.checkbox("重量%に変換する (初期値を100%とする)", value=True)
        initial_weight_input = 0.0
        
        if normalize_option:
            calc_method = st.radio(
                "初期重量($W_0$)の決定方法", 
                ["データの最初の値を使用", "手動で入力"]
            )
            if calc_method == "手動で入力":
                initial_weight_manual = st.number_input("初期重量 (mg等)", value=10.0)

    # 4. スタイル設定（各ファイルごと）
    with col_control2:
        st.markdown("#### グラフスタイル設定")
        style_expander = st.expander("詳細なスタイル設定 (色・線種)", expanded=False)
        
        styles = {}
        with style_expander:
            st.write("ファイルごとのスタイルを指定できます")
            for i, filename in enumerate(selected_files):
                cols = st.columns(4)
                cols[0].write(f"**{filename}**")
                
                # デフォルト色の割り当て（ループ）
                default_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                
                color = cols[1].color_picker(f"色 ({i})", value=default_color, key=f"c_{i}")
                line_style = cols[2].selectbox(f"線種 ({i})", LINE_STYLES, index=0, key=f"ls_{i}")
                width = cols[3].number_input(f"太さ ({i})", value=2.0, min_value=0.5, step=0.5, key=f"w_{i}")
                
                styles[filename] = {"color": color, "dash": line_style, "width": width}

    # 5. グラフ描画
    fig = go.Figure()

    for filename in selected_files:
        df = data_dict[filename]
        x_data = df[temp_col_name]
        y_data = df[weight_col_name]
        
        # 重量%への変換ロジック
        if normalize_option:
            if calc_method == "データの最初の値を使用":
                w0 = y_data.iloc[0]
            else:
                w0 = initial_weight_manual
            
            # 0除算回避
            if w0 != 0:
                y_data = (y_data / w0) * 100
        
        # スタイルの取得
        style = styles.get(filename, {"color": DEFAULT_COLORS[0], "dash": "solid", "width": 2})

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=filename,
            line=dict(
                color=style['color'],
                dash=style['dash'],
                width=style['width']
            )
        ))

    # 軸範囲の設定
    st.markdown("#### 軸範囲の設定")
    range_col1, range_col2 = st.columns(2)
    with range_col1:
        x_min_input = st.number_input("横軸 (Temp) Min", value=0)
        x_max_input = st.number_input("横軸 (Temp) Max", value=800)
    with range_col2:
        y_min_input = st.number_input("縦軸 (Weight %) Min", value=0)
        y_max_input = st.number_input("縦軸 (Weight %) Max", value=110)

    fig.update_layout(
        title="TGA Curve",
        xaxis_title="Temperature (℃)",
        yaxis_title="Weight (%)" if normalize_option else "Weight (raw)",
        xaxis=dict(range=[x_min_input, x_max_input]),
        yaxis=dict(range=[y_min_input, y_max_input]),
        template="plotly_white",
        height=600,
        hovermode="x unified" # ホバー時に全系列の値を表示
    )

    st.plotly_chart(fig, use_container_width=True)

    # 6. 2点間の差分解析
    st.markdown("---")
    st.subheader("解析: 2点間の重量差算出")
    st.info("指定した2つの温度における重量（%）の差を計算します。分解量などの確認に使用できます。")

    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    temp1 = analysis_col1.number_input("温度点 1 (℃)", value=100.0, step=10.0)
    temp2 = analysis_col2.number_input("温度点 2 (℃)", value=500.0, step=10.0)
    
    if st.button("差分を計算"):
        results = []
        for filename in selected_files:
            df = data_dict[filename]
            
            # データ再構築（プロット時と同じ変換を行う）
            x_vals = df[temp_col_name].values
            y_vals = df[weight_col_name].values
            
            if normalize_option:
                if calc_method == "データの最初の値を使用":
                    w0 = y_vals[0]
                else:
                    w0 = initial_weight_manual
                if w0 != 0:
                    y_vals = (y_vals / w0) * 100
            
            # 線形補間で指定温度の値を取得するための関数
            def get_val_at_temp(target_t, x_arr, y_arr):
                # 範囲外チェック
                if target_t < x_arr.min() or target_t > x_arr.max():
                    return None
                # 最も近いインデックスを探す（簡易的）または補間
                # ここでは正確性を出すためnumpyのinterpを使用
                import numpy as np
                return np.interp(target_t, x_arr, y_arr)

            val1 = get_val_at_temp(temp1, x_vals, y_vals)
            val2 = get_val_at_temp(temp2, x_vals, y_vals)
            
            if val1 is not None and val2 is not None:
                diff = val1 - val2
                results.append({
                    "ファイル名": filename,
                    f"{temp1}℃での重量(%)": f"{val1:.2f}",
                    f"{temp2}℃での重量(%)": f"{val2:.2f}",
                    "差分 (Weight Loss)": f"{diff:.2f} %"
                })
            else:
                results.append({
                    "ファイル名": filename,
                    "結果": "指定温度がデータ範囲外です"
                })
        
        st.table(pd.DataFrame(results))

else:
    st.info("サイドバーからデータをアップロードしてください。")

# フッター
st.markdown("---")
st.caption("Developed with Streamlit & Plotly")