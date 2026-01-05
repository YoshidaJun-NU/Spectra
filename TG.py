import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# 日本語フォント設定（Windows環境などを想定）
# 環境に合わせてフォントファミリーは変更が必要な場合があります
plt.rcParams['font.family'] = 'Meiryo'

def parse_asc_file(uploaded_file):
    """
    提供されたASC形式のファイルを解析してDataFrameを返す関数
    """
    # バイナリとして読み込み、Shift-JISでデコード（日本語パスなどが含まれるため）
    content = uploaded_file.getvalue().decode('shift_jis', errors='ignore')
    lines = content.splitlines()
    
    header_line_index = -1
    data_start_index = -1
    headers = []
    
    # 行を走査してヘッダーとデータの開始位置を探す
    for i, line in enumerate(lines):
        # 列名定義行を探す (Time, Tempなどが含まれる行)
        if "Time" in line and "Temp" in line and not line.startswith('#'):
            header_line_index = i
            # タブ区切りでヘッダーを取得（先頭の空白除去）
            headers = [h.strip() for h in line.strip().split('\t') if h.strip()]
            continue
        
        # データ開始行 (#GD) を探す
        if line.startswith('#GD'):
            data_start_index = i
            break
            
    if header_line_index == -1 or data_start_index == -1:
        st.error("ASCファイルのフォーマットを解析できませんでした。")
        return None

    # データを抽出
    data_rows = []
    for line in lines[data_start_index:]:
        if line.startswith('#GD'):
            # #GDを除去し、タブで分割
            raw_values = line.replace('#GD', '').strip().split('\t')
            # 数値変換
            try:
                values = [float(x) for x in raw_values if x.strip()]
                data_rows.append(values)
            except ValueError:
                continue

    # DataFrame作成 (ヘッダーと列数が合わない場合の調整)
    df = pd.DataFrame(data_rows)
    
    # ヘッダー割り当て（列数が合う範囲で）
    if len(headers) == df.shape[1]:
        df.columns = headers
    else:
        st.warning(f"ヘッダーの列数({len(headers)})とデータの列数({df.shape[1]})が一致しません。自動割り当てを行います。")
        # 可能な限り割り当て
        df.columns = headers[:df.shape[1]]

    return df

def main():
    st.title("熱分析データ (TG/DTA) 可視化ツール")
    st.write("ASCファイルまたはCSVファイルをアップロードしてください。")

    # 1. ファイルアップロード
    uploaded_file = st.file_uploader("ファイルを選択", type=['asc', 'csv', 'txt'])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # 2. ファイル読み込み処理
        try:
            if file_ext == 'csv':
                # CSVの場合（ヘッダーがある前提）
                df = pd.read_csv(uploaded_file)
            else:
                # ASCファイルの場合
                df = parse_asc_file(uploaded_file)
        except Exception as e:
            st.error(f"ファイルの読み込みエラー: {e}")
            return

        if df is None or df.empty:
            st.error("データフレームが空です。")
            return

        st.write("### 読み込みデータプレビュー")
        st.dataframe(df.head())

        # 3. 列の選択設定（サイドバー）
        st.sidebar.header("プロット設定")
        
        # 列名のリスト
        columns = df.columns.tolist()
        
        # デフォルト値の推測
        default_x = [c for c in columns if "Temp" in c][0] if any("Temp" in c for c in columns) else columns[0]
        default_tg = [c for c in columns if "TG" in c and "DTG" not in c][0] if any("TG" in c and "DTG" not in c for c in columns) else columns[1] if len(columns) > 1 else columns[0]
        default_dta = [c for c in columns if "DTA" in c and "DDTA" not in c][0] if any("DTA" in c and "DDTA" not in c for c in columns) else columns[2] if len(columns) > 2 else columns[0]

        # ユーザー選択
        x_col = st.sidebar.selectbox("X軸 (温度など)", columns, index=columns.index(default_x))
        tg_col = st.sidebar.selectbox("TGデータ列 (%)", columns, index=columns.index(default_tg))
        dta_col = st.sidebar.selectbox("DTAデータ列 (μV)", columns, index=columns.index(default_dta))

        # 微分曲線の設定
        st.sidebar.markdown("---")
        st.sidebar.subheader("微分曲線の表示")
        show_dtg = st.sidebar.checkbox("DTG (TGの微分) を表示", value=True)
        show_ddta = st.sidebar.checkbox("DDTA (DTAの微分) を表示", value=False)

        # 4. 微分データの準備
        # データセットにDTG/DDTAが既に含まれているか確認し、なければ計算する
        
        # DTG
        dtg_data = None
        dtg_label = "DTG (calc)"
        existing_dtg = [c for c in columns if "DTG" in c]
        if existing_dtg:
             use_existing_dtg = st.sidebar.checkbox(f"ファイル内の {existing_dtg[0]} を使用", value=True)
             if use_existing_dtg:
                 dtg_data = df[existing_dtg[0]]
                 dtg_label = existing_dtg[0]
        
        if dtg_data is None and show_dtg:
            # 微分計算 (dy/dx)
            dtg_data = df[tg_col].diff() / df[x_col].diff()
            # ノイズ軽減のため移動平均をかける場合（オプション）
            # dtg_data = dtg_data.rolling(window=5).mean()

        # DDTA
        ddta_data = None
        ddta_label = "DDTA (calc)"
        existing_ddta = [c for c in columns if "DDTA" in c]
        if existing_ddta:
             use_existing_ddta = st.sidebar.checkbox(f"ファイル内の {existing_ddta[0]} を使用", value=True)
             if use_existing_ddta:
                 ddta_data = df[existing_ddta[0]]
                 ddta_label = existing_ddta[0]

        if ddta_data is None and show_ddta:
            # 微分計算
            ddta_data = df[dta_col].diff() / df[x_col].diff()

        # 5. プロット描画
        st.write("### グラフ")
        
        # グラフの作成 (2軸プロット)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- 第1軸 (左): TG ---
        color_tg = 'tab:red'
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(tg_col, color=color_tg)
        l1, = ax1.plot(df[x_col], df[tg_col], color=color_tg, label=tg_col, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color_tg)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        lines = [l1]

        # --- 第2軸 (右): DTA ---
        ax2 = ax1.twinx()  # X軸を共有
        color_dta = 'tab:blue'
        ax2.set_ylabel(dta_col, color=color_dta)
        l2, = ax2.plot(df[x_col], df[dta_col], color=color_dta, label=dta_col, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color_dta)
        lines.append(l2)

        # --- 微分曲線の追加 (軸は適宜調整、今回は第3, 第4の軸を作ると複雑になるため、既存軸に重ねるかオフセットしますが、
        # 見やすさのため、DTGはTG軸(左)に関連付け、DDTAはDTA軸(右)に関連付けるか、
        # あるいは「正規化」して表示するのが一般的ですが、ここではスケールが違うため
        # 簡易的に twinx をさらに追加して表示します (Matplotlibのparasite axes的なアプローチ)
        
        # DTGのプロット (破線で表示)
        if show_dtg and dtg_data is not None:
            # DTG用に新しい軸を作る（TGと同じ左側だがスケールが違うため）
            # 視認性を上げるため、右側に軸を追加して対応
            ax3 = ax1.twinx()
            # 右側の軸の位置を少し外側にずらす
            ax3.spines["right"].set_position(("axes", 1.15))
            
            color_dtg = 'salmon'
            ax3.set_ylabel(dtg_label, color=color_dtg)
            l3, = ax3.plot(df[x_col], dtg_data, color=color_dtg, linestyle='--', label=dtg_label, alpha=0.8)
            ax3.tick_params(axis='y', labelcolor=color_dtg)
            lines.append(l3)

        # DDTAのプロット (点線で表示)
        if show_ddta and ddta_data is not None:
            ax4 = ax1.twinx()
            # 右側の軸の位置をさらに外側にずらす
            ax4.spines["right"].set_position(("axes", 1.3))
            
            color_ddta = 'skyblue'
            ax4.set_ylabel(ddta_label, color=color_ddta)
            l4, = ax4.plot(df[x_col], ddta_data, color=color_ddta, linestyle=':', label=ddta_label, alpha=0.8)
            ax4.tick_params(axis='y', labelcolor=color_ddta)
            lines.append(l4)

        # 凡例をまとめて表示
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        st.pyplot(fig)

        # データダウンロード
        st.write("### データのエクスポート")
        csv = df.to_csv(index=False).encode('utf-8_sig')
        st.download_button(
            "CSVとしてダウンロード",
            csv,
            "processed_data.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == '__main__':
    main()