import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import io

# ---------------------------------------------------------
# 関数定義: データの読み込み
# ---------------------------------------------------------
def load_data(uploaded_files, separator, header_row):
    data_list = []
    
    for uploaded_file in uploaded_files:
        try:
            # 拡張子や選択された区切り文字で読み込み
            if separator == 'comma':
                sep_char = ','
            else: # tab
                sep_char = '\t'
            
            # データの読み込み
            df = pd.read_csv(uploaded_file, sep=sep_char, header=header_row)
            
            # 列名の正規化（1列目X, 2列目Yと仮定）
            # 実際にはデータに合わせて調整が必要ですが、汎用的にilocを使います
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
            # ファイル名（拡張子なし）を取得
            label = uploaded_file.name.rsplit('.', 1)[0]
            
            data_list.append({
                'label': label,
                'x': x,
                'y': y,
                'df_raw': df # Gnuplot出力用に保持
            })
            
        except Exception as e:
            st.error(f"エラー: {uploaded_file.name} を読み込めませんでした。({e})")
    
    return data_list

# ---------------------------------------------------------
# 関数定義: Gnuplot用データの作成 (結合データ)
# ---------------------------------------------------------
def create_gnuplot_data(data_list):
    if not data_list:
        return None
    
    # 全データを波長(x)をキーにして結合する処理
    # 基準となるDataFrameを作成
    df_merged = pd.DataFrame({'Wavelength': data_list[0]['x'], data_list[0]['label']: data_list[0]['y']})
    
    for item in data_list[1:]:
        df_temp = pd.DataFrame({'Wavelength': item['x'], item['label']: item['y']})
        # 波長で外部結合（波長が微妙にずれていてもデータが消えないように）
        df_merged = pd.merge(df_merged, df_temp, on='Wavelength', how='outer')
    
    # 波長でソート
    df_merged = df_merged.sort_values('Wavelength')
    
    # CSV (Space separated for Gnuplot)
    return df_merged.to_csv(sep='\t', index=False, float_format='%.4f')

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Plotter", layout="wide")

    st.title("UV-vis Spectra Viewer & Converter")
    st.markdown("複数のスペクトルデータをプロットし、Gnuplot形式で保存できます。")

    # --- サイドバー：設定 ---
    st.sidebar.header("1. データ設定")
    separator = st.sidebar.radio("区切り文字", ('comma', 'tab'), index=0, format_func=lambda x: "カンマ (CSV)" if x=='comma' else "タブ (TXT)")
    header_row = st.sidebar.number_input("ヘッダー行番号 (0始まり)", value=0, min_value=0)

    st.sidebar.header("2. グラフ設定")
    # カラーマップ選択
    cmap_options = ['viridis', 'jet', 'coolwarm', 'rainbow', 'plasma', 'Manual']
    cmap_name = st.sidebar.selectbox("カラーマップ", cmap_options, index=0)
    
    # 凡例位置
    legend_loc = st.sidebar.radio("凡例の位置", ('Outside (外側)', 'Inside (内側)'))
    
    # 軸設定
    x_label = st.sidebar.text_input("X軸ラベル", "Wavelength (nm)")
    y_label = st.sidebar.text_input("Y軸ラベル", "Abs.")
    
    # 範囲設定
    use_manual_range = st.sidebar.checkbox("軸範囲を手動設定する")
    x_min, x_max, y_min, y_max = None, None, None, None
    if use_manual_range:
        col1, col2 = st.sidebar.columns(2)
        x_min = col1.number_input("X Min", value=200)
        x_max = col2.number_input("X Max", value=800)
        y_min = col1.number_input("Y Min", value=0.0)
        y_max = col2.number_input("Y Max", value=1.5)

    # --- メインエリア：ファイルアップロード ---
    uploaded_files = st.file_uploader("ファイルをここにドラッグ＆ドロップ (複数可)", accept_multiple_files=True, type=['csv', 'txt'])

    if uploaded_files:
        data_list = load_data(uploaded_files, separator, header_row)
        
        if data_list:
            # --- プロット処理 ---
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 色の生成
            num_files = len(data_list)
            if cmap_name == 'Manual':
                # シンプルな手動設定例（必要に応じて拡張可能）
                base_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
                colors = base_colors * (num_files // len(base_colors) + 1)
            else:
                cmap = plt.get_cmap(cmap_name)
                colors = [cmap(i) for i in np.linspace(0, 1, num_files)]

            # プロットループ
            for i, item in enumerate(data_list):
                ax.plot(item['x'], item['y'], label=item['label'], color=colors[i], linewidth=1.5, alpha=0.8)

            # グラフ装飾
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.tick_params(direction='out', length=6, width=1)
            ax.grid(True, linestyle=':', alpha=0.5)

            if use_manual_range:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

            # 凡例
            if legend_loc == 'Outside (外側)':
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            else:
                ax.legend(loc='best')

            st.pyplot(fig)

            # --- ダウンロードエリア ---
            st.markdown("---")
            st.subheader("📥 ダウンロード")

            col_d1, col_d2 = st.columns(2)

            # 1. 画像として保存
            fn = "spectra_plot.png"
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
            img.seek(0)
            col_d1.download_button(
                label="画像をダウンロード (PNG)",
                data=img,
                file_name=fn,
                mime="image/png"
            )

            # 2. Gnuplot形式で保存
            gnuplot_data = create_gnuplot_data(data_list)
            if gnuplot_data:
                col_d2.download_button(
                    label="Gnuplot用データをダウンロード (.dat)",
                    data=gnuplot_data,
                    file_name="spectra_data.dat",
                    mime="text/plain"
                )
                
                # Gnuplot用スクリプトのヒント表示
                with st.expander("Gnuplot用のスクリプト例を表示"):
                    plot_cmd = "plot "
                    for i in range(len(data_list)):
                        # 列番号はGnuplotでは1始まり。1列目がX, 2列目以降が各データ
                        # data_listの順番通りに列が結合されていると仮定
                        col_idx = i + 2 
                        title = data_list[i]['label']
                        plot_cmd += f"'spectra_data.dat' using 1:{col_idx} with lines title '{title}', \\\n     "
                    
                    st.code(f"""
# gnuplot script example
set terminal pngcairo size 800,600
set output 'plot.png'
set xlabel "{x_label}"
set ylabel "{y_label}"
set grid
set key outside

{plot_cmd.strip().rstrip(', \\')}
                    """, language='gnuplot')

if __name__ == "__main__":
    main()