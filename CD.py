import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ---------------------------------------------------------
# 関数定義: CD用ダミーデータの生成
# ---------------------------------------------------------
def generate_cd_dummy_data():
    """動作確認用にCDスペクトル（正負あり）を作成する"""
    data_list = []
    x = np.linspace(200, 350, 300)
    
    # パターン1: Type A
    y1 = 20 * np.exp(-((x - 280)**2) / (2 * 10**2)) - 10 * np.exp(-((x - 220)**2) / (2 * 15**2))
    y1 += np.random.normal(0, 0.2, len(x))
    
    # パターン2: Type B
    y2 = -15 * np.exp(-((x - 280)**2) / (2 * 10**2)) + 12 * np.exp(-((x - 225)**2) / (2 * 15**2))
    y2 += np.random.normal(0, 0.2, len(x))

    # パターン3: Type C (Flat)
    y3 = 5 * np.sin((x - 200)/20) * np.exp(-((x - 250)**2) / (2 * 50**2))

    data_list.append({'label': 'Type_A_Protein', 'x': x, 'y': y1})
    data_list.append({'label': 'Type_B_Mutant', 'x': x, 'y': y2})
    data_list.append({'label': 'Type_C_Buffer', 'x': x, 'y': y3})
    
    return data_list

# ---------------------------------------------------------
# 関数定義: ファイル読み込み
# ---------------------------------------------------------
def load_data(uploaded_files, separator, skip_rows, has_header):
    data_list = []
    for uploaded_file in uploaded_files:
        try:
            sep_char = ',' if separator == 'comma' else '\t'
            header_setting = 0 if has_header else None
            df = pd.read_csv(uploaded_file, sep=sep_char, skiprows=skip_rows, header=header_setting)
            
            x = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
            y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            
            label = uploaded_file.name.rsplit('.', 1)[0]
            data_list.append({'label': label, 'x': x, 'y': y})
        except Exception as e:
            st.error(f"エラー: {uploaded_file.name} を読み込めませんでした。\n{e}")
    return data_list

# ---------------------------------------------------------
# 関数定義: Gnuplotデータ作成
# ---------------------------------------------------------
def create_gnuplot_data(data_list):
    if not data_list: return None
    df_merged = pd.DataFrame({'Wavelength': data_list[0]['x'], data_list[0]['label']: data_list[0]['y']})
    for item in data_list[1:]:
        df_temp = pd.DataFrame({'Wavelength': item['x'], item['label']: item['y']})
        df_merged = pd.merge(df_merged, df_temp, on='Wavelength', how='outer')
    return df_merged.sort_values('Wavelength').to_csv(sep='\t', index=False, float_format='%.4f')

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="CD Spectra Plotter", layout="wide")
    st.title("CD Spectra Plotter (Circular Dichroism)")

    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- 1. データ読み込み ---
    st.sidebar.header("1. データ読み込み")
    if st.sidebar.button("サンプルデータをロード (3種)"):
        st.session_state['data_list'] = generate_cd_dummy_data()
        st.sidebar.success("サンプルデータを生成しました")

    st.sidebar.markdown("---")
    separator = st.sidebar.radio("区切り文字", ('comma', 'tab'), format_func=lambda x: "タブ (TXT)" if x=='tab' else "カンマ (CSV)")
    skip_rows = st.sidebar.number_input("スキップ行数", value=19, min_value=0)
    has_header = st.sidebar.checkbox("ヘッダーあり", value=True)
    
    uploaded_files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files=True)
    if uploaded_files:
        st.session_state['data_list'] = load_data(uploaded_files, separator, skip_rows, has_header)

    # --- 2. グラフ設定 ---
    st.sidebar.header("2. グラフ設定")
    
    data_list = st.session_state['data_list']
    
    # 凡例位置
    legend_loc = st.sidebar.radio("凡例の位置", ('Inside (図中)', 'Outside (外側)'))

    # スタイル設定モード
    style_mode = st.sidebar.selectbox(
        "配色・スタイル設定", 
        ["Auto (Distinct Colors)", "CoolWarm (Gradation)", "Manual (個別設定)"]
    )

    # 線種の定義辞書
    line_style_dict = {
        '実線 (Solid)': '-',
        '破線 (Dashed)': '--',
        '点線 (Dotted)': ':',
        '一点鎖線 (Dash-dot)': '-.'
    }

    # プロット用の設定リストを作成
    plot_settings = []

    if data_list:
        if style_mode == "Manual (個別設定)":
            st.sidebar.markdown("### 個別ライン設定")
            st.sidebar.info("各データの色、線種、太さを設定できます。")
            
            # デフォルト色リスト
            default_cols = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
            
            # 各データごとにExpanderを作る、または並べる
            for i, item in enumerate(data_list):
                with st.sidebar.expander(f"{i+1}. {item['label']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    # 色設定
                    c_val = default_cols[i % len(default_cols)]
                    color = col1.color_picker("色", c_val, key=f"c_{i}")
                    
                    # 太さ設定
                    width = col2.number_input("太さ", value=2.0, step=0.5, key=f"w_{i}")
                    
                    # 線種設定
                    s_key = st.selectbox("線種", list(line_style_dict.keys()), key=f"s_{i}")
                    style = line_style_dict[s_key]
                    
                    plot_settings.append({'color': color, 'ls': style, 'lw': width})
        
        else:
            # 自動モードの場合の設定生成
            for i in range(len(data_list)):
                # 色の計算
                if style_mode == "Auto (Distinct Colors)":
                    c = plt.cm.tab10(i % 10)
                else: # CoolWarm
                    c = plt.cm.coolwarm(i / max(len(data_list)-1, 1))
                
                plot_settings.append({'color': c, 'ls': '-', 'lw': 2.0}) # デフォルトは実線・太さ2

    # 軸設定
    st.sidebar.subheader("軸とラベル")
    x_label = st.sidebar.text_input("X軸ラベル", "Wavelength (nm)")
    y_label = st.sidebar.text_input("Y軸ラベル", "Ellipticity (mdeg)")
    
    use_manual_range = st.sidebar.checkbox("軸範囲を手動設定")
    x_min, x_max, y_min, y_max = None, None, None, None
    if use_manual_range:
        c1, c2 = st.sidebar.columns(2)
        x_min = c1.number_input("X Min", value=200.0)
        x_max = c2.number_input("X Max", value=350.0)
        y_min = c1.number_input("Y Min", value=-20.0)
        y_max = c2.number_input("Y Max", value=20.0)

    # --- 3. プロット描画 ---
    if data_list:
        st.subheader("CD Spectra Overlay")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)

        # ループ描画
        for i, item in enumerate(data_list):
            settings = plot_settings[i]
            ax.plot(
                item['x'], 
                item['y'], 
                label=item['label'], 
                color=settings['color'], 
                linestyle=settings['ls'],   # 線種
                linewidth=settings['lw'],   # 太さ
                alpha=0.9
            )

        # 装飾
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(direction='out', top=False, right=False)
        
        if use_manual_range:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
        if legend_loc == 'Outside (外側)':
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        else:
            ax.legend(loc='best', frameon=True, framealpha=0.9)
            
        plt.tight_layout()
        st.pyplot(fig)

        # --- 4. ダウンロード ---
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        img_png = io.BytesIO()
        plt.savefig(img_png, format='png', bbox_inches='tight', dpi=300)
        img_png.seek(0)
        col1.download_button("画像 (PNG)", img_png, "cd_spectra.png", "image/png")
        
        img_tiff = io.BytesIO()
        plt.savefig(img_tiff, format='tiff', bbox_inches='tight', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        img_tiff.seek(0)
        col2.download_button("画像 (TIFF)", img_tiff, "cd_spectra.tiff", "image/tiff")
        
        gnu_data = create_gnuplot_data(data_list)
        if gnu_data:
            col3.download_button("Gnuplotデータ (.dat)", gnu_data, "cd_data.dat", "text/plain")
            
    else:
        st.info("👈 左側の「サンプルデータをロード」を押すか、ファイルをアップロードしてください。")

if __name__ == "__main__":
    main()