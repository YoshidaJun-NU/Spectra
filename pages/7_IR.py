import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. モデル関数定義
# ---------------------------------------------------------
def multi_gaussian(x, *params):
    """
    複数のガウス関数の和を計算
    params: [amp1, cen1, sig1, amp2, cen2, sig2, ..., offset]
    """
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        amp, cen, sigma = params[i], params[i+1], params[i+2]
        y += amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
    y += params[-1] # offset
    return y

# ---------------------------------------------------------
# 2. データ読み込み (KeyError対策版)
# ---------------------------------------------------------
def load_data(uploaded_files):
    data_list = []
    for f in uploaded_files:
        try:
            content = f.getvalue()
            # 文字コードの判定
            for enc in ['utf-8', 'cp932', 'shift_jis', 'latin1']:
                try:
                    text = content.decode(enc)
                    break
                except:
                    continue
            
            lines = text.splitlines()
            # デフォルト値の設定（KeyError防止）
            x_unit, y_unit = "Wavelength / Wavenumber", "Intensity"
            use_skip = 0
            
            # JASCOヘッダー解析
            for i, line in enumerate(lines):
                if 'XUNITS' in line:
                    val = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                    if val: x_unit = val
                if 'YUNITS' in line:
                    val = line.split(',')[-1].strip() or line.split('\t')[-1].strip()
                    if val: y_unit = val
                if 'XYDATA' in line:
                    use_skip = i + 1
                    break
            
            # セパレータ（カンマかタブか）の自動判定
            sep = ',' if f.name.lower().endswith('.csv') else None
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=use_skip, header=None, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] >= 2:
                data_list.append({
                    'label': f.name.rsplit('.', 1)[0],
                    'x': df.iloc[:, 0].values,
                    'y': df.iloc[:, 1].values,
                    'x_unit': x_unit,
                    'y_unit': y_unit
                })
        except Exception as e:
            st.error(f"{f.name} の読み込み中にエラーが発生しました: {e}")
    return data_list

# ---------------------------------------------------------
# 3. メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Spectra Analyzer Pro", layout="wide")
    st.title("Spectra Analyzer Pro: IR & UV-Vis 🧪")

    # セッション状態の初期化
    if 'data_list' not in st.session_state:
        st.session_state['data_list'] = []

    # --- サイドバー：1. データロード ---
    st.sidebar.header("1. データ読み込み")
    files = st.sidebar.file_uploader("JASCO形式のCSV/TXTをアップロード", accept_multiple_files=True, type=['csv', 'txt', 'dat'])
    
    if files:
        # 新しくロードしたデータで上書き（KeyError防止のため一度クリアを推奨）
        if st.sidebar.button("データを反映（リセット）"):
            st.session_state['data_list'] = load_data(files)

    # データがない場合は中断
    if not st.session_state['data_list']:
        st.info("👈 左側のサイドバーからスペクトルデータをアップロードしてください。")
        return

    # --- サイドバー：2. 表示・補正設定 ---
    st.sidebar.header("2. 表示・補正設定")
    all_labels = [d['label'] for d in st.session_state['data_list']]
    selected_labels = st.sidebar.multiselect("表示するデータ", all_labels, default=all_labels)
    
    # 軸の反転設定
    invert_x = st.sidebar.checkbox("X軸を逆転させる (IR標準)", value=True)
    
    # ベースライン補正
    st.sidebar.subheader("ベースライン補正")
    bl_mode = st.sidebar.selectbox("モード", ["None", "Constant (一点補正)", "Linear (二点補正)"])
    bl_params = {}
    if bl_mode != "None":
        bl_params['p1'] = st.sidebar.number_input("補正基準点1 (x値)", value=float(st.session_state['data_list'][0]['x'].max()))
        if bl_mode == "Linear":
            bl_params['p2'] = st.sidebar.number_input("補正基準点2 (x値)", value=float(st.session_state['data_list'][0]['x'].min()))

    # --- サイドバー：3. フィッティング ---
    st.sidebar.header("3. マルチガウスフィッティング")
    do_fit = st.sidebar.checkbox("フィッティングを実行")
    num_peaks = st.sidebar.number_input("ピーク数", 1, 10, 1)
    fit_target = st.sidebar.selectbox("対象データ", selected_labels) if selected_labels else None

    # --- グラフ描画エリア ---
    if selected_labels:
        fig, ax = plt.subplots(figsize=(10, 6))
        display_data = [d for d in st.session_state['data_list'] if d['label'] in selected_labels]

        for item in display_data:
            x, y = item['x'], item['y'].copy()

            # ベースライン補正の計算
            if bl_mode == "Constant (一点補正)":
                y -= y[np.abs(x - bl_params['p1']).argmin()]
            elif bl_mode == "Linear (二点補正)":
                i1, i2 = np.abs(x - bl_params['p1']).argmin(), np.abs(x - bl_params['p2']).argmin()
                slope = (y[i2] - y[i1]) / (x[i2] - x[i1])
                y -= (slope * (x - x[i1]) + y[i1])

            # プロット
            ax.plot(x, y, label=item['label'], alpha=0.8)

            # フィッティング実行
            if do_fit and item['label'] == fit_target:
                try:
                    # 初期値の自動推定
                    p0 = []
                    found, _ = find_peaks(y, prominence=0.01)
                    idx_peaks = found[:num_peaks] if len(found) >= num_peaks else np.linspace(0, len(x)-1, num_peaks, dtype=int)
                    for idx in idx_peaks:
                        p0.extend([y[idx], x[idx], 10.0]) # Amp, Center, Sigma
                    p0.append(np.min(y)) # Offset
                    
                    popt, _ = curve_fit(multi_gaussian, x, y, p0=p0)
                    
                    # 合計曲線の描画
                    ax.plot(x, multi_gaussian(x, *popt), 'r--', lw=2, label="Total Fit")
                    
                    # 個別ピークの情報を表示
                    res_list = []
                    for n in range(num_peaks):
                        res_list.append({
                            "Peak": n+1, 
                            "Center": f"{popt[n*3+1]:.2f}", 
                            "FWHM": f"{2.355*abs(popt[n*3+2]):.2f}"
                        })
                    st.sidebar.write("📌 フィッティング結果")
                    st.sidebar.table(pd.DataFrame(res_list))
                except Exception as e:
                    st.sidebar.error(f"Fitting failed: {e}")

        # 軸ラベルの設定 (KeyError対策: .get()を使用)
        if display_data:
            ax.set_xlabel(display_data[0].get('x_unit', "X-axis"))
            ax.set_ylabel(display_data[0].get('y_unit', "Y-axis"))

        if invert_x:
            ax.invert_xaxis()
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        # ダウンロード
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        st.download_button("グラフを保存 (PNG)", buf.getvalue(), "spectra_result.png", "image/png")
    else:
        st.info("表示するデータを選択してください。")

if __name__ == "__main__":
    main()