# ---------------------------------------------------------
# データ読み込み (エラー対策を強化)
# ---------------------------------------------------------
def load_data(uploaded_files):
    data_list = []
    for f in uploaded_files:
        try:
            content = f.getvalue()
            for enc in ['utf-8', 'cp932', 'shift_jis']:
                try: text = content.decode(enc); break
                except: continue
            
            lines = text.splitlines()
            # 初期値を設定（KeyErrorを防ぐ）
            x_unit, y_unit = "Wavenumber (cm-1)", "Intensity"
            use_skip = 0
            
            # ヘッダー解析
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
            
            sep = ',' if f.name.lower().endswith('.csv') else None
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=use_skip, header=None, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] >= 2:
                data_list.append({
                    'label': f.name.rsplit('.', 1)[0],
                    'x': df.iloc[:, 0].values,
                    'y': df.iloc[:, 1].values,
                    # 必ず辞書にキーを含める
                    'x_unit': x_unit,
                    'y_unit': y_unit
                })
        except Exception as e:
            st.error(f"{f.name} の読み込みに失敗しました: {e}")
    return data_list

# ---------------------------------------------------------
# メインアプリ（プロット部分の修正）
# ---------------------------------------------------------
def main():
    # ... (前略: サイドバー設定などはそのまま) ...

    if selected:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 表示対象のデータを取り出し
        display_data = [d for d in st.session_state['data_list'] if d['label'] in selected]
        
        for item in display_data:
            x, y = item['x'], item['y'].copy()

            # ベースライン補正などの処理 (省略)

            # プロット
            ax.plot(x, y, label=item['label'], alpha=0.8)

            # フィッティング処理など (省略)

        # --- エラー回避用の軸ラベル設定 ---
        # item.get('key', 'default') を使うことで、キーがなくてもエラーになりません
        if display_data:
            last_item = display_data[-1]
            ax.set_xlabel(last_item.get('x_unit', 'Wavenumber (cm-1)'))
            ax.set_ylabel(last_item.get('y_unit', 'Intensity'))

        if invert_x: ax.invert_xaxis()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)