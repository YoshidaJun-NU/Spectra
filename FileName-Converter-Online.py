import streamlit as st
import io
import zipfile
from pathlib import Path

# ページ設定
st.set_page_config(layout="wide", page_title="ファイル名変換アプリ (Web版)")

st.title("📂 ファイル名変換アプリ (Web版)")
st.markdown("""
サーバー上で動作するため、以下の手順で使用してください。
1. 名前を変えたいファイルを**アップロード**します。
2. 下の設定で新しい名前を決めます。
3. **「変換してダウンロード」**ボタンを押すと、リネームされたファイルがZIPで保存されます。
""")

# --- サイドバー設定 ---
st.sidebar.header("設定")

# 1. ファイルアップロード
uploaded_files = st.sidebar.file_uploader(
    "ファイルをアップロードしてください (複数可)", 
    accept_multiple_files=True
)

st.sidebar.markdown("---")

# 2. モード選択
mode = st.sidebar.radio("モード選択", ["自動モード (温度追加)", "手動モード (個別編集)"])

# --- メイン処理 ---

if uploaded_files:
    st.success(f"{len(uploaded_files)} 個のファイルを読み込みました。")
    
    # ファイル名リストを作成（アップロードされたファイルオブジェクトから名前を取得）
    # 並び順を安定させるためにファイル名でソート
    files_sorted = sorted(uploaded_files, key=lambda x: x.name)

    # 変換結果を格納するリスト [(original_file_obj, new_filename), ...]
    rename_pairs = []

    st.markdown("---")

    # --- リスト表示と設定 ---

    if mode == "自動モード (温度追加)":
        st.info("区切り文字、温度、単位を設定してください。")
        
        # ヘッダー (5列構成)
        c_name, c_sep, c_temp, c_unit, c_prev = st.columns([3, 0.8, 1, 0.8, 3])
        c_name.markdown("### 📄 元のファイル名")
        c_sep.markdown("### 区切り")
        c_temp.markdown("### 温度")
        c_unit.markdown("### 単位")
        c_prev.markdown("### 📝 変更後 (プレビュー)")

        for i, file_obj in enumerate(files_sorted):
            c_l, c_sep, c_temp, c_unit, c_r = st.columns([3, 0.8, 1, 0.8, 3])
            
            # 1. 元のファイル名
            c_l.text(file_obj.name)
            
            # パス操作用にPathオブジェクト化（名前のみ）
            p = Path(file_obj.name)

            # 2. 区切り文字
            sep_val = c_sep.selectbox(
                "区切り", 
                ["-", "_", "(なし)"], 
                index=0,
                key=f"sep_{i}", 
                label_visibility="collapsed"
            )
            
            # 3. 温度
            temp_val = c_temp.number_input(
                "温度", 
                value=50, 
                step=1, 
                key=f"temp_{i}", 
                label_visibility="collapsed"
            )
            
            # 4. 単位
            unit_val = c_unit.text_input(
                "単位", 
                value="℃", 
                key=f"unit_{i}", 
                label_visibility="collapsed"
            )
            
            # プレビュー生成
            display_sep = "" if sep_val == "(なし)" else sep_val
            new_name = f"{p.stem}{display_sep}{temp_val}{unit_val}{p.suffix}"
            
            # 5. 変更後プレビュー
            c_r.code(new_name, language="text")
            
            rename_pairs.append((file_obj, new_name))

    else:
        # 手動モード
        st.info("右側のファイル名を直接編集してください。")
        c1, c2 = st.columns(2)
        c1.markdown("### 📄 元のファイル名")
        c2.markdown("### 📝 変更後のファイル名")

        for i, file_obj in enumerate(files_sorted):
            c_l, c_r = st.columns(2)
            c_l.text(file_obj.name)
            
            new_name_input = c_r.text_input(
                "新しいファイル名",
                value=file_obj.name,
                key=f"manual_{i}",
                label_visibility="collapsed"
            )
            rename_pairs.append((file_obj, new_name_input))

    # --- ZIP作成とダウンロードボタン ---
    st.markdown("---")
    
    # ボタンを押す前からZIP作成用関数を準備
    def create_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            seen_names = set()
            for file_obj, new_name in rename_pairs:
                # 重複回避ロジック（簡易版）
                if new_name in seen_names:
                    base, ext = os.path.splitext(new_name)
                    counter = 1
                    while f"{base}_{counter}{ext}" in seen_names:
                        counter += 1
                    new_name = f"{base}_{counter}{ext}"
                
                seen_names.add(new_name)
                
                # アップロードされたファイルの中身を読み込む
                file_obj.seek(0)
                data = file_obj.read()
                
                # ZIPに書き込む（新しい名前で）
                zf.writestr(new_name, data)
        
        return zip_buffer.getvalue()

    # ダウンロードボタン
    if rename_pairs:
        zip_data = create_zip()
        st.download_button(
            label="📥 変換してZIPでダウンロード",
            data=zip_data,
            file_name="renamed_files.zip",
            mime="application/zip",
            type="primary"
        )

else:
    st.info("👈 左側のサイドバーからファイルをアップロードしてください。")