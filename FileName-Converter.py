import streamlit as st
import os
import platform
import subprocess
from pathlib import Path

# ページ設定
st.set_page_config(layout="wide", page_title="ファイル名変換アプリ")

st.title("📂 ローカルファイル名変換アプリ")

# --- 関数: ローカルフォルダを開く ---
def open_local_folder(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", path])
        else:  # Linux
            subprocess.Popen(["xdg-open", path])
        st.toast(f"フォルダを開きました: {path}")
    except Exception as e:
        st.error(f"フォルダを開けませんでした: {e}")

# --- サイドバー設定 ---
st.sidebar.header("設定")

# 1. フォルダパスの入力
folder_path = st.sidebar.text_input(
    "対象フォルダのパスを入力してください",
    placeholder=r"C:\Users\Name\Documents\TargetFolder"
)

# 2. フォルダを開くボタン
if folder_path and os.path.isdir(folder_path):
    if st.sidebar.button("📁 このフォルダを開く"):
        open_local_folder(folder_path)

# 3. モード選択
st.sidebar.markdown("---")
mode = st.sidebar.radio("モード選択", ["自動モード (温度追加)", "手動モード (個別編集)"])

# 変数の初期化
file_list = []
final_rename_pairs = [] # 実行用のリスト [(path_obj, new_name_str), ...]

# --- メイン処理 ---

if folder_path:
    target_dir = Path(folder_path)

    # フォルダの存在確認
    if not target_dir.exists():
        st.error("指定されたフォルダが見つかりません。パスを確認してください。")
    elif not target_dir.is_dir():
        st.error("指定されたパスはフォルダではありません。")
    else:
        # ファイル一覧の取得（隠しファイル除外）
        file_list = sorted([f for f in target_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
        
        if not file_list:
            st.warning("フォルダ内にファイルがありません。")
        else:
            st.sidebar.success(f"{len(file_list)} 個のファイルを検出しました。")

            st.markdown("---")

            # --- モード別の表示処理 ---

            if mode == "自動モード (温度追加)":
                st.info("中列の数値を変更すると、右側のファイル名に反映されます（形式: `元の名前` + `数値℃` + `拡張子`）。")
                
                # ヘッダー (3列)
                c1, c2, c3 = st.columns([3, 1, 3])
                c1.markdown("### 📄 現在のファイル名")
                c2.markdown("### 🌡️ 温度設定")
                c3.markdown("### 📝 変更後 (プレビュー)")

                # リスト表示
                for i, f in enumerate(file_list):
                    col_l, col_m, col_r = st.columns([3, 1, 3])
                    
                    # 左: 現在のファイル名
                    col_l.text(f.name)
                    
                    # 中: 温度入力 (デフォルト50)
                    temp_val = col_m.number_input(
                        "温度", 
                        value=50, 
                        step=1, 
                        key=f"temp_{i}", 
                        label_visibility="collapsed"
                    )
                    
                    # 右: 変更後のファイル名生成
                    new_name = f"{f.stem}{temp_val}℃{f.suffix}"
                    col_r.code(new_name, language="text")
                    
                    final_rename_pairs.append((f, new_name))

            else:
                # 手動モードの場合
                st.info("右側の入力欄で新しいファイル名を自由に編集してください。")
                
                # ヘッダー (2列)
                c1, c2 = st.columns(2)
                c1.markdown("### 📄 現在のファイル名")
                c2.markdown("### 📝 変更後のファイル名")

                for i, f in enumerate(file_list):
                    col_l, col_r = st.columns(2)
                    
                    # 左: 現在のファイル名
                    col_l.text(f.name)
                    
                    # 右: 手動入力
                    new_name_input = col_r.text_input(
                        "新しいファイル名",
                        value=f.name,
                        key=f"manual_{i}",
                        label_visibility="collapsed"
                    )
                    final_rename_pairs.append((f, new_name_input))

            # --- 実行ボタン ---
            st.markdown("---")
            if st.button("変換を実行する", type="primary"):
                success_count = 0
                error_log = []
                
                # プログレスバー
                progress_bar = st.progress(0)

                for idx, (original_path, new_filename) in enumerate(final_rename_pairs):
                    try:
                        # 名前が変わっていない場合はスキップ
                        if original_path.name == new_filename:
                            continue

                        new_path = original_path.parent / new_filename
                        
                        # 既に同名ファイルが存在する場合のチェック
                        if new_path.exists():
                            error_log.append(f"スキップ: '{new_filename}' は既に存在します。")
                            continue
                        
                        # リネーム実行
                        original_path.rename(new_path)
                        success_count += 1
                        
                    except Exception as e:
                        error_log.append(f"エラー ({original_path.name}): {e}")
                    
                    # プログレスバー更新
                    progress_bar.progress((idx + 1) / len(final_rename_pairs))

                # 結果表示
                if success_count > 0:
                    st.success(f"{success_count} 個のファイル名を変更しました！")
                    st.balloons()
                    # 画面更新ボタン
                    if st.button("一覧を更新"):
                        st.experimental_rerun()
                elif len(error_log) == 0:
                    st.info("変更が必要なファイルはありませんでした。")

                if error_log:
                    st.error("以下のエラーが発生しました:")
                    for err in error_log:
                        st.write(f"- {err}")

else:
    st.info("左側のサイドバーから対象のフォルダパスを入力してください。")