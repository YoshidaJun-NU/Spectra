import streamlit as st
import os
import sys
import platform
import subprocess
from pathlib import Path

# --- Tkinterの環境設定（Windows用エラー回避） ---
if platform.system() == "Windows":
    try:
        base_path = os.path.dirname(sys.executable)
        tcl_path = os.path.join(base_path, 'tcl', 'tcl8.6')
        tk_path = os.path.join(base_path, 'tcl', 'tk8.6')
        if os.path.exists(tcl_path) and os.path.exists(tk_path):
            os.environ['TCL_LIBRARY'] = tcl_path
            os.environ['TK_LIBRARY'] = tk_path
    except Exception:
        pass

import tkinter as tk
from tkinter import filedialog

# ページ設定
st.set_page_config(layout="wide", page_title="ファイル名変換アプリ")

st.title("📂 ローカルファイル名変換アプリ")

# --- session_state の初期化 ---
if "folder_path" not in st.session_state:
    st.session_state["folder_path"] = ""

# --- 関数: フォルダ選択ダイアログを開く ---
def select_folder_dialog():
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        
        folder_selected = filedialog.askdirectory(master=root)
        
        root.destroy()
        
        if folder_selected:
            st.session_state["folder_path"] = folder_selected
            st.rerun()
    except Exception as e:
        st.error(f"フォルダ選択ダイアログの起動に失敗しました: {e}")
        st.error("以下の手動入力欄を使用してください。")

# --- 関数: フォルダを開く ---
def open_local_folder(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        st.error(f"フォルダを開けませんでした: {e}")

# --- サイドバー ---
st.sidebar.header("設定")
st.sidebar.info("下のボタンを押して対象フォルダを選択してください。")

if st.sidebar.button("📂 フォルダを選択する"):
    select_folder_dialog()

folder_path = st.sidebar.text_input(
    "選択されたパス:",
    value=st.session_state["folder_path"],
    placeholder="フォルダが未選択です"
)
if folder_path != st.session_state["folder_path"]:
    st.session_state["folder_path"] = folder_path

if st.session_state["folder_path"]:
    if st.sidebar.button("👀 選択したフォルダを開く"):
        open_local_folder(st.session_state["folder_path"])

st.sidebar.markdown("---")
mode = st.sidebar.radio("モード選択", ["自動モード (温度追加)", "手動モード (個別編集)"])


# --- メイン処理 ---
if folder_path:
    target_dir = Path(folder_path)

    if not target_dir.exists():
        st.error("指定されたフォルダが見つかりません。")
    elif not target_dir.is_dir():
        st.error("指定されたパスはフォルダではありません。")
    else:
        file_list = sorted([f for f in target_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
        
        if not file_list:
            st.warning("フォルダ内にファイルがありません。")
        else:
            st.sidebar.success(f"{len(file_list)} 個のファイルを検出しました。")
            
            # --- 変換実行ボタン ---
            if st.button("変換を実行する", type="primary"):
                success_count = 0
                error_log = []
                progress_bar = st.progress(0)
                
                for idx, f in enumerate(file_list):
                    try:
                        new_filename = ""
                        
                        if mode == "自動モード (温度追加)":
                            # session_stateから各値を取得
                            temp_val = st.session_state.get(f"temp_{idx}", 50)
                            sep_val = st.session_state.get(f"sep_{idx}", "-")
                            unit_val = st.session_state.get(f"unit_{idx}", "℃")
                            
                            # 区切り文字が「(なし)」の場合は空文字にする
                            if sep_val == "(なし)":
                                sep_val = ""
                                
                            # ファイル名生成: 元の名前 + 区切り + 温度 + 単位 + 拡張子
                            new_filename = f"{f.stem}{sep_val}{temp_val}{unit_val}{f.suffix}"
                            
                        else:
                            # 手動モード
                            manual_val = st.session_state.get(f"manual_{idx}", f.name)
                            new_filename = manual_val

                        # 変更がない場合はスキップ
                        if f.name == new_filename:
                            continue

                        new_path = f.parent / new_filename
                        
                        if new_path.exists():
                            error_log.append(f"スキップ (重複): {new_filename}")
                            continue
                        
                        f.rename(new_path)
                        success_count += 1
                        
                    except Exception as e:
                        error_log.append(f"エラー ({f.name}): {e}")
                    
                    progress_bar.progress((idx + 1) / len(file_list))

                if success_count > 0:
                    st.success(f"{success_count} 個のファイル名を変更しました！")
                    st.balloons()
                    st.rerun()
                elif not error_log:
                    st.info("変更が必要なファイルはありませんでした。")
                
                if error_log:
                    st.error("以下のエラーが発生しました:")
                    for err in error_log:
                        st.write(f"- {err}")

            st.markdown("---")

            # --- リスト表示 ---
            
            if mode == "自動モード (温度追加)":
                st.info("区切り文字、温度、単位を設定してください。")
                
                # ヘッダー (5列構成)
                # 比率: [名前:3, 区切り:1, 温度:1, 単位:1, プレビュー:3]
                c_name, c_sep, c_temp, c_unit, c_prev = st.columns([3, 0.8, 1, 0.8, 3])
                c_name.markdown("### 📄 現在のファイル名")
                c_sep.markdown("### 区切り")
                c_temp.markdown("### 温度")
                c_unit.markdown("### 単位")
                c_prev.markdown("### 📝 変更後")

                for i, f in enumerate(file_list):
                    col_l, col_sep, col_temp, col_unit, col_r = st.columns([3, 0.8, 1, 0.8, 3])
                    
                    # 1. 現在のファイル名
                    col_l.text(f.name)
                    
                    # 2. 区切り文字 (デフォルト: - )
                    default_sep = st.session_state.get(f"sep_{i}", "-")
                    sep_val = col_sep.selectbox(
                        "区切り", 
                        ["-", "_", "(なし)"], 
                        index=["-", "_", "(なし)"].index(default_sep) if default_sep in ["-", "_", "(なし)"] else 0,
                        key=f"sep_{i}", 
                        label_visibility="collapsed"
                    )
                    
                    # 3. 温度 (デフォルト: 50)
                    default_temp = st.session_state.get(f"temp_{i}", 50)
                    temp_val = col_temp.number_input(
                        "温度", 
                        value=default_temp, 
                        step=1, 
                        key=f"temp_{i}", 
                        label_visibility="collapsed"
                    )
                    
                    # 4. 単位 (デフォルト: ℃, 手動入力可)
                    default_unit = st.session_state.get(f"unit_{i}", "℃")
                    unit_val = col_unit.text_input(
                        "単位", 
                        value=default_unit, 
                        key=f"unit_{i}", 
                        label_visibility="collapsed"
                    )
                    
                    # プレビュー用ロジック
                    display_sep = "" if sep_val == "(なし)" else sep_val
                    new_name = f"{f.stem}{display_sep}{temp_val}{unit_val}{f.suffix}"
                    
                    # 5. 変更後プレビュー
                    col_r.code(new_name, language="text")

            else:
                # 手動モード
                st.info("右側のファイル名を直接編集してください。")
                c1, c2 = st.columns(2)
                c1.markdown("### 📄 現在のファイル名")
                c2.markdown("### 📝 変更後のファイル名")

                for i, f in enumerate(file_list):
                    col_l, col_r = st.columns(2)
                    col_l.text(f.name)
                    
                    default_name = st.session_state.get(f"manual_{i}", f.name)
                    col_r.text_input("新しいファイル名", value=default_name, key=f"manual_{i}", label_visibility="collapsed")

else:
    st.info("サイドバーの「フォルダを選択する」ボタンを押して開始してください。")