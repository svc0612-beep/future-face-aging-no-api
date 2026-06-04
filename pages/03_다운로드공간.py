from datetime import datetime
from pathlib import Path
import re

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "outputs" / "download_space"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="다운로드공간",
    page_icon="⬇️",
    layout="wide",
)


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]', "_", name).strip()
    return cleaned or "uploaded_file"


st.title("다운로드공간")
st.caption("파일을 업로드해 보관하고, 나중에 다시 다운로드할 수 있는 공간입니다.")

with st.sidebar:
    st.header("페이지 구성")
    st.page_link("app.py", label="홈")
    st.page_link("pages/01_시각화.py", label="시각화")
    st.page_link("pages/02_용어정리.py", label="용어정리")
    st.page_link("pages/03_다운로드공간.py", label="다운로드공간")
    st.page_link("pages/04_현재작업.py", label="현재작업")
    st.page_link("pages/05_프로젝트흐름.py", label="프로젝트흐름")
    st.divider()
    st.caption("저장 폴더")
    st.code(str(DOWNLOAD_DIR))


uploaded_files = st.file_uploader(
    "보관할 파일을 업로드하세요.",
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = safe_filename(uploaded_file.name)
        save_path = DOWNLOAD_DIR / f"{timestamp}_{filename}"
        save_path.write_bytes(uploaded_file.getbuffer())
        st.success(f"업로드 완료: {save_path.name}")

st.divider()
st.subheader("보관된 파일")

files = sorted(
    [path for path in DOWNLOAD_DIR.iterdir() if path.is_file()],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not files:
    st.info("아직 업로드된 파일이 없습니다.")
else:
    for path in files:
        size_kb = path.stat().st_size / 1024
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.write(path.name)
            st.caption(f"수정 시간: {modified}")
        with cols[1]:
            st.write(f"{size_kb:,.1f} KB")
        with cols[2]:
            st.download_button(
                "다운로드",
                data=path.read_bytes(),
                file_name=path.name,
                key=f"download_{path.name}",
                width="stretch",
            )

