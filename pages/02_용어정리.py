from base64 import b64encode
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "model_summary"
GLOSSARY_PATH = REPORT_DIR / "project_glossary_page.html"


st.set_page_config(
    page_title="용어정리",
    page_icon="📘",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def embed_local_images(html: str) -> str:
    image_names = [
        "age_val_mae_comparison.png",
        "age_val_loss_comparison.png",
        "age_model_best_mae_bar.png",
        "age_test_mae_improvement.png",
        "gender_training_metrics.png",
    ]
    for image_name in image_names:
        image_path = REPORT_DIR / image_name
        if image_path.exists():
            encoded = b64encode(image_path.read_bytes()).decode("ascii")
            html = html.replace(
                f'src="{image_name}"',
                f'src="data:image/png;base64,{encoded}"',
            )
    return html


st.title("용어정리")
st.caption("프로젝트에서 사용한 모델, 데이터, 학습 지표, 그래프 해석을 정리한 페이지입니다.")

with st.sidebar:
    st.header("페이지 구성")
    st.page_link("app.py", label="홈")
    st.page_link("pages/01_시각화.py", label="시각화")
    st.page_link("pages/02_용어정리.py", label="용어정리")
    st.page_link("pages/03_다운로드공간.py", label="다운로드공간")
    st.page_link("pages/05_프로젝트흐름.py", label="프로젝트흐름")

if not GLOSSARY_PATH.exists():
    st.error("용어 정리 HTML 파일을 찾을 수 없습니다.")
    st.code(str(GLOSSARY_PATH))
    st.stop()

html = embed_local_images(GLOSSARY_PATH.read_text(encoding="utf-8"))
components.html(html, height=1400, scrolling=True)

