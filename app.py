from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from PIL import Image
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.face_preprocessing import FaceNotDetectedError, crop_largest_face
from src.predict_age import load_age_model, make_future_ages, predict_age_from_pil
from src.predict_gender import load_gender_model, predict_gender_from_pil
from src.sam_aging_simulator import generate_future_faces_sam


APP_TITLE = "API 없는 안면 에이징 - 재미로만 봐주세요"
YEARS_LIST = [10, 20, 30, 40, 50]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "aged_results"
CAPTURE_DIR = PROJECT_ROOT / "outputs" / "captures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="홈",
    page_icon="🏠",
    layout="wide",
)


st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    .app-title {
        color: #172033;
        font-size: 2.2rem;
        font-weight: 850;
        line-height: 1.24;
        margin: 0 0 .35rem;
    }
    .app-subtitle {
        color: #5f6c80;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }
    .step-band {
        border: 1px solid #d8dee8;
        background: #f7f9fc;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: .8rem 0 1rem;
    }
    .server-notice {
        border: 1px solid #b9cff7;
        background: #f3f7ff;
        color: #243554;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 1rem 0;
    }
    .server-notice strong {
        color: #174ea6;
    }
    .server-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .55rem .9rem;
        margin-top: .75rem;
        font-size: .94rem;
    }
    .server-grid div {
        border: 1px solid #d7e2f8;
        background: #ffffff;
        border-radius: 8px;
        padding: .72rem .8rem;
    }
    .step-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .7rem;
    }
    .step-item {
        border: 1px solid #d8dee8;
        background: #fff;
        border-radius: 8px;
        padding: .8rem;
        color: #2a3548;
        min-height: 76px;
    }
    .step-num {
        color: #1f6feb;
        font-size: .82rem;
        font-weight: 800;
        margin-bottom: .2rem;
    }
    .tip-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .45rem .8rem;
        color: #263247;
        font-size: .94rem;
    }
    .metric-card {
        border: 1px solid #d8dee8;
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    .metric-label {
        color: #627086;
        font-size: .9rem;
        margin-bottom: .25rem;
    }
    .metric-value {
        color: #185abc;
        font-size: 2.15rem;
        line-height: 1.1;
        font-weight: 850;
    }
    .metric-sub {
        color: #627086;
        font-size: .9rem;
        margin-top: .35rem;
    }
    .notice {
        border: 1px solid #e5c05c;
        background: #fff8e6;
        color: #5e4a12;
        border-radius: 8px;
        padding: .9rem 1rem;
        margin: 1rem 0;
    }
    .danger-note {
        border: 1px solid #ffc8b2;
        background: #fff3ed;
        color: #8a3412;
        border-radius: 8px;
        padding: .9rem 1rem;
        margin: 1rem 0;
    }
    @media (max-width: 850px) {
        .step-row,
        .tip-grid,
        .server-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    <script>
    const replaceCameraButtonText = () => {
      const buttons = window.parent.document.querySelectorAll('button');
      buttons.forEach((button) => {
        if (button.innerText && button.innerText.trim() === 'Take Photo') {
          button.innerText = '이미지 촬영';
        }
      });
    };
    setInterval(replaceCameraButtonText, 500);
    </script>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_age_model():
    return load_age_model(device=torch.device("cpu"))


@st.cache_resource(show_spinner=False)
def get_gender_model():
    return load_gender_model(device=torch.device("cpu"))


def reset_home_state():
    for key in [
        "captured_image",
        "captured_path",
        "captured_source",
        "analysis_result",
    ]:
        st.session_state.pop(key, None)


def store_captured_image(image: Image.Image, source: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CAPTURE_DIR / f"capture_{timestamp}.png"
    image.convert("RGB").save(path)
    st.session_state.captured_image = image.convert("RGB")
    st.session_state.captured_path = str(path)
    st.session_state.captured_source = source
    st.session_state.pop("analysis_result", None)


def gender_label_ko(label: str) -> str:
    return "여성" if label == "female" else "남성"


def save_result_images(original_image: Image.Image, face_image: Image.Image, future_results, pred_age: float, gender_result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"result_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    original_path = run_dir / "original.png"
    face_path = run_dir / "face_crop.png"
    original_image.save(original_path)
    face_image.save(face_path)

    saved_future = []
    for item in future_results:
        years = item["years_later"]
        future_age = pred_age + years
        out_path = run_dir / f"future_{years}years_age_{future_age:.1f}.png"
        item["image"].save(out_path)
        saved_future.append(
            {
                "years_later": years,
                "future_age": future_age,
                "target_age": item.get("target_age"),
                "engine": item.get("engine"),
                "path": out_path,
            }
        )

    summary_path = run_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"pred_age={pred_age:.2f}",
                f"gender={gender_result['label']}",
                f"gender_confidence={gender_result['confidence']:.4f}",
                f"engine={future_results[0].get('engine') if future_results else 'none'}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "run_dir": run_dir,
        "original": original_path,
        "face_crop": face_path,
        "future": saved_future,
        "summary": summary_path,
    }


def run_analysis(input_image: Image.Image):
    age_model, age_device, _ = get_age_model()
    gender_model, gender_device, _ = get_gender_model()

    face_result = crop_largest_face(input_image)
    face_image = face_result.image
    pred_age = predict_age_from_pil(face_image, age_model, age_device)
    gender_result = predict_gender_from_pil(face_image, gender_model, gender_device)
    future_results = generate_future_faces_sam(
        face_image,
        current_age=pred_age,
        years_list=YEARS_LIST,
    )
    saved = save_result_images(input_image, face_image, future_results, pred_age, gender_result)

    return {
        "face_image": face_image,
        "face_detector": face_result.detector,
        "face_box": face_result.box,
        "pred_age": pred_age,
        "gender_result": gender_result,
        "future_results": future_results,
        "saved": saved,
    }


def show_future_results(future_results, pred_age: float):
    tabs = st.tabs([f"{item['years_later']}년 뒤" for item in future_results])
    for tab, item in zip(tabs, future_results):
        years = item["years_later"]
        future_age = pred_age + years
        target_age = item.get("target_age", round(future_age))
        engine = item.get("engine", "unknown")
        engine_label = "SAM" if engine == "sam" else "예비 필터"
        with tab:
            st.image(
                item["image"],
                caption=f"{years}년 뒤 예상 나이: {future_age:.1f}세 / 변환 기준 나이: {target_age}세 / 생성 방식: {engine_label}",
                width="stretch",
            )

    st.write("전체 요약")
    cols = st.columns(5)
    for col, item in zip(cols, future_results):
        years = item["years_later"]
        future_age = pred_age + years
        with col:
            st.image(item["image"], caption=f"{years}년 뒤: {future_age:.1f}세", width="stretch")


def render_future_age_table(pred_age: float):
    future_age_rows = make_future_ages(pred_age, years_list=YEARS_LIST)
    st.write("미래 예상 나이")
    header_cols = st.columns([1, 1, 1])
    header_cols[0].markdown("**기준**")
    header_cols[1].markdown("**미래 예상 나이**")
    header_cols[2].markdown("**SAM 변환 기준 나이**")

    for item in future_age_rows:
        future_age = item["future_age"]
        target_age = int(max(0, min(round(future_age), 100)))
        row_cols = st.columns([1, 1, 1])
        row_cols[0].write(f"{item['years_later']}년 뒤")
        row_cols[1].write(f"{future_age:.1f}세")
        row_cols[2].write(f"{target_age}세")


def show_analysis_result(input_image: Image.Image, result: dict):
    st.divider()
    st.subheader("분석 결과")

    image_cols = st.columns(2)
    with image_cols[0]:
        st.image(input_image, caption="저장된 촬영 이미지", width="stretch")
    with image_cols[1]:
        st.image(
            result["face_image"],
            caption=f"분석용 얼굴 crop / detector: {result['face_detector']}",
            width="stretch",
        )

    metric_cols = st.columns(3)
    gender_result = result["gender_result"]
    pred_age = result["pred_age"]
    with metric_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">현재 예측 나이</div>
              <div class="metric-value">{pred_age:.1f}세</div>
              <div class="metric-sub">ConvNeXt-Tiny 기반 시각적 나이 추정</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">성별 분류 결과</div>
              <div class="metric-value">{gender_label_ko(gender_result['label'])}</div>
              <div class="metric-sub">신뢰도 {gender_result['confidence'] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_cols[2]:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-label">미래 얼굴 생성기</div>
              <div class="metric-value">SAM</div>
              <div class="metric-sub">API 없는 로컬 시뮬레이션</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_future_age_table(pred_age)

    st.subheader("미래 얼굴 시뮬레이션")
    show_future_results(result["future_results"], pred_age)

    if any(item.get("engine") == "rule_based_fallback" for item in result["future_results"]):
        st.warning("SAM 생성에 실패하여 예비 rule-based 필터 결과를 표시했습니다.")
    else:
        st.success("SAM 기반 미래 얼굴 시뮬레이션이 완료되었습니다.")

    with st.expander("저장된 결과 경로"):
        saved = result["saved"]
        st.write("결과 폴더:", str(saved["run_dir"]))
        st.write("원본 이미지:", str(saved["original"]))
        st.write("얼굴 crop:", str(saved["face_crop"]))
        for item in saved["future"]:
            st.write(
                f"{item['years_later']}년 뒤 / 예상 나이 {item['future_age']:.1f}세 / 생성 방식 {item['engine']}:",
                str(item["path"]),
            )


st.markdown(f'<h1 class="app-title">{APP_TITLE}</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-subtitle">
    현재 이 웹앱은 개발자 로컬 PC에서 실행되는 테스트 서버입니다.
    분석 버튼을 눌러 현재 나이·성별·미래 얼굴 시뮬레이션 결과를 확인합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("페이지 구성")
    st.page_link("app.py", label="홈")
    st.page_link("pages/01_시각화.py", label="시각화")
    st.page_link("pages/02_용어정리.py", label="용어정리")
    st.page_link("pages/03_다운로드공간.py", label="다운로드공간")
    st.page_link("pages/05_프로젝트흐름.py", label="프로젝트흐름")
    st.divider()
    st.caption("사용 모델")
    st.write("나이 예측: ConvNeXt-Tiny")
    st.write("성별 분류: MobileNetV3-Small")
    st.write("미래 얼굴: SAM")


st.markdown(
    """
    <div class="server-notice">
      <strong>로컬 서버 테스트 안내</strong><br>
      이 웹앱은 현재 Streamlit Cloud 단독 서버가 아니라 개발자 PC에서 실행되는 임시 테스트 서버입니다.
      사용자는 HTTPS 터널 주소로 접속하지만, 실제 분석 처리는 개발자 PC에서 실행됩니다.
      <div class="server-grid">
        <div>서버 PC가 꺼지거나 절전모드에 들어가면 접속이 중단됩니다.</div>
        <div>ngrok 터널 창 또는 Streamlit 실행 창이 꺼지면 서비스가 종료됩니다.</div>
        <div>촬영 또는 업로드한 얼굴 이미지는 분석을 위해 서버 PC로 전송됩니다.</div>
        <div>카메라 권한 요청이 뜨면 허용해야 촬영 기능을 사용할 수 있습니다.</div>
        <div>웹캠이 2대 이상이면 브라우저 주소창 왼쪽 자물쇠 아이콘의 사이트 설정에서 사용할 카메라를 선택한 뒤 새로고침해주세요.</div>
      </div>
    </div>

    <div class="step-band">
      <div class="step-row">
        <div class="step-item"><div class="step-num">1단계</div>카메라 권한 허용 후 얼굴 사진 촬영</div>
        <div class="step-item"><div class="step-num">2단계</div>촬영 이미지가 서버 PC로 전송되어 저장</div>
        <div class="step-item"><div class="step-num">3단계</div>로컬 모델로 현재 나이·성별 예측</div>
        <div class="step-item"><div class="step-num">4단계</div>미래 얼굴 시뮬레이션 결과 출력</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("촬영 전 유의사항", expanded=True):
    st.markdown(
        """
        <div class="tip-grid">
          <div>얼굴이 화면 중앙에 크게 보이도록 촬영해주세요.</div>
          <div>정면을 바라보고 촬영해주세요.</div>
          <div>모자, 마스크, 선글라스는 제거해주세요.</div>
          <div>앞머리나 손이 눈, 코, 입을 가리지 않게 해주세요.</div>
          <div>너무 어둡거나 역광인 환경은 피해주세요.</div>
          <div>안경, 수염, 진한 화장, 필터 앱 효과는 결과에 영향을 줄 수 있습니다.</div>
          <div>웹캠이 여러 대이거나 카메라가 안 켜지면 Chrome/Edge 사이트 설정에서 카메라 권한과 사용할 카메라를 확인해주세요.</div>
          <div>회사/학교 PC에서 웹캠이 차단되면 이미지 업로드 방식으로 테스트하는 것을 권장합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="notice">
    AI 추정 나이는 실제 나이가 아니라 이미지에서 보이는 시각적 나이입니다.
    미래 얼굴 결과는 실제 미래 모습 예측이 아니라 AI 기반 재미용 시뮬레이션입니다.
    촬영 또는 업로드한 이미지는 분석을 위해 서버 PC로 전송됩니다.
    개인정보가 포함된 민감한 이미지는 사용하지 않는 것을 권장합니다.
    </div>
    """,
    unsafe_allow_html=True,
)


if "captured_image" not in st.session_state:
    st.subheader("1. 얼굴 이미지 입력")
    st.write("카메라로 촬영하거나, 웹캠이 작동하지 않으면 얼굴 이미지를 업로드하세요.")

    input_tabs = st.tabs(["카메라 촬영", "이미지 업로드"])
    with input_tabs[0]:
        st.write("아래 카메라 화면에서 **이미지 촬영** 버튼을 눌러 얼굴 사진을 저장하세요.")
        camera_file = st.camera_input("얼굴 사진 촬영")

        if camera_file is not None:
            captured = Image.open(camera_file).convert("RGB")
            store_captured_image(captured, "카메라 촬영")
            st.rerun()

        st.info("촬영 후 이미지가 자동으로 저장되고 카메라 화면은 닫힙니다.")

    with input_tabs[1]:
        st.write("카메라가 작동하지 않거나 다른 사진으로 테스트하려면 얼굴 이미지를 업로드하세요.")
        uploaded_file = st.file_uploader(
            "얼굴 이미지 업로드",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            uploaded = Image.open(uploaded_file).convert("RGB")
            store_captured_image(uploaded, "이미지 업로드")
            st.rerun()

        st.info("업로드한 이미지는 서버 PC에 저장된 뒤 기존 분석 흐름과 동일하게 처리됩니다. DB는 사용하지 않습니다.")

    st.stop()


captured_image = st.session_state.captured_image
st.subheader("1. 저장된 얼굴 이미지")
st.image(
    captured_image,
    caption=f"입력 방식: {st.session_state.captured_source} / 저장 경로: {st.session_state.captured_path}",
    width="stretch",
)

button_cols = st.columns([1, 1, 3])
with button_cols[0]:
    analyze_clicked = st.button("분석 시작", type="primary", width="stretch")
with button_cols[1]:
    retry_clicked = st.button("다시하기", width="stretch")

if retry_clicked:
    reset_home_state()
    st.rerun()

if analyze_clicked:
    try:
        with st.spinner("현재 나이, 성별, 미래 얼굴을 분석하는 중입니다. SAM 생성에는 시간이 걸릴 수 있습니다..."):
            st.session_state.analysis_result = run_analysis(captured_image)
    except FaceNotDetectedError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("분석 중 오류가 발생했습니다.")
        st.exception(exc)

if "analysis_result" in st.session_state:
    show_analysis_result(captured_image, st.session_state.analysis_result)

    if st.button("처음으로 돌아가기", width="stretch"):
        reset_home_state()
        st.rerun()

    st.markdown(
        """
        <div class="danger-note">
        본 결과는 실제 미래 모습 예측이 아니라, 입력 얼굴을 나이 조건에 맞춰 변환한 AI 기반 재미용 시뮬레이션입니다.
        실제 외모 변화와 다를 수 있습니다. 또한 SAM은 공개 연구 모델 기반의 로컬 변환 결과이므로
        인종적 특징, 성별 특징, 얼굴 정체성이 완벽하게 유지되지 않을 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

