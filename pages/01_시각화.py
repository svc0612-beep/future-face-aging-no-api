from pathlib import Path
import csv

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "model_summary"


st.set_page_config(
    page_title="시각화",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    .visual-title {
        color: #172033;
        font-size: 2rem;
        font-weight: 850;
        margin: 0 0 .35rem;
    }
    .visual-subtitle {
        color: #637083;
        margin-bottom: 1rem;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin: 1rem 0 1.2rem;
    }
    .summary-card {
        border: 1px solid #d8dee8;
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
    }
    .summary-label {
        color: #637083;
        font-size: .9rem;
        margin-bottom: .25rem;
    }
    .summary-value {
        color: #185abc;
        font-size: 1.5rem;
        font-weight: 850;
    }
    .chart-card {
        border: 1px solid #d8dee8;
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0 1.2rem;
    }
    .chart-caption {
        color: #637083;
        font-size: .95rem;
        margin: .5rem 0 .8rem;
    }
    .read-box {
        border: 1px solid #cfe0ff;
        background: #f2f7ff;
        border-radius: 8px;
        padding: .9rem 1rem;
        margin-top: .8rem;
    }
    .read-box strong {
        color: #185abc;
    }
    .warning-box {
        border: 1px solid #f0ddb4;
        background: #fff8eb;
        border-radius: 8px;
        padding: .9rem 1rem;
        color: #5c4307;
        margin-top: .8rem;
    }
    @media (max-width: 800px) {
        .summary-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_csv_table(path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = list(csv.reader(file))
    if not reader:
        st.info("표시할 데이터가 없습니다.")
        return
    headers = reader[0]
    rows = reader[1:]
    header_cols = st.columns(len(headers))
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")
    for row in rows:
        row_cols = st.columns(len(headers))
        for col, cell in zip(row_cols, row):
            col.write(cell)


def render_weight_change_table():
    rows = [
        {
            "구분": "재학습 전",
            "가중치 파일": "best_convnext_tiny_full.pth",
            "역할": "기존 ConvNeXt-Tiny best 모델",
            "변화 의미": "fine-tuning을 시작하기 전 기준 모델입니다. Test MAE는 5.7122세였습니다.",
        },
        {
            "구분": "재학습 후 best",
            "가중치 파일": "best_convnext_tiny_finetune_lr1e5.pth",
            "역할": "최종 웹앱 나이 예측 모델",
            "변화 의미": "낮은 학습률로 추가 조정된 가중치입니다. Test MAE가 5.4106세로 개선되어 최종 선택했습니다.",
        },
        {
            "구분": "두 번째 추가 epoch last",
            "가중치 파일": "last_convnext_tiny_finetune_lr1e5_epoch2.pth",
            "역할": "마지막 추가 학습 상태",
            "변화 의미": "두 번째 추가 epoch 결과입니다. Validation MAE가 더 좋아지지 않아 best로 채택하지 않았습니다.",
        },
    ]
    headers = ["구분", "가중치 파일", "역할", "변화 의미"]
    header_cols = st.columns([1.0, 2.2, 1.8, 3.2])
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")
    for row in rows:
        row_cols = st.columns([1.0, 2.2, 1.8, 3.2])
        for col, header in zip(row_cols, headers):
            col.write(row[header])


def show_chart(title: str, image_name: str, caption: str, how_to_read: str, conclusion: str, caution: str | None = None):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(f'<div class="chart-caption">{caption}</div>', unsafe_allow_html=True)
    image_path = REPORT_DIR / image_name
    if image_path.exists():
        st.image(str(image_path), width="stretch")
    else:
        st.warning(f"그래프 파일을 찾을 수 없습니다: {image_path}")
    st.markdown(
        f"""
        <div class="read-box">
          <strong>읽는 방법</strong><br>
          {how_to_read}
          <br><br>
          <strong>현재 결론</strong><br>
          {conclusion}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if caution:
        st.markdown(f'<div class="warning-box">{caution}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<h1 class="visual-title">시각화</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="visual-subtitle">학습 로그를 기반으로 모델 성능을 비교하고, 각 그래프가 무엇을 의미하는지 설명합니다.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("페이지 구성")
    st.page_link("app.py", label="홈")
    st.page_link("pages/01_시각화.py", label="시각화")
    st.page_link("pages/02_용어정리.py", label="용어정리")
    st.page_link("pages/03_다운로드공간.py", label="다운로드공간")
    st.page_link("pages/05_프로젝트흐름.py", label="프로젝트흐름")

st.markdown(
    """
    <div class="summary-grid">
      <div class="summary-card">
        <div class="summary-label">나이 예측 최종 후보</div>
        <div class="summary-value">ConvNeXt-Tiny Fine-tuned</div>
        <div>Test MAE 5.4106세</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">성별 분류 최종 후보</div>
        <div class="summary-value">MobileNetV3-Small</div>
        <div>검증 정확도 88.19%</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">미래 얼굴 후보</div>
        <div class="summary-value">SAM</div>
        <div>API 없는 로컬 시뮬레이션</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

show_chart(
    title="나이 예측 모델 검증 평균 절대 오차 비교",
    image_name="age_val_mae_comparison.png",
    caption="각 모델이 검증 데이터에서 실제 나이와 얼마나 차이 났는지 epoch별로 보여줍니다.",
    how_to_read="세로축 값이 낮을수록 평균 나이 오차가 작습니다. 선이 아래로 내려갈수록 모델이 실제 나이에 더 가깝게 예측합니다.",
    conclusion="ConvNeXt-Tiny 원본을 낮은 학습률로 추가 튜닝한 모델이 Validation MAE 5.6993세로 가장 낮았습니다. 최종 앱에는 이 fine-tuned 모델을 우선 사용합니다.",
    caution="평균 오차가 5.4106세라는 뜻이지 모든 사진이 5세 이내로 맞는다는 뜻은 아닙니다. 조명, 각도, 얼굴 crop 상태에 따라 더 크게 틀릴 수 있습니다.",
)

show_chart(
    title="나이 예측 모델 검증 손실 비교",
    image_name="age_val_loss_comparison.png",
    caption="모델이 검증 데이터에서 학습 목표 함수 기준으로 얼마나 틀렸는지 보여줍니다.",
    how_to_read="손실은 낮을수록 좋습니다. 다만 손실은 수학적 학습 기준이라 사용자가 체감하는 나이 오차와 완전히 같지는 않습니다.",
    conclusion="손실 그래프는 모델이 안정적으로 학습되는지 확인하는 보조 지표입니다. 최종 모델 선택은 사용자가 이해하기 쉬운 평균 절대 오차를 더 중요하게 봤습니다.",
)

show_chart(
    title="나이 예측 모델 최고 성능 비교",
    image_name="age_model_best_mae_bar.png",
    caption="각 나이 예측 모델의 가장 좋은 검증 평균 절대 오차만 모아 비교한 그래프입니다.",
    how_to_read="막대가 낮을수록 좋은 모델입니다. 숫자는 해당 모델이 검증 데이터에서 평균적으로 몇 세 정도 차이 났는지를 뜻합니다.",
    conclusion="현재 순위는 ConvNeXt-Tiny Fine-tuned, ConvNeXt-Tiny Fine-tuned 2nd, ConvNeXt-Tiny 원본, EfficientNet-B0, MobileNetV3 순서입니다. 두 번째 추가 epoch는 더 좋아지지 않아 첫 번째 fine-tuned best 모델을 유지합니다.",
)

show_chart(
    title="ConvNeXt-Tiny Test MAE 개선 비교",
    image_name="age_test_mae_improvement.png",
    caption="원본 ConvNeXt-Tiny와 fine-tuned 모델을 Test set에서 비교한 그래프입니다.",
    how_to_read="막대가 낮을수록 실제 테스트 이미지에서 평균 나이 오차가 작습니다. Validation이 아니라 최종 Test 기준이므로 실제 성능 판단에 더 중요합니다.",
    conclusion="Test MAE가 5.7122세에서 5.4106세로 약 0.3016세 줄었습니다. 5세 이내 비율도 58.58%에서 59.61%로 상승했습니다.",
    caution="개선은 확인됐지만 고령층, 어두운 웹캠 사진, 얼굴이 작게 찍힌 사진에서는 여전히 오차가 커질 수 있습니다.",
)

show_chart(
    title="성별 분류 학습 지표",
    image_name="gender_training_metrics.png",
    caption="성별 분류 모델의 정확도와 손실 변화를 함께 보여줍니다.",
    how_to_read="정확도는 높을수록 좋고 손실은 낮을수록 좋습니다. 검증 정확도가 가장 높은 지점이 현재 모델 선택 기준입니다.",
    conclusion="MobileNetV3-Small은 5 epoch에서 검증 정확도 88.19%를 기록했습니다. 가볍고 빠르기 때문에 웹앱 성별 분류 모델로 사용합니다.",
    caution="성별 분류 결과도 조명, 얼굴 각도, 머리카락, 화장, 안경 등에 따라 틀릴 수 있습니다.",
)

st.divider()
st.subheader("요약 표")

age_summary_path = REPORT_DIR / "age_model_best_summary.csv"
gender_summary_path = REPORT_DIR / "gender_model_best_summary.csv"
test_summary_path = REPORT_DIR / "age_test_improvement_summary.csv"

col1, col2 = st.columns(2)
with col1:
    st.write("나이 예측 모델 최고 평균 절대 오차")
    if age_summary_path.exists():
        render_csv_table(age_summary_path)
    else:
        st.warning("나이 모델 요약 파일이 없습니다.")

with col2:
    st.write("성별 분류 모델 최고 정확도")
    if gender_summary_path.exists():
        render_csv_table(gender_summary_path)
    else:
        st.warning("성별 모델 요약 파일이 없습니다.")

st.write("ConvNeXt-Tiny Test 성능 개선")
if test_summary_path.exists():
    render_csv_table(test_summary_path)
else:
    st.warning("Test 개선 요약 파일이 없습니다.")

st.divider()
st.subheader("가중치/모델 파일 변화 요약")
st.markdown(
    """
    모델의 **가중치**는 학습으로 조정된 내부 숫자들이고, `.pth` 파일은 그 가중치를 저장한 파일입니다.
    이번 재학습에서는 원본 ConvNeXt-Tiny 가중치를 낮은 학습률로 조금 더 다듬어 최종 모델을 선택했습니다.
    """
)
render_weight_change_table()
st.info(
    "이 표는 사용자가 이해하기 위한 요약입니다. 수백만 개 파라미터의 실제 숫자 변화량을 레이어별로 분석한 표는 아니며, "
    "어떤 가중치 파일을 기준으로 최종 모델을 선택했는지 설명합니다."
)

