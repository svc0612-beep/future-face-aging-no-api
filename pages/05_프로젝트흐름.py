import streamlit as st


st.set_page_config(
    page_title="프로젝트흐름",
    page_icon="🧭",
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
    .flow-title {
        color: #172033;
        font-size: 2rem;
        font-weight: 850;
        margin: 0 0 .35rem;
    }
    .flow-subtitle {
        color: #637083;
        margin-bottom: 1rem;
    }
    .overview-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .7rem;
        margin: 1rem 0 1.3rem;
    }
    .overview-card {
        border: 1px solid #d8dee8;
        background: #ffffff;
        border-radius: 8px;
        padding: .9rem;
        min-height: 112px;
    }
    .overview-index {
        color: #185abc;
        font-weight: 850;
        font-size: .92rem;
        margin-bottom: .35rem;
    }
    .overview-title {
        color: #172033;
        font-weight: 850;
        font-size: 1.08rem;
        margin-bottom: .35rem;
    }
    .overview-desc {
        color: #637083;
        font-size: .9rem;
        line-height: 1.45;
    }
    .stage-card {
        border: 1px solid #d8dee8;
        background: #ffffff;
        border-radius: 8px;
        padding: 1.1rem;
        margin: 1rem 0 1.15rem;
    }
    .stage-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid #edf1f7;
        padding-bottom: .8rem;
        margin-bottom: .9rem;
    }
    .stage-title {
        color: #172033;
        font-size: 1.35rem;
        font-weight: 850;
    }
    .stage-result {
        color: #185abc;
        font-weight: 800;
        text-align: right;
    }
    .step-grid {
        display: grid;
        grid-template-columns: 70px minmax(0, 1.4fr) minmax(0, 2.2fr);
        gap: .65rem;
        align-items: start;
        padding: .55rem 0;
        border-bottom: 1px solid #f0f3f8;
    }
    .step-grid:last-child {
        border-bottom: 0;
    }
    .step-no {
        color: #185abc;
        font-weight: 850;
    }
    .step-work {
        color: #172033;
        font-weight: 760;
    }
    .step-why {
        color: #4e5b6d;
        line-height: 1.48;
    }
    .why-box {
        border: 1px solid #cfe0ff;
        background: #f2f7ff;
        border-radius: 8px;
        padding: .9rem 1rem;
        color: #243b63;
        margin-top: .8rem;
    }
    .note-box {
        border: 1px solid #f0ddb4;
        background: #fff8eb;
        border-radius: 8px;
        padding: .9rem 1rem;
        color: #5c4307;
        margin-top: .8rem;
    }
    @media (max-width: 900px) {
        .overview-grid {
            grid-template-columns: 1fr;
        }
        .step-grid {
            grid-template-columns: 1fr;
        }
        .stage-head {
            display: block;
        }
        .stage-result {
            text-align: left;
            margin-top: .4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def sidebar_links():
    with st.sidebar:
        st.header("페이지 구성")
        st.page_link("app.py", label="홈")
        st.page_link("pages/01_시각화.py", label="시각화")
        st.page_link("pages/02_용어정리.py", label="용어정리")
        st.page_link("pages/03_다운로드공간.py", label="다운로드공간")
        st.page_link("pages/05_프로젝트흐름.py", label="프로젝트흐름")


def render_stage(title: str, result: str, rows: list[tuple[str, str]], why: str):
    st.markdown('<div class="stage-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="stage-head">
          <div class="stage-title">{title}</div>
          <div class="stage-result">{result}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-grid">
          <div class="step-no">순서</div>
          <div class="step-work">작업</div>
          <div class="step-why">왜 했는지</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for index, (work, reason) in enumerate(rows, start=1):
        st.markdown(
            f"""
            <div class="step-grid">
              <div class="step-no">{index}</div>
              <div class="step-work">{work}</div>
              <div class="step-why">{reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="why-box"><strong>이 단계의 목적</strong><br>{why}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


sidebar_links()

st.markdown('<h1 class="flow-title">프로젝트흐름</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="flow-subtitle">데이터셋 수집부터 전처리, 학습, 웹앱 제작, 배포 준비까지 어떤 순서로 만들었고 왜 그렇게 했는지 정리했습니다.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="overview-grid">
      <div class="overview-card">
        <div class="overview-index">1단계</div>
        <div class="overview-title">데이터셋 수집</div>
        <div class="overview-desc">AIHub와 UTKFace 얼굴 데이터를 준비했습니다.</div>
      </div>
      <div class="overview-card">
        <div class="overview-index">2단계</div>
        <div class="overview-title">전처리</div>
        <div class="overview-desc">라벨 매핑, CSV 생성, split, 누수 검사를 진행했습니다.</div>
      </div>
      <div class="overview-card">
        <div class="overview-index">3단계</div>
        <div class="overview-title">학습</div>
        <div class="overview-desc">나이 예측과 성별 분류 모델을 비교 학습했습니다.</div>
      </div>
      <div class="overview-card">
        <div class="overview-index">4단계</div>
        <div class="overview-title">웹앱 제작</div>
        <div class="overview-desc">Streamlit으로 촬영, 분석, 결과 출력 흐름을 연결했습니다.</div>
      </div>
      <div class="overview-card">
        <div class="overview-index">5단계</div>
        <div class="overview-title">배포 준비</div>
        <div class="overview-desc">GitHub 저장소와 문서, 제외 파일을 정리했습니다.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_stage(
    "데이터셋",
    "AIHub 45,200개 + UTKFace 24,102개",
    [
        ("AIHub 안면 인식 에이징 데이터 준비", "한국인 얼굴 기반 데이터를 포함해야 실제 카메라 입력에 가까운 나이 예측 모델을 만들 수 있기 때문입니다."),
        ("UTKFace 데이터 준비", "공개 얼굴 데이터로 나이 범위를 넓히고 학습량을 보강하기 위해 사용했습니다."),
        ("최종 통합 대상 선정", "두 데이터셋의 장점을 섞어 나이대와 얼굴 다양성을 확보하려고 했습니다."),
    ],
    "한 가지 데이터셋만 쓰면 특정 촬영 조건이나 특정 얼굴 분포에 치우칠 수 있어, 한국인 중심 데이터와 공개 얼굴 데이터를 함께 사용했습니다.",
)

render_stage(
    "전처리",
    "최종 metadata 69,302장 구성",
    [
        ("AIHub 이미지와 JSON 라벨 매핑", "이미지만으로는 나이 라벨을 알 수 없기 때문에 JSON 라벨과 정확히 연결해야 했습니다."),
        ("파일명 대소문자 문제 해결", "이미지 파일명 끝의 d/f와 JSON 파일명 끝의 D/F 차이 때문에 매칭 실패가 생겨 stem을 lower 처리했습니다."),
        ("UTKFace 파일명에서 age/gender/race 파싱", "UTKFace는 파일명 자체에 라벨이 들어 있어 자동으로 나이, 성별, 인종 정보를 추출했습니다."),
        ("AIHub + UTKFace 통합 metadata 생성", "모델 학습에서 하나의 CSV 기준으로 이미지 경로와 라벨을 읽을 수 있게 만들었습니다."),
        ("train/val/test split 생성", "학습, 검증, 최종 평가 데이터를 분리해 모델 성능을 더 신뢰할 수 있게 했습니다."),
        ("group leakage 검사", "같은 사람이 train과 test에 동시에 들어가 성능이 부풀려지는 문제를 막기 위해 확인했습니다."),
        ("DataLoader 이미지 로딩 검증", "학습 전에 이미지 경로, PIL 로딩, Tensor 변환, batch shape가 정상인지 확인했습니다."),
    ],
    "전처리는 모델이 잘 배우게 만드는 기초 작업입니다. 라벨이 틀리거나 같은 사람이 여러 split에 섞이면 좋은 모델처럼 보여도 실제 웹앱에서는 틀릴 수 있습니다.",
)

render_stage(
    "학습",
    "나이: ConvNeXt-Tiny Fine-tuned / 성별: MobileNetV3-Small",
    [
        ("ResNet18 파이프라인 검증", "처음부터 큰 학습을 돌리기 전에 데이터 로딩과 학습 코드가 정상인지 확인했습니다."),
        ("ResNet18 본학습", "기준 성능을 만들고 이후 모델들과 비교하기 위한 baseline으로 사용했습니다."),
        ("MobileNetV3 학습", "가볍고 배포가 쉬운 모델이 실제로 충분한 정확도를 내는지 확인했습니다."),
        ("ConvNeXt-Tiny 학습", "나이 예측 정확도를 더 높이기 위해 표현력이 좋은 모델을 비교했습니다."),
        ("EfficientNet-B0 학습", "정확도와 효율의 균형이 좋은 다른 후보와 비교하기 위해 학습했습니다."),
        ("ConvNeXt-Tiny fine-tuning", "가장 성능이 좋았던 ConvNeXt-Tiny를 낮은 학습률로 더 다듬어 Test MAE를 줄였습니다."),
        ("Test set 최종 평가", "모델 선택에 쓰지 않은 최종 데이터로 실제 성능을 확인했습니다."),
        ("성별 분류 MobileNetV3-Small 학습", "미래 얼굴 처리와 결과 안내에 성별 정보를 함께 사용하기 위해 별도 분류 모델을 만들었습니다."),
    ],
    "여러 모델을 학습한 이유는 속도, 용량, 정확도 사이의 균형을 비교하기 위해서입니다. 최종 나이 모델은 Test MAE 5.4106세의 ConvNeXt-Tiny Fine-tuned를 선택했습니다.",
)

render_stage(
    "미래 얼굴 시뮬레이션",
    "SAM 사용, 실패 시 rule-based fallback",
    [
        ("rule-based aging filter 제작", "생성 모델 없이도 MVP 결과를 만들 수 있는 최소 대안을 먼저 만들었습니다."),
        ("LATS 후보 검토", "얼굴 나이 변환 연구 모델을 테스트했지만 한국인 정체성 및 성별 유지 문제가 있어 제외했습니다."),
        ("CUSP 후보 검토", "얼굴 구조 보존을 기대했지만 공식 가중치 접근과 Docker 기반 구조 때문에 MVP 통합을 보류했습니다."),
        ("SAM 후보 연결", "입력 얼굴을 target age 조건으로 바꾸는 공개 모델이라 로컬 테스트 후보로 사용했습니다."),
        ("실패 시 fallback 구성", "SAM 파일이 없거나 실행 실패해도 웹앱이 완전히 멈추지 않도록 rule-based 결과를 대신 출력하게 했습니다."),
    ],
    "API 없이 미래 얼굴을 생성하려면 품질과 실행 난도가 모두 문제가 됩니다. 그래서 여러 후보를 비교하고, 최종 앱에는 SAM과 fallback을 함께 둔 구조로 만들었습니다.",
)

render_stage(
    "웹앱 제작",
    "Streamlit 기반 로컬 서버 웹앱",
    [
        ("Streamlit 기반 홈 화면 구성", "Python 모델과 결과 화면을 빠르게 연결하고 로컬 PC에서 바로 확인하기 위해 사용했습니다."),
        ("로컬 서버 테스트 안내 추가", "현재 서비스가 개발자 PC와 ngrok HTTPS 터널을 통해 제공되는 임시 테스트 서버임을 사용자에게 명확히 알리기 위해 추가했습니다."),
        ("카메라 촬영 기능", "사용자가 웹앱에서 직접 얼굴 사진을 촬영하는 최종 사용 흐름을 구현했습니다."),
        ("카메라 권한 안내", "외부 사용자는 HTTPS 터널 주소로 접속하고 브라우저 카메라 권한을 허용해야 촬영할 수 있으므로 안내 문구를 넣었습니다."),
        ("이미지 업로드 기능", "카메라가 안 되거나 기존 사진으로 테스트할 수 있도록 대체 입력을 준비했습니다."),
        ("얼굴 crop 전처리", "사진 전체가 아니라 얼굴 영역 중심으로 나이와 성별을 예측하기 위해 넣었습니다."),
        ("나이 예측 연결", "촬영 이미지에서 현재 예측 나이를 먼저 계산해야 미래 예상 나이를 만들 수 있습니다."),
        ("성별 분류 연결", "성별 정보가 결과 해석과 미래 얼굴 시뮬레이션 판단에 도움이 되도록 연결했습니다."),
        ("SAM 미래 얼굴 생성 연결", "10년, 20년, 30년, 40년, 50년 뒤 target age별 결과를 만들기 위해 연결했습니다. 단, SAM은 GPU/CUDA 의존성이 있어 배포 환경에서는 fallback 가능성을 함께 둡니다."),
        ("결과 이미지 저장", "분석 결과를 나중에 확인하고 다운로드할 수 있도록 outputs 폴더에 저장했습니다."),
        ("시각화 페이지 제작", "모델 성능 그래프와 평가 결과를 사용자가 직접 확인할 수 있게 했습니다."),
        ("용어정리 페이지 제작", "Loss, MAE, fine-tuning, pipeline 같은 용어를 프로젝트 맥락에 맞게 설명했습니다."),
        ("다운로드공간 제작", "사용자가 업로드한 파일이나 결과 자료를 다시 받을 수 있는 공간을 만들었습니다."),
    ],
    "웹앱 제작의 핵심은 카메라 촬영 → 서버 PC 전송 → 현재 나이/성별 분석 → 미래 얼굴 시뮬레이션 → 결과 확인으로 이어지는 사용 흐름을 사용자가 이해할 수 있게 만드는 것이었습니다.",
)

render_stage(
    "배포 준비",
    "GitHub + Git LFS + 로컬 ngrok 테스트",
    [
        ("GitHub 저장소 생성", "프로젝트 코드와 문서를 버전 관리하고 나중에 배포 준비를 하기 위해 저장소를 만들었습니다."),
        (".gitignore 정리", "원본 데이터, 사용자 촬영 이미지, 대용량 로그가 저장소에 올라가지 않게 막았습니다."),
        ("Git LFS로 모델 가중치 업로드", "나이 예측 모델과 성별 분류 모델은 일반 GitHub 파일로 올리기 어려워 Git LFS로 업로드했습니다."),
        ("Streamlit Cloud 배포 시도", "클라우드 배포를 시도했지만 SAM 실행과 모델 파일/실행 환경 문제 때문에 최종 테스트는 로컬 서버 방식으로 전환했습니다."),
        ("로컬 PC 서버 방식 전환", "개발자 PC에서 Streamlit을 실행하고 ngrok HTTPS 주소로 외부 사용자가 접속하는 방식으로 실제 분석 기능을 확인할 수 있게 했습니다."),
        ("ngrok HTTPS 터널 사용", "외부 사용자의 브라우저에서 카메라 권한 요청이 뜨도록 HTTPS 주소를 제공하기 위해 사용했습니다."),
        ("README 작성", "프로젝트 목적, 기능, 모델 성능, 실행 방법, 필요한 모델 파일을 정리했습니다."),
        ("requirements.txt 작성", "로컬 또는 배포 환경에서 필요한 Python 패키지를 설치할 수 있게 했습니다."),
        ("홈 화면 안내 수정", "사용자가 로컬 서버 테스트 구조, 이미지 전송, 서버 종료 조건, 카메라 권한 조건을 알 수 있도록 배포 안내를 수정했습니다."),
    ],
    "현재 배포 방식은 완전한 상시 운영 서비스가 아니라, 개발자 PC가 켜져 있을 때 ngrok HTTPS 주소로 접속하는 테스트 배포입니다. 그래서 사용자가 서버 구조와 개인정보 전송 여부를 이해하도록 안내하는 것이 중요합니다.",
)

st.markdown(
    """
    <div class="note-box">
      <strong>정리:</strong> 이 프로젝트는 단순히 웹 화면만 만든 것이 아니라,
      데이터셋 정리 → 전처리 검증 → 여러 모델 비교 학습 → 최종 모델 선택 → 미래 얼굴 후보 검토 → Streamlit 웹앱 제작 → GitHub/Git LFS 정리 → 로컬 PC + ngrok 테스트 배포까지 이어진 작업입니다.
    </div>
    """,
    unsafe_allow_html=True,
)
