from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = PROJECT_ROOT / "outputs" / "logs" / "age_improvement_status.json"
CONSOLE_LOG = PROJECT_ROOT / "outputs" / "logs" / "age_improvement_pipeline_console.log"


st.set_page_config(
    page_title="현재작업",
    page_icon="⏱",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    .page-title {
        color: #172033;
        font-size: 2rem;
        font-weight: 850;
        margin: 0 0 .35rem;
    }
    .page-subtitle {
        color: #637083;
        margin-bottom: 1rem;
    }
    .main-status {
        border: 1px solid #cfe0ff;
        background: #f2f7ff;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .main-label {
        color: #185abc;
        font-weight: 850;
        font-size: .95rem;
        margin-bottom: .25rem;
    }
    .main-value {
        color: #172033;
        font-size: 1.65rem;
        line-height: 1.35;
        font-weight: 850;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .8rem;
        margin: 1rem 0;
    }
    .metric-box {
        border: 1px solid #d8dee8;
        background: #ffffff;
        border-radius: 8px;
        padding: .95rem;
        min-height: 104px;
    }
    .metric-label {
        color: #637083;
        font-size: .88rem;
        margin-bottom: .25rem;
    }
    .metric-value {
        color: #172033;
        font-size: 1.45rem;
        font-weight: 850;
        line-height: 1.25;
    }
    .metric-note {
        color: #637083;
        font-size: .82rem;
        margin-top: .35rem;
    }
    .timeline {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .65rem;
        margin: 1rem 0;
    }
    .step {
        border: 1px solid #d8dee8;
        background: #fbfcff;
        border-radius: 8px;
        padding: .85rem;
    }
    .step.done {
        background: #f0fbf5;
        border-color: #b8e4cc;
    }
    .step.active {
        background: #f2f7ff;
        border-color: #cfe0ff;
    }
    .step-title {
        font-weight: 850;
        color: #172033;
        margin-bottom: .25rem;
    }
    .step-state {
        color: #185abc;
        font-size: .9rem;
        font-weight: 800;
    }
    .warning-box {
        border: 1px solid #f0ddb4;
        background: #fff8eb;
        color: #5c4307;
        border-radius: 8px;
        padding: .9rem 1rem;
        margin: 1rem 0;
    }
    @media (max-width: 900px) {
        .metric-grid,
        .timeline {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_status() -> dict | None:
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "error", "error": "상태 파일을 읽지 못했습니다."}


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "-"
    try:
        return str(timedelta(seconds=int(float(seconds))))
    except (TypeError, ValueError):
        return "-"


def parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    return None


def finish_time(status: dict) -> str:
    eta = status.get("eta_seconds") or status.get("phase_eta_seconds")
    updated_at = parse_time(status.get("updated_at"))
    if eta is None or updated_at is None:
        return "-"
    try:
        return (updated_at + timedelta(seconds=float(eta))).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return "-"


def current_task_name(status: dict) -> str:
    step = status.get("pipeline_step")
    task = status.get("task")
    model = status.get("current_model", "-")
    phase = status.get("phase", "-")

    if step == "baseline_test_eval" or (task == "test_evaluation" and model == "convnext_tiny_full"):
        return "기존 ConvNeXt-Tiny 모델 Test 평가 중"
    if step == "low_lr_finetune" or task == "convnext_finetune":
        if phase == "train":
            return "ConvNeXt-Tiny 낮은 학습률 추가 학습 중"
        if phase == "valid":
            return "ConvNeXt-Tiny 추가 학습 검증 중"
        return "ConvNeXt-Tiny fine-tuning 진행 중"
    if step == "finetuned_test_eval":
        return "fine-tuning 모델 Test 평가 중"
    if step == "refresh_visuals":
        return "새 시각화 그래프 생성 중"
    if status.get("status") == "completed":
        return "모든 개선 작업 완료"
    if status.get("status") == "error":
        return "작업 오류 발생"
    return "현재 작업 확인 중"


def step_state(status: dict, key: str) -> str:
    order = ["baseline_test_eval", "low_lr_finetune", "finetuned_test_eval", "refresh_visuals", "completed"]
    current = status.get("pipeline_step")
    if status.get("status") == "completed":
        return "done"
    if current == key:
        return "active"
    try:
        if order.index(current) > order.index(key):
            return "done"
    except (ValueError, TypeError):
        pass
    return "pending"


with st.sidebar:
    st.header("페이지 구성")
    st.page_link("app.py", label="홈")
    st.page_link("pages/01_시각화.py", label="시각화")
    st.page_link("pages/02_용어정리.py", label="용어정리")
    st.page_link("pages/03_다운로드공간.py", label="다운로드공간")
    st.page_link("pages/04_현재작업.py", label="현재작업")

st.markdown('<h1 class="page-title">현재작업</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">백그라운드에서 돌아가는 나이 예측 개선 작업을 눈으로 확인하는 전용 페이지입니다. 10초마다 자동 새로고침됩니다.</div>',
    unsafe_allow_html=True,
)
st.markdown("<meta http-equiv='refresh' content='10'>", unsafe_allow_html=True)

status = read_status()
if status is None:
    st.info("아직 작업 상태 파일이 없습니다. 백그라운드 작업이 시작되면 여기에 표시됩니다.")
    st.stop()

progress = float(status.get("progress_percent", 0) or 0)
progress = max(0.0, min(progress, 100.0))

st.markdown(
    f"""
    <div class="main-status">
      <div class="main-label">지금 백그라운드에서 하는 일</div>
      <div class="main-value">{current_task_name(status)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.progress(progress / 100.0)

eta = status.get("eta_seconds") or status.get("phase_eta_seconds")

st.markdown(
    f"""
    <div class="metric-grid">
      <div class="metric-box">
        <div class="metric-label">현재 작업 모델</div>
        <div class="metric-value">{status.get("current_model", "-")}</div>
        <div class="metric-note">현재 실행 중인 모델/작업 이름</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">현재 진행률</div>
        <div class="metric-value">{progress:.1f}%</div>
        <div class="metric-note">{status.get("step", "-")} / {status.get("total_steps", "-")} step</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">예상 남은 시간</div>
        <div class="metric-value">{fmt_duration(eta)}</div>
        <div class="metric-note">현재 단계 기준 예상</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">종료 예정 시간</div>
        <div class="metric-value">{finish_time(status)}</div>
        <div class="metric-note">마지막 갱신 시각 + 남은 시간</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-grid">
      <div class="metric-box">
        <div class="metric-label">현재 MAE</div>
        <div class="metric-value">{status.get("current_mae", status.get("val_mae", "-"))}</div>
        <div class="metric-note">현재까지 계산된 평균 나이 오차</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">현재 Loss</div>
        <div class="metric-value">{status.get("current_loss", "-")}</div>
        <div class="metric-note">학습/검증 중일 때 표시</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Epoch</div>
        <div class="metric-value">{status.get("epoch", "-")} / {status.get("total_epochs", "-")}</div>
        <div class="metric-note">fine-tuning 단계에서 표시</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">마지막 갱신</div>
        <div class="metric-value">{status.get("updated_at", "-")}</div>
        <div class="metric-note">상태 파일 갱신 시각</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("전체 작업 순서")
steps = [
    ("baseline_test_eval", "1. 기존 모델 Test 평가"),
    ("low_lr_finetune", "2. 낮은 학습률 추가 학습"),
    ("finetuned_test_eval", "3. 개선 모델 Test 평가"),
    ("refresh_visuals", "4. 새 시각화 생성"),
    ("completed", "5. 결과/용어 정리"),
]

step_html = ['<div class="timeline">']
for key, title in steps:
    state = "done" if status.get("status") == "completed" else step_state(status, key)
    state_text = {"done": "완료", "active": "진행 중", "pending": "대기"}[state]
    step_html.append(
        f"""
        <div class="step {state}">
          <div class="step-title">{title}</div>
          <div class="step-state">{state_text}</div>
        </div>
        """
    )
step_html.append("</div>")
st.markdown("".join(step_html), unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="warning-box">
    완료 목표: {status.get("deadline_note", "내일 오전 8시 전 완료")}<br>
    현재 페이지는 알림을 보내지 않고, 사용자가 직접 확인하는 작업 모니터입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("콘솔 로그 최근 내용"):
    if CONSOLE_LOG.exists():
        text = CONSOLE_LOG.read_text(encoding="utf-8", errors="replace")
        st.code(text[-8000:])
    else:
        st.info("아직 콘솔 로그가 없습니다.")

with st.expander("상태 원본 JSON"):
    st.json(status)
