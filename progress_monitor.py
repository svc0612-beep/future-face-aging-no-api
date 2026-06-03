from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
STATUS_PATH = BASE_DIR / "outputs" / "logs" / "age_improvement_status.json"
CONSOLE_LOG_PATH = BASE_DIR / "outputs" / "logs" / "age_improvement_pipeline_console.log"


st.set_page_config(
    page_title="현재 학습 진행률",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <meta http-equiv="refresh" content="8">
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1280px;
    }
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .subtle {
        color: #64748b;
        font-size: 0.98rem;
    }
    .status-band {
        padding: 1rem 1.15rem;
        border-radius: 10px;
        border: 1px solid #dbeafe;
        background: #eff6ff;
        color: #1e3a8a;
        margin: 1.2rem 0 1rem 0;
        font-weight: 700;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        min-height: 118px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        color: #111827;
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
        word-break: keep-all;
    }
    .metric-small {
        color: #475569;
        font-size: 0.9rem;
        margin-top: 0.45rem;
    }
    .timeline-row {
        display: grid;
        grid-template-columns: 150px 1fr;
        gap: 0.8rem;
        align-items: start;
        padding: 0.75rem 0;
        border-bottom: 1px solid #eef2f7;
    }
    .timeline-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-weight: 800;
        font-size: 0.85rem;
        text-align: center;
    }
    .done { background: #dcfce7; color: #166534; }
    .running { background: #fef3c7; color: #92400e; }
    .pending { background: #f1f5f9; color: #475569; }
    .error { background: #fee2e2; color: #991b1b; }
    .log-box {
        background: #0f172a;
        color: #dbeafe;
        border-radius: 10px;
        padding: 1rem;
        font-family: Consolas, monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_status() -> dict:
    if not STATUS_PATH.exists():
        return {
            "status": "waiting",
            "current_model": "대기 중",
            "phase": "상태 파일 없음",
            "progress_percent": 0,
        }
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "error",
            "current_model": "상태 파일 읽기 실패",
            "phase": str(exc),
            "progress_percent": 0,
        }


def fmt_seconds(value: object) -> str:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return "-"
    if seconds <= 0:
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분"
    if minutes:
        return f"{minutes}분 {seconds}초"
    return f"{seconds}초"


def fmt_finish_time(seconds: object) -> str:
    try:
        seconds_int = int(float(seconds))
    except (TypeError, ValueError):
        return "-"
    if seconds_int <= 0:
        return "-"
    return (datetime.now() + timedelta(seconds=seconds_int)).strftime("%m-%d %H:%M")


def short_phase(phase: str) -> str:
    mapping = {
        "train": "학습 중",
        "valid": "검증 중",
        "test": "테스트 평가 중",
        "completed": "완료",
        "error": "오류",
    }
    return mapping.get(str(phase), str(phase))


def metric_card(label: str, value: str, small: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-small">{small}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_log_tail(max_lines: int = 70) -> str:
    if not CONSOLE_LOG_PATH.exists():
        return "아직 콘솔 로그가 없습니다."
    try:
        lines = CONSOLE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        return f"로그 읽기 실패: {exc}"
    return "\n".join(lines[-max_lines:]) if lines else "로그가 비어 있습니다."


status = load_status()
progress = float(status.get("progress_percent") or 0)
progress_ratio = max(0.0, min(progress / 100, 1.0))
phase_eta = status.get("phase_eta_seconds") or status.get("eta_seconds")

st.markdown('<div class="main-title">현재 학습 진행률</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">이 화면은 app.py와 분리된 학습 상태 전용 모니터입니다. 8초마다 자동 새로고침됩니다.</div>',
    unsafe_allow_html=True,
)

status_text = {
    "running": "백그라운드 작업 실행 중",
    "completed": "백그라운드 작업 완료",
    "error": "백그라운드 작업 오류",
    "waiting": "상태 대기 중",
}.get(str(status.get("status")), str(status.get("status")))

st.markdown(
    f'<div class="status-band">{status_text} · {status.get("deadline_note", "내일 오전 8시 전 완료 목표")}</div>',
    unsafe_allow_html=True,
)

cols = st.columns(4)
with cols[0]:
    metric_card("현재 작업 모델", str(status.get("current_model", "-")), short_phase(status.get("phase", "-")))
with cols[1]:
    metric_card("현재 진행률", f"{progress:.2f}%", f"{status.get('step', '-')} / {status.get('total_steps', '-')}")
with cols[2]:
    metric_card("현재 단계 남은 시간", fmt_seconds(phase_eta), f"완료 예상 {fmt_finish_time(phase_eta)}")
with cols[3]:
    metric_card("Epoch", f"{status.get('epoch', '-')} / {status.get('total_epochs', '-')}", f"Early stopping {status.get('patience', '-')}")

st.progress(progress_ratio)

cols = st.columns(4)
with cols[0]:
    metric_card("현재 MAE", str(status.get("current_mae", "-")), "낮을수록 평균 나이 오차가 작습니다.")
with cols[1]:
    metric_card("현재 Loss", str(status.get("current_loss", "-")), "학습 목표 함수 기준 오차입니다.")
with cols[2]:
    metric_card("기존 Test MAE", str(status.get("test_mae", "-")), "재학습 전 ConvNeXt-Tiny 테스트 결과")
with cols[3]:
    metric_card("최근 갱신", str(status.get("updated_at", "-")), f"시작 {status.get('started_at', '-')}")

st.divider()

st.subheader("작업 순서")
steps = [
    ("기존 모델 Test 평가", "baseline_test_eval"),
    ("낮은 학습률 추가 학습", "low_lr_finetune"),
    ("개선 모델 Test 평가", "finetuned_test_eval"),
    ("새 시각화 생성", "refresh_visuals"),
    ("결과 및 용어 정리", "completed"),
]
current_step = str(status.get("pipeline_step", ""))
overall_status = str(status.get("status", ""))

for label, key in steps:
    if overall_status == "error":
        badge_class, badge_text = "error", "확인 필요"
    elif overall_status == "completed" or key == "completed" and overall_status == "completed":
        badge_class, badge_text = "done", "완료"
    elif key == current_step:
        badge_class, badge_text = "running", "진행 중"
    elif steps.index((label, key)) < [item[1] for item in steps].index(current_step) if current_step in [item[1] for item in steps] else False:
        badge_class, badge_text = "done", "완료"
    else:
        badge_class, badge_text = "pending", "대기"
    st.markdown(
        f"""
        <div class="timeline-row">
            <div><span class="timeline-badge {badge_class}">{badge_text}</span></div>
            <div><b>{label}</b><br><span class="subtle">{key}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("실시간 로그")
st.markdown(f'<div class="log-box">{read_log_tail()}</div>', unsafe_allow_html=True)

with st.expander("상태 파일 원본 보기"):
    st.json(status)
