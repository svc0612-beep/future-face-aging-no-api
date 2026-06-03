# future-face-aging-no-api

API 없이 로컬 모델로 동작하는 안면 에이징 웹앱 프로젝트입니다.

사용자가 카메라로 얼굴 사진을 촬영하면 현재 나이와 성별을 예측하고, 현재 예측 나이를 기준으로 10년, 20년, 30년, 40년, 50년 뒤의 미래 얼굴 시뮬레이션 이미지를 보여줍니다.

> 본 결과는 실제 미래 모습 예측이 아니라 AI 기반 재미용 시뮬레이션입니다.

## 주요 기능

- Streamlit 기반 웹앱
- 카메라 촬영 및 이미지 업로드
- 얼굴 crop 전처리
- ConvNeXt-Tiny fine-tuned 모델 기반 나이 예측
- MobileNetV3-Small 기반 성별 분류
- SAM 기반 미래 얼굴 시뮬레이션 시도
- SAM 실행 실패 시 rule-based aging filter fallback
- 시각화 페이지
- 용어정리 페이지
- 다운로드공간 페이지

## 최종 모델 성능

### 나이 예측

최종 후보: `ConvNeXt-Tiny Fine-tuned`

- Test MAE: `5.4106`
- Test RMSE: `7.4872`
- 5세 이내 비율: `59.61%`
- 10세 이내 비율: `84.30%`

재학습 전 ConvNeXt-Tiny Test MAE는 `5.7122`였고, fine-tuning 후 약 `0.3016세` 개선되었습니다.

### 성별 분류

최종 후보: `MobileNetV3-Small`

- Best Validation Accuracy: `88.19%`
- Best Epoch: `5`

## 프로젝트 구조

```text
app.py
config.py
progress_monitor.py
pages/
  01_시각화.py
  02_용어정리.py
  03_다운로드공간.py
  04_현재작업.py
src/
  predict_age.py
  predict_gender.py
  face_preprocessing.py
  sam_aging_simulator.py
  aging_simulator.py
  generate_model_summary_visuals.py
outputs/reports/model_summary/
```

## 로컬 실행

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

## 모델 파일 안내

GitHub 저장소에는 대용량 모델 가중치와 원본 데이터셋을 포함하지 않습니다.

아래 파일은 로컬에서 직접 준비해야 합니다.

```text
models/age_estimator/best_convnext_tiny_finetune_lr1e5.pth
models/gender_classifier/best_gender_mobilenet_v3_small.pth
third_party/SAM/pretrained_models/sam_ffhq_aging.pt
third_party/SAM/shape_predictor_68_face_landmarks.dat
```

SAM 관련 파일이 없으면 웹앱은 rule-based aging fallback을 사용합니다.

## 제외한 항목

아래 항목은 저장소에 올리지 않습니다.

- `.venv`, `.venv-1`
- `data/`
- `models/`
- `third_party/`
- `outputs/captures`
- `outputs/aged_results`
- `outputs/sam_jobs`
- 대용량 로그 파일
- 사용자 촬영 이미지

## 주의사항

- 나이 예측 결과는 실제 나이가 아니라 이미지에서 보이는 시각적 나이 추정입니다.
- 조명, 얼굴 각도, 표정, 안경, 화장, 카메라 화질에 따라 오차가 커질 수 있습니다.
- 미래 얼굴 이미지는 실제 미래 모습이 아니라 AI 기반 시뮬레이션입니다.
- 공개 SAM 가중치는 한국인 정체성, 성별 특징, 얼굴 정체성을 완벽히 유지하지 못할 수 있습니다.

