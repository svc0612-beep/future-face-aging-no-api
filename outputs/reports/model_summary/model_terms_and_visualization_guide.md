# API 없는 안면 에이징 프로젝트 모델/용어/시각화 정리

작성 기준: 2026-06-03  
프로젝트명: API 없는 안면 에이징 - 재미로만 봐주세요

이 문서는 웹앱 제작 전에 현재 프로젝트에서 사용했거나 검토한 모델을 정리하고, 학습 그래프와 평가 지표가 무엇을 의미하는지 설명하기 위한 문서입니다.

## 1. 최종 웹앱에서 사용할 현재 기준 베스트 모델

| 기능 | 최종 후보 | 한국어 발음 | 현재 성능 | 선택 이유 |
|---|---|---:|---:|---|
| 나이 예측 | ConvNeXt-Tiny | 컨브넥스트 타이니 | Best Val MAE 5.9904세 | 현재 학습한 나이 예측 모델 중 검증 오차가 가장 낮음 |
| 성별 분류 | MobileNetV3-Small | 모바일넷 브이쓰리 스몰 | Best Val Accuracy 88.19% | 가볍고 빠르며 성별 분류용 MVP에 적합 |
| 미래 얼굴 시뮬레이션 | SAM | 샘 | 로컬 샘플 생성 성공 | API 없이 target age를 넣어 얼굴 나이 변환 가능 |

최종 조합은 `ConvNeXt-Tiny 나이 예측 + MobileNetV3-Small 성별 분류 + SAM 미래 얼굴 시뮬레이션`입니다.

단, SAM은 외국인 중심 얼굴 데이터로 학습된 공개 가중치를 사용하므로 한국인 얼굴 정체성 유지가 완벽하지 않을 수 있습니다. 따라서 웹앱에는 반드시 “실제 미래 예측이 아니라 AI 기반 재미용 시뮬레이션”이라는 안내 문구가 들어가야 합니다.

## 2. 나이 예측 모델 비교

### 2.1 ResNet18

- 풀네임: Residual Network 18
- 한국어 발음: 레즈넷 에이틴 또는 레스넷 십팔
- 역할: 나이 예측 1차 본학습 모델
- 특징:
  - 비교적 가볍고 안정적인 CNN 구조
  - 이미지 분류/회귀에서 오래 사용된 기본 모델
  - 배포 부담이 크지 않음
- 프로젝트 결과:
  - Validation Best MAE: 6.4439세
  - Test MAE: 6.1019세
  - Test RMSE: 8.4282세
  - 오차 5세 이내 비율: 54.99%
  - 오차 10세 이내 비율: 80.23%
- 판단:
  - 파이프라인 검증과 baseline으로 의미 있음
  - 최종 모델로는 ConvNeXt-Tiny보다 성능이 낮아 보류

### 2.2 MobileNetV3

- 풀네임: MobileNet Version 3
- 한국어 발음: 모바일넷 브이쓰리
- 역할: 나이 예측 비교 학습 모델
- 특징:
  - 모바일/배포 환경을 고려한 경량 CNN
  - 빠르고 가벼움
  - 작은 서버나 Streamlit 배포에 유리
- 프로젝트 결과:
  - Best Val MAE: 6.2679세
  - Best Epoch: 12
- 판단:
  - 속도와 배포성은 좋음
  - 정확도는 ConvNeXt-Tiny보다 낮음
  - 최종 나이 예측 모델로는 보류

### 2.3 ConvNeXt-Tiny

- 풀네임: ConvNeXt Tiny
- 한국어 발음: 컨브넥스트 타이니
- 역할: 현재 나이 예측 최종 후보
- 특징:
  - CNN 구조를 현대적으로 개선한 모델
  - ResNet/MobileNet보다 무겁지만 이미지 특징 추출력이 좋음
  - 얼굴의 주름, 윤곽, 피부 질감 같은 시각적 특징 학습에 유리할 가능성이 있음
- 프로젝트 결과:
  - Best Val MAE: 5.9904세
  - Best Epoch: 7
- 판단:
  - 현재 학습한 나이 예측 모델 중 검증 MAE가 가장 낮음
  - 최종 웹앱의 나이 예측 기본 모델로 사용
  - 단, 최종 Test 평가를 추가로 실행해 성능을 확정하는 것이 좋음

### 2.4 EfficientNet-B0

- 풀네임: EfficientNet-B0
- 한국어 발음: 이피션트넷 비제로
- 역할: 나이 예측 비교 학습 모델
- 특징:
  - 모델 크기, 입력 해상도, 채널 수를 균형 있게 확장하는 구조
  - 정확도와 효율의 균형이 좋은 모델
  - B0는 EfficientNet 계열 중 가장 가벼운 기본 모델
- 프로젝트 결과:
  - Best Val MAE: 6.1760세
  - Best Epoch: 9
- 판단:
  - MobileNetV3보다 약간 좋지만 ConvNeXt-Tiny보다 낮음
  - 최종 후보로는 보류

## 3. 성별 분류 모델

### MobileNetV3-Small

- 풀네임: MobileNet Version 3 Small
- 한국어 발음: 모바일넷 브이쓰리 스몰
- 역할: 성별 분류 모델
- 특징:
  - 가볍고 빠름
  - 성별 분류처럼 상대적으로 단순한 이진 분류 작업에 적합
  - 웹앱 배포 시 부담이 적음
- 프로젝트 결과:
  - Best Val Accuracy: 88.19%
  - Best Epoch: 5
  - Best Val Loss: 0.2732
- 판단:
  - MVP 성별 분류 모델로 사용 가능
  - 조명, 각도, 머리카락, 화장, 안경 등에 따라 오분류 가능성은 안내해야 함

## 4. 미래 얼굴 생성/에이징 후보 모델

### 4.1 SAM

- 풀네임: Style-based Age Manipulation
- 한국어 발음: 샘
- 관련 논문명: Only a Matter of Style: Age Transformation Using a Style-Based Regression Model
- 역할: 현재 미래 얼굴 시뮬레이션 최종 후보
- 특징:
  - 입력 얼굴 이미지를 StyleGAN 기반 latent space로 변환
  - target age를 입력하면 해당 나이대 얼굴로 변환
  - `sam_ffhq_aging.pt` 가중치를 사용
  - 10년, 20년, 30년, 40년, 50년 뒤 예상 나이를 target age로 넣을 수 있음
- 프로젝트 테스트 결과:
  - 로컬 실행 성공
  - 샘플 1장 기준 target age 1개당 약 20~25초
  - 5장 생성 시 약 2분 내외
- 장점:
  - API 없이 로컬에서 실행 가능
  - LATS보다 결과 붕괴가 적음
  - 주름, 눈가, 입가 등 노화 변화가 실제로 보임
- 한계:
  - 외국인 중심 데이터로 학습된 공개 가중치
  - 한국인 얼굴 정체성 유지가 완벽하지 않을 수 있음
  - 얼굴 정렬 실패 시 결과가 크게 흔들릴 수 있음
  - 속도가 느림
- 현재 판단:
  - MVP에서는 SAM을 사용하되 “재미용 AI 시뮬레이션”으로 명확히 안내

### 4.2 LATS

- 풀네임: Lifespan Age Transformation Synthesis
- 한국어 발음: 래츠 또는 엘에이티에스
- 역할: 미래 얼굴 생성 후보였으나 실패/보류
- 특징:
  - 한 장의 얼굴 사진에서 다양한 나이대 얼굴을 생성하는 face aging 모델
  - 얼굴의 texture와 shape 변화를 함께 다루는 방향
- 프로젝트 테스트 결과:
  - 로컬 테스트에서 얼굴 정체성 유지가 부족함
  - 여성 얼굴에 원치 않는 수염/성별 변화 느낌이 나타남
  - 한국인 얼굴이 점차 서양인처럼 변하는 문제가 관찰됨
  - 피부가 자연스럽게 늙기보다 흐물거리거나 붕괴되는 느낌이 발생
- 현재 판단:
  - 최종 미래 얼굴 생성기로 사용하지 않음
  - 보고서에는 “실패 후보 및 제외 사유”로 기록

### 4.3 CUSP

- 풀네임: Custom Structure Preservation in Face Aging
- 한국어 발음: 커스프
- 역할: 미래 얼굴 생성 후보였으나 보류
- 특징:
  - 얼굴 구조 보존을 강조하는 face aging 연구 모델
  - Docker/연구 코드 기반
  - 공식 가중치 접근성이 좋지 않음
- 프로젝트 확인 결과:
  - GitHub 코드는 확보 가능
  - 필요한 가중치 다운로드 링크가 접근되지 않음
  - Docker 기반 의존성이 복잡함
  - 현재 PC 환경에서 즉시 MVP 통합이 어려움
- 현재 판단:
  - 최종 모델로 사용하지 않음
  - 추후 가중치 확보 및 실행 환경이 준비되면 다시 검토 가능

## 5. 주요 지표 설명

### 5.1 Loss

- 한국어 발음: 로스
- 의미:
  - 모델이 학습 중 얼마나 틀렸는지를 수학적으로 계산한 값
  - 모델은 이 Loss를 낮추는 방향으로 학습함
- Loss가 낮으면:
  - 학습 목표 함수 기준으로 예측이 좋아지고 있다는 뜻
  - 하지만 사용자 체감 정확도와 100% 같은 의미는 아님
- Loss가 높으면:
  - 모델이 아직 데이터 패턴을 잘 못 잡고 있거나
  - 학습률, 데이터, 모델 구조, 전처리 문제가 있을 수 있음
- 주의:
  - Loss는 모델/작업/손실 함수 종류에 따라 단위가 달라짐
  - 서로 다른 작업의 Loss를 직접 비교하면 안 됨
  - 예: 나이 예측 Loss와 성별 분류 Loss는 의미가 다름

### 5.2 MAE

- 풀네임: Mean Absolute Error
- 한국어 발음: 엠에이이
- 한국어 의미: 평균 절대 오차
- 프로젝트에서의 의미:
  - 실제 나이와 예측 나이의 차이를 절댓값으로 계산한 뒤 평균낸 값
  - 예: 실제 35세, 예측 41세라면 오차는 6세
- MAE가 낮으면:
  - 평균적으로 실제 나이에 더 가깝게 예측함
  - 나이 예측 모델 성능을 해석하기 가장 쉬운 지표
- MAE가 높으면:
  - 평균 오차가 크다는 뜻
  - 사용자 입장에서 “나이를 너무 틀린다”고 느낄 가능성이 큼
- 현재 프로젝트 기준:
  - ConvNeXt-Tiny Best Val MAE 5.9904세
  - 즉 검증 데이터 기준 평균적으로 약 6세 정도 차이가 난다는 의미

### 5.3 RMSE

- 풀네임: Root Mean Squared Error
- 한국어 발음: 알엠에스이
- 한국어 의미: 평균 제곱근 오차
- 의미:
  - 큰 오차에 더 민감한 지표
  - 한두 장에서 크게 틀리면 RMSE가 MAE보다 더 크게 올라감
- 해석:
  - MAE는 평균적인 체감 오차
  - RMSE는 큰 실패 사례까지 포함한 위험도 확인용

### 5.4 Accuracy

- 한국어 발음: 애큐러시
- 한국어 의미: 정확도
- 프로젝트에서의 의미:
  - 성별 분류에서 남/여를 맞힌 비율
  - 예: Accuracy 88.19%면 검증 이미지 중 약 88.19%를 맞혔다는 뜻
- Accuracy가 높으면:
  - 분류를 더 많이 맞힘
- Accuracy가 낮으면:
  - 오분류가 많음
- 주의:
  - 데이터가 한쪽 성별에 치우쳐 있으면 Accuracy만으로는 부족할 수 있음
  - 필요하면 Precision, Recall, Confusion Matrix도 추가해야 함

## 6. Loss와 MAE의 관계

나이 예측에서는 Loss와 MAE가 둘 다 낮아지는 것이 이상적입니다.

- Train Loss가 계속 낮아짐:
  - 모델이 학습 데이터를 점점 잘 맞추고 있음
- Val Loss와 Val MAE도 같이 낮아짐:
  - 새로운 데이터에도 잘 맞고 있음
  - 좋은 학습 흐름
- Train Loss는 낮아지는데 Val MAE가 올라감:
  - 과적합 가능성
  - 학습 데이터는 외웠지만 새로운 얼굴에는 약해지는 상황
- Val MAE가 가장 낮은 epoch:
  - 실제 앱에 사용할 best checkpoint를 고르는 기준

현재 프로젝트에서는 나이 예측 모델 선택 기준으로 `Best Validation MAE`를 가장 중요하게 봅니다. 사용자는 Loss보다 “몇 살 정도 틀리는지”를 더 직접적으로 느끼기 때문입니다.

## 7. 그래프 설명

### 7.1 Age Estimation Validation MAE

이미지 파일: `age_val_mae_comparison.png`

이 그래프는 epoch가 진행될수록 각 나이 예측 모델의 검증 MAE가 어떻게 변했는지 보여줍니다.

- x축: Epoch
- y축: Validation MAE
- y값이 낮을수록 좋음
- 가장 낮은 지점이 해당 모델의 best 성능

해석:
ConvNeXt-Tiny가 가장 낮은 Val MAE 5.9904세를 기록했으므로 현재 나이 예측 최종 후보입니다.

### 7.2 Age Estimation Validation Loss

이미지 파일: `age_val_loss_comparison.png`

이 그래프는 모델별 검증 Loss 변화를 보여줍니다.

- x축: Epoch
- y축: Validation Loss
- 낮을수록 학습 목표 기준으로 예측이 안정됨

해석:
Loss가 낮아지는 흐름은 모델이 나이 예측 문제를 더 잘 학습하고 있다는 뜻입니다. 다만 최종 사용자 체감은 MAE가 더 직접적이므로 모델 선택은 MAE 중심으로 합니다.

### 7.3 Best Validation MAE by Age Model

이미지 파일: `age_model_best_mae_bar.png`

이 그래프는 각 나이 예측 모델의 가장 좋은 Validation MAE만 비교합니다.

현재 순위:

| 순위 | 모델 | Best Val MAE | Best Epoch |
|---:|---|---:|---:|
| 1 | ConvNeXt-Tiny Fine-tuned | 5.6993세 | 추가 epoch 1 |
| 2 | ConvNeXt-Tiny Fine-tuned 2nd | 5.7009세 | 추가 epoch 1 |
| 3 | ConvNeXt-Tiny | 5.9904세 | 7 |
| 4 | EfficientNet-B0 | 6.1760세 | 9 |
| 5 | MobileNetV3 | 6.2679세 | 12 |

해석:
ConvNeXt-Tiny 원본이 가장 좋은 기본 모델이었고, 낮은 학습률 fine-tuning을 적용한 뒤 Validation MAE가 5.6993세까지 낮아졌습니다. 두 번째 추가 epoch는 5.7009세로 더 좋아지지 않아 첫 번째 fine-tuned best 모델을 유지합니다.

### 7.4 ConvNeXt-Tiny Test MAE Improvement

이미지 파일: `age_test_mae_improvement.png`

이 그래프는 ConvNeXt-Tiny 원본과 fine-tuned 모델을 최종 Test 데이터에서 비교합니다.

| 모델 | Test MAE | Test RMSE | 5세 이내 | 10세 이내 |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny before fine-tuning | 5.7122세 | 7.9482세 | 58.58% | 82.54% |
| ConvNeXt-Tiny after fine-tuning | 5.4106세 | 7.4872세 | 59.61% | 84.30% |

해석:
Fine-tuning 후 Test MAE가 5.7122세에서 5.4106세로 약 0.3016세 낮아졌습니다. 5세 이내 비율과 10세 이내 비율도 함께 상승했으므로, 현재 웹앱의 나이 예측 모델은 fine-tuned ConvNeXt-Tiny를 사용하는 것이 가장 합리적입니다.

### 7.5 Gender Classification Metrics

이미지 파일: `gender_training_metrics.png`

이 그래프는 성별 분류 모델의 Accuracy와 Loss 변화를 함께 보여줍니다.

- Accuracy는 높을수록 좋음
- Loss는 낮을수록 좋음
- Validation Accuracy가 가장 높은 epoch가 best epoch

현재 결과:

| 모델 | Best Val Accuracy | Best Epoch | Best Val Loss |
|---|---:|---:|---:|
| MobileNetV3-Small | 88.19% | 5 | 0.2732 |

## 8. 재학습 이후 추가 용어

### 8.1 Fine-tuning

Fine-tuning은 이미 학습된 모델을 처음부터 다시 만드는 것이 아니라, 저장된 best 모델 가중치를 가져와 낮은 학습률로 조금 더 조정하는 방식입니다.

이번 프로젝트에서는 `best_convnext_tiny_full.pth`를 기준으로 추가 학습을 진행했고, 최종적으로 `best_convnext_tiny_finetune_lr1e5.pth`가 나이 예측 최종 후보가 되었습니다.

### 8.2 Low Learning Rate

낮은 학습률은 모델을 조금씩만 수정하겠다는 뜻입니다. 이미 성능이 괜찮은 모델을 크게 흔들면 기존에 잘 맞히던 것까지 망가질 수 있으므로, 이번 추가 학습에서는 `1e-5`를 사용했습니다.

### 8.3 Test MAE Improvement

Test MAE Improvement는 재학습 전후 모델을 같은 Test 데이터로 비교했을 때 평균 오차가 얼마나 줄었는지 보는 지표입니다.

이번 결과:

- 재학습 전 Test MAE: 5.7122세
- 재학습 후 Test MAE: 5.4106세
- 개선폭: 약 0.3016세
- 5세 이내 비율: 58.58% → 59.61%
- 10세 이내 비율: 82.54% → 84.30%

### 8.4 Stale Epoch

Stale epoch는 학습을 더 했지만 Validation 성능이 좋아지지 않은 epoch를 뜻합니다. 두 번째 추가 epoch는 Validation MAE가 5.6993세에서 5.7009세로 약간 나빠졌기 때문에 best 모델로 채택하지 않았습니다.

### 8.5 Pipeline

Pipeline, 즉 파이프라인은 순서대로 자동 실행되는 작업 묶음입니다.

이번 프로젝트에서 파이프라인은 다음 흐름을 뜻했습니다.

```text
기존 모델 평가 → 추가 학습 → 다시 평가 → 그래프 생성 → 용어/보고서 정리
```

사람이 명령을 하나씩 계속 입력하지 않아도 정해진 순서대로 다음 작업이 이어지는 구조입니다. 다만 사용자가 직접 진행 상황을 볼 수 있도록 별도 진행률 페이지를 두는 것이 중요합니다.

### 8.6 Training / Tuning / Fine-tuning

학습, 튜닝, 파인튜닝은 비슷해 보이지만 의미가 다릅니다.

| 용어 | 뜻 | 프로젝트에서의 의미 |
| --- | --- | --- |
| 파이프라인 | 순서대로 자동 실행되는 작업 묶음 | 기존 모델 평가 → 추가 학습 → 다시 평가 → 그래프 생성 → 용어/보고서 정리를 순서대로 이어서 처리하는 흐름입니다. |
| 학습 | 모델을 처음부터 또는 크게 훈련시키는 것 | ResNet18, MobileNetV3, ConvNeXt-Tiny, EfficientNet-B0 같은 모델이 얼굴 이미지와 나이/성별 라벨을 보고 규칙을 배우는 과정입니다. |
| 튜닝 | 이미 학습된 모델을 조금 더 다듬는 작업 | 이미 만들어진 모델의 결과가 아쉬울 때 학습률, epoch, 데이터 흐름 등을 조정해 성능을 더 끌어올리는 작업입니다. |
| 파인튜닝 | 튜닝 중에서도 기존 가중치를 가져와서 새 데이터/목표에 맞게 이어 학습하는 방식 | ConvNeXt-Tiny 기존 best 가중치를 가져와 낮은 학습률로 이어 학습했고, 최종적으로 fine-tuned 모델을 나이 예측 기준 모델로 선택했습니다. |

이번 나이 예측 개선은 ConvNeXt-Tiny 기존 best 가중치를 가져와 낮은 학습률로 이어 학습한 파인튜닝입니다.

## 9. 웹앱에 넣을 안내 문구

### 9.1 촬영 전 안내

정확한 나이 추정을 위해 아래 조건에 맞춰 촬영해주세요.

- 얼굴이 화면 중앙에 크게 보이도록 촬영해주세요.
- 정면을 바라보고 촬영해주세요.
- 모자, 마스크, 선글라스는 제거해주세요.
- 앞머리나 손이 눈, 코, 입을 가리지 않게 해주세요.
- 너무 어둡거나 역광인 환경은 피해주세요.
- 얼굴에 강한 그림자가 생기지 않도록 해주세요.
- 카메라가 너무 멀거나 얼굴이 작게 나오면 예측 오차가 커질 수 있습니다.
- 여러 명이 함께 나오면 정확도가 떨어질 수 있습니다.
- 안경, 수염, 진한 화장, 필터 앱 효과는 예측 결과에 영향을 줄 수 있습니다.
- AI가 추정하는 나이는 실제 나이가 아니라 이미지에서 보이는 시각적 나이입니다.

### 9.2 결과 안내

AI 추정 나이는 촬영 환경, 조명, 얼굴 각도, 표정, 안경, 화장, 화질에 따라 실제 나이와 차이가 날 수 있습니다.

### 9.3 미래 얼굴 안내

본 결과는 실제 미래 모습 예측이 아니라, 입력 얼굴을 나이 조건에 맞춰 변환한 AI 기반 재미용 시뮬레이션입니다. 실제 외모 변화와 다를 수 있습니다.

### 9.4 SAM 한계 안내

미래 얼굴 이미지는 공개 연구 모델 기반의 로컬 변환 결과입니다. 인종적 특징, 성별 특징, 얼굴 정체성이 완벽하게 유지되지 않을 수 있으며, 결과가 부자연스러운 경우 재촬영을 권장합니다.

## 10. 출처 및 참고

- SAM: Style-based Age Manipulation, 공식 구현: https://github.com/yuval-alaluf/SAM
- SAM 논문: Only a Matter of Style: Age Transformation Using a Style-Based Regression Model
- LATS: Lifespan Age Transformation Synthesis
- CUSP: Custom Structure Preservation in Face Aging, 공식 구현: https://github.com/guillermogotre/CUSP
