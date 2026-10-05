# Model Card — 이음 수어 인식 모델

## 1. 목적

카페 주문이라는 제한된 상황에서 사전에 정의된 한국수어 동작을 분류해 AAC 메시지 생성에 활용하기 위한 시계열 분류 모델입니다.

범용 한국수어 번역, 문법 분석, 통역 대체, 의료·법률 의사결정 용도로 설계되지 않았습니다.

## 2. 현재 배포 기준 모델 (v1)

- Feature schema: `ieum_v1_90`
- 입력 shape: `[1, 100, 90]`
- 첫 번째 감지 손 21 landmark 상대좌표: 63
- 손바닥 법선: 3
- 얼굴 핵심 8점 상대좌표: 24
- 총 90차원
- LSTM(128) → LSTM(64) → Dense(32) → Softmax(16)
- 총 164,144 parameters

라벨:

`안녕하세요, 아메리카노, 뜨거운, 아이스, 2잔, 주세요, 제일, 큰걸로, 해주세요, 테이크아웃, 포인트, 카드, 영수증, 감사합니다, 와이파이, 있나요`

## 3. 기존 metadata 성능

저장된 `model/metadata.json`:

- training samples: 2,944
- validation samples: 184
- train accuracy: 0.9959
- validation accuracy: 0.9891

### 해석 제한

이 저장소에는 당시 원본 데이터와 split 코드가 남아 있지 않습니다. 따라서:

- 동일 수행자의 표본이 train/validation에 함께 들어갔는지
- 증강된 표본이 양쪽 split에 섞였는지
- 새 사용자에 대한 일반화가 어느 정도인지

재현 검증할 수 없습니다.

따라서 **98.91%는 내부 validation 수치일 뿐 실제 한국수어 사용자에 대한 정확도라고 표현하지 않습니다.**

## 4. v1의 구조적 한계

1. Hand Landmarker는 최대 2손을 검출하지만 v1 feature는 첫 번째 손만 모델 입력에 사용합니다.
2. 얼굴 정보는 일부 landmark 좌표일 뿐 비수지신호를 충분히 모델링하지 않습니다.
3. 손 위치를 손목 기준으로만 상대화해 손과 얼굴 사이의 위치 관계가 크게 사라집니다.
4. Softmax 분류기는 학습하지 않은 동작도 기존 클래스 중 하나에 확률을 부여할 수 있습니다.
5. 데이터가 소수 수행자 중심이었다면 signer generalization이 제한될 수 있습니다.

## 5. 개선 feature schema (v2)

코드·수집기·평가 파이프라인 구현 및 자동 parity/split 검증 완료, **실제 다인원 재학습은 아직 수행하지 않음**.

`ieum_v2_190`:

- 왼손 local normalized landmarks + palm normal + 얼굴 기준 손목 위치: 69
- 오른손 동일: 69
- 두 손 손목 상대 위치: 3
- 얼굴 핵심점 normalized geometry: 24
- 입/눈/눈썹/head-roll 파생 특징: 8
- 선택 Face Blendshape: 14
- left/right/face presence masks: 3
- 합계: 190

v2는 양손, 손-얼굴 공간 관계, 일부 비수지 특징을 보존하도록 설계했습니다.

## 6. 모델 후보

v2 학습 파이프라인은 동일 데이터 split에서:

- 2-layer LSTM
- 2-layer GRU

를 학습하고 validation macro-F1을 우선 기준으로 선택합니다.

최종 결과는 선택된 후보를 별도 test split에서 1회 평가하도록 설계했습니다. 현재 파이프라인은 추가로:

- held-out test accuracy / macro-F1 / weighted-F1
- bootstrap 95% CI
- confusion matrix와 classification report
- test sample별 true/pred/confidence/margin
- participant별 test 성능
- `기타` 클래스가 있을 때 unknown false-accept rate
- rejection 적용 후 known-class coverage / selective accuracy

를 저장합니다.

## 7. Split 정책

우선순위:

1. participant-group split
2. session-group split
3. stratified random split fallback

3번을 사용하면 같은 수행자의 특성이 여러 split에 들어갈 수 있으므로 일반화 성능으로 과장하지 않습니다.

## 8. 추론 엔진

현재 v1은 세 가지 후보를 보존합니다.

- TensorFlow.js FP32
- ONNX Runtime Web FP32
- ONNX Runtime Web Dynamic INT8

앱의 `auto` 모드는 ONNX FP32를 우선 사용하고 로드 실패 시 TF.js로 fallback합니다.

## 9. 양자화 검증

합성 입력 300개, GitHub Actions CPU:

- ONNX FP32: 695,959 B
- ONNX INT8: 239,430 B
- 크기 감소: 65.60%
- FP32 평균 추론: 2.310 ms
- INT8 평균 추론: 2.525 ms
- 합성 입력 argmax agreement: 99.0%
- mean absolute output error: 0.000992

이는 실제 수어 정확도 검증이 아닙니다. INT8 채택은 실제 validation/test 데이터의 macro-F1과 실제 브라우저 지연시간을 함께 보고 결정해야 합니다.

## 10. 사람 중심 안전장치

- max probability뿐 아니라 top-1/top-2 margin 사용
- 낮은 확신도에서 자동 메시지 생성 억제
- 상위 후보 3개를 사용자에게 제시해 repair 허용
- 직접 텍스트 입력 제공
- 빠른 표현 제공
- 원 인식 단어와 자연화 문장을 분리 표시
- 생성형 AI가 임의로 의미를 추가하지 않도록 현재 문장 자연화는 deterministic rule 기반

## 11. 공정성/접근성 과제

향후 실제 한국수어 사용자 데이터를 다양한:

- 손 크기/성별/연령
- 주사용손
- 피부톤
- 카메라 기기
- 조명
- 배경
- 수행 속도
- 수어 변이

조건에서 수집해야 합니다.

## 12. v2 ONNX/양자화 의사결정 규칙

v2 학습 후 `tools/run_v2_pipeline.py`는 동일 held-out test IDs에서 ONNX FP32와 Dynamic INT8을 비교합니다.

INT8은 최소한:
- test accuracy 감소 ≤ 1%p
- macro-F1 감소 ≤ 1%p
- FP32/INT8 argmax 일치율 ≥ 99%

조건을 확인하도록 구현했습니다. 이 gate를 통과해도 브라우저 benchmark 결과가 느리면 기본 런타임으로 채택하지 않습니다.

## 13. 특징 구현 동일성

브라우저 JavaScript와 Python 학습 코드가 같은 `ieum_v2_190`을 생성하는지 공유 fixture로 자동 비교합니다. CI에서 190차원 전체에 대한 numerical parity를 확인합니다.

## 14. 권장 보고 표현

권장:

> 제한된 카페 주문 어휘를 대상으로 개발한 AI 기반 한국수어 AAC 프로토타입

피해야 할 표현:

> 한국수어를 99% 정확도로 번역하는 시스템
