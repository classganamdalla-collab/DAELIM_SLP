# 이음(理音) — 수어를 잇다

웹 브라우저에서 한국수어 동작을 인식해 문자·음성으로 전달하고, 상대방 음성을 문자로 보여주는 **상황 특화형 AAC 프로토타입**입니다. 현재 적용 상황은 카페 주문이며, 범용 한국수어 번역기를 표방하지 않습니다.

## 현재 상태

이 저장소의 `main`은 2026년 6월 버전을 보존합니다. 공모전 개선 작업은 **`contest-2026-v2-final` 브랜치**에서 통합 검증합니다. 이전 `contest-2026-onnx` 브랜치는 ONNX 전환 실험 이력으로 보존합니다.

현재 개선 브랜치에서 완료된 항목:

- 기존 Keras LSTM 모델을 ONNX FP32로 변환
- ONNX Dynamic INT8 양자화 및 자동 검증
- ONNX Runtime Web FP32 우선 + TensorFlow.js fallback
- TF.js / ONNX FP32 / ONNX INT8 브라우저 벤치마크 페이지
- 양손 + 얼굴 비수지 특징을 위한 `ieum_v2_190` feature schema
- 새 데이터 수집기: 양손 landmark, 확장 얼굴 landmark, Face Blendshape, 가명 참여자/세션 코드
- LSTM vs GRU 학습·평가 파이프라인\n- JS/Python v2 특징 추출 parity 자동검사\n- 참여자 단위 split 자동검사 및 test prediction 저장\n- test accuracy/macro-F1 bootstrap 95% CI\n- `기타` 표본이 있을 때 unknown false-accept를 고려한 rejection threshold 보정
- 참여자 단위 split 우선 평가
- 인식 실패 시 상위 후보 선택을 통한 communication repair
- 직접 텍스트 입력/빠른 표현 등 fallback AAC
- 투명한 규칙 기반 카페 문장 자연화
- 파일럿 과제 로거와 결과 대시보드
- GitHub Actions 자동 품질 검사

**아직 완료되지 않은 핵심:** `ieum_v2_190`용 실제 다인원 데이터 수집 및 재학습. 원시 수집 데이터가 GitHub에 남아 있지 않기 때문에 이 단계는 실제 표본 수집 후 수행해야 합니다. 현재 앱은 기존 v1 모델을 사용하며, v2 모델 파일이 추가되면 자동으로 v2를 우선 로드하도록 설계되어 있습니다.

## 현재 v1 모델

- 입력: `[batch, 100, 90]`
- 구조: Masking → LSTM(128) → LSTM(64) → Dense(32) → 16-class Softmax
- 파라미터: 164,144
- 카페 관련 16개 클래스
- 저장된 metadata 기준 train accuracy 99.59%, validation accuracy 98.91%

이 validation accuracy는 **작은 내부 데이터셋의 분할 성능이며 새로운 한국수어 사용자에 대한 실제 정확도를 뜻하지 않습니다.**

## ONNX / INT8 결과

GitHub Actions CPU smoke test, 합성 입력 300개 기준:

| 항목 | ONNX FP32 | ONNX Dynamic INT8 |
|---|---:|---:|
| 파일 크기 | 695,959 B | 239,430 B |
| 평균 추론 | 2.310 ms | 2.525 ms |
| p95 | 2.784 ms | 3.514 ms |

INT8은 파일 크기를 약 65.6% 줄였지만 CPU smoke test에서는 FP32보다 조금 느렸습니다. 따라서 INT8을 무조건 기본 모델로 사용하지 않고 실제 브라우저와 실제 수어 데이터로 검증 후 채택합니다.

## 실행

정적 파일이므로 HTTPS 환경(Vercel 권장) 또는 로컬 서버에서 실행합니다.

```bash
python -m http.server 8000
```

그 후 `http://localhost:8000`을 엽니다. 카메라는 브라우저 보안 정책상 localhost 또는 HTTPS에서 사용해야 합니다.

### 추론 엔진 강제 선택

URL query로 비교할 수 있습니다.

```
?engine=auto
?engine=onnx-fp32
?engine=onnx-int8
?engine=tfjs
```

기본 `auto`는 검증된 ONNX FP32를 먼저 시도하고 실패 시 TensorFlow.js로 fallback합니다.

## 데이터 수집

`collector.html`을 엽니다.

새 수집기는 다음 정보를 저장합니다.

- 최대 2개의 손 landmark
- 손 handedness
- 얼굴 핵심/확장 landmark
- 선택된 Face Blendshape
- 가명 참여자 코드(예: P01)
- 세션 코드(예: S01)
- 라벨, 메모, 환경 태그
- 양손/얼굴 감지 품질 요약

**실명을 입력하지 마세요.** 원시 JSON은 `.gitignore`로 공개 저장소 커밋을 차단했습니다.

## v2 학습

원시 JSON을 로컬 `data/raw/`에 둡니다.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-training.txt

python training/train_v2.py --data "data/raw/**/*.json"
```

학습기는:

1. `ieum_v2_190` 특징 생성
2. 희소 라벨 필터링
3. 참여자 단위 split 우선
4. LSTM / GRU 후보 학습
5. validation macro-F1로 후보 선택
6. 별도 test split에서 최종 평가
7. confusion matrix / classification report / split manifest 저장

을 수행합니다.

참여자 코드가 충분하지 않으면 stratified random split으로 fallback하며, 이 경우 결과가 낙관적일 수 있다는 경고를 metadata에 남깁니다.

## v2 ONNX 내보내기

v2 학습 후:

```bash
pip install -r requirements-onnx.txt
python tools/export_v2_onnx.py
```

이후 실제 검증 데이터를 이용해 FP32와 INT8 정확도 차이를 비교한 뒤 양자화 채택 여부를 결정합니다.

## 브라우저 성능 비교

`benchmark.html`에서 v1 런타임을, v2 학습 후에는 `benchmark-v2.html`에서 v2 FP32/INT8 런타임을 비교합니다.\n\n`benchmark.html`에서:

- TensorFlow.js WebGL FP32
- ONNX Runtime Web WASM FP32
- ONNX Runtime Web WASM INT8
- 지원될 경우 ONNX WebGPU FP32

의 로딩 시간, 평균/중앙값/p95 추론시간과 출력 차이를 비교합니다.

## 파일럿

앱을 다음처럼 실행합니다.

```
/?study=1&participant=P01
```

화면의 **과제 시작 → 성공/실패 → 기록 저장**을 사용합니다. 로그에는 카메라 영상, landmark, 음성, STT 문장이 저장되지 않습니다.

여러 참여자의 JSON을 `pilot-dashboard.html`에 넣으면:

- 과제 성공률
- 평균 수행시간
- 평균 재시도
- 후보 선택 횟수
- 직접입력 사용률
- 평균 모델 추론시간

을 집계하고 CSV로 내보낼 수 있습니다.

상세 절차는 `PILOT_PROTOCOL.md`를 참고하세요.

## 주요 파일

```
index.html / script.js       실제 AAC 앱
collector.html / collector.js 연구용 데이터 수집
benchmark.html              v1 브라우저 추론 벤치마크\nbenchmark-v2.html           v2 ONNX FP32/INT8 브라우저 벤치마크
pilot-dashboard.html        파일럿 결과 집계
src/feature-schema.js       v1/v2 브라우저 특징 추출
src/inference-engine.js     TF.js / ONNX 런타임
src/cafe-decoder.js         투명한 문장 자연화
src/study-logger.js         개인정보 최소 파일럿 로거
training/                   v2 학습·평가
tools/                      ONNX 변환·양자화·검증
model/onnx/                 v1 ONNX FP32/INT8
```

## 해석상의 주의

- 한국수어는 독립된 언어이며 교정 대상이 아닙니다.
- 이 도구는 한국수어 사용자가 비수어 사용자와 특정 상황에서 의사소통할 때 사용할 수 있는 AAC 보조수단을 탐색하는 프로토타입입니다.
- 현재 내부 validation 수치를 실제 사용자 성능으로 일반화하지 않습니다.
- AI가 실패할 수 있으므로 후보 선택, 직접 입력 등 대체 의사소통 경로를 유지합니다.
- 실제 임상적·사회적 효과는 한국수어 사용자와의 공동 설계 및 별도 검증이 필요합니다.


## 현재 완료선과 남은 실제 작업

코드로 수행 가능한 구조 개선·검증 자동화는 공모전용 브랜치에 반영되어 있습니다. 다만 다음은 실제 사람/기기가 필요하므로 저장소에서 임의 생성하지 않습니다.

1. 가명 참여자 코드로 실제 수어 표본 수집
2. 다인원 v2 재학습 및 held-out test 평가
3. Chrome/Android/iPhone 등 실제 기기 브라우저 QA와 benchmark JSON 수집
4. 실제 파일럿 과제 및 사용성 설문
5. 위 실측값으로 공모전 결과 문단 확정

이 단계 전까지는 v2의 정확도·파일럿 성공률을 수치로 주장하지 않습니다.


## 기존 2.0-facemesh JSON으로 재촬영 없이 학습

예전 수집 JSON에 `hands`, `handedness`, `landmarks`, `face`가 있으면 재사용할 수 있습니다. `faceExtended`/blendshape가 없는 기존 자료는 `base8` 프로필을 사용합니다.

최종 카페 파일을 그대로 쓰는 경우:

```bash
python tools/run_v2_pipeline.py \
  --data "/절대경로/samples.json" \
  --feature-profile base8
```

기존 카페의 `기타`가 적다면, 과거 카페 밖 수어를 target-vocabulary 밖 예시로 일부 재사용할 수 있습니다.

```bash
python tools/build_legacy_training_set.py \
  --cafe "/절대경로/최종카페/samples.json" \
  --unknown "/절대경로/자기소개1.json" "/절대경로/자기소개2.json" "/절대경로/자기소개3.json"

python tools/run_v2_pipeline.py \
  --data "data/raw/legacy_cafe_base8.json" \
  --feature-profile base8
```

기본 unknown 재매핑 대상은 `나`, `사랑해`, `만나서 반갑습니다`이며 각 라벨 최대 20개만 가져와 클래스 불균형을 피합니다. 이 표본들은 잘못된 수어가 아니라 **현재 카페 target vocabulary 밖의 정상적인 한국수어 표현**으로 취급합니다.
