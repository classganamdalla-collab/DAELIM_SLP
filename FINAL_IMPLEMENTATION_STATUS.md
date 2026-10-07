# 이음 2026 공모전 개선판 — 구현 상태

기준 브랜치: contest-2026-v2-final

이 문서는 “코드로 완료된 것”과 “실제 사람/기기가 있어야 완료되는 것”을 분리해 기록합니다.

## 코드로 완료

### 추론 및 모델 런타임
- 기존 TensorFlow.js FP32 모델 보존
- v1 Keras H5 → ONNX FP32 변환
- Dynamic INT8 양자화
- ONNX Runtime Web WASM 지원
- WebGPU 후보 benchmark 지원
- ONNX 오류 시 v1 TensorFlow.js fallback
- 런타임/모델 계약 검사

### v1 실제 자동 실험
- ONNX FP32 695,959 B
- ONNX INT8 239,430 B
- 크기 감소 65.60%
- CI CPU 합성 입력 300개:
  - FP32 평균 2.310 ms
  - INT8 평균 2.525 ms
  - argmax agreement 99.0%
- 결론: INT8은 작지만 해당 CPU에서는 더 빠르지 않았으므로 자동 기본 채택하지 않음

### v2 입력
- ieum_v2_190 특징 설계
- 왼손/오른손을 별도 슬롯으로 보존
- 각 손 local-normalized 21 landmarks
- palm normal
- 얼굴 기준 손목 위치
- 양손 간 위치
- 얼굴 geometry
- 입/눈/눈썹 파생 특징
- 선택 Face Blendshape
- left/right/face presence mask
- JS ↔ Python 특징 numerical parity CI

### 데이터 수집
- 최대 2손 landmark 저장
- handedness 저장
- 확장 얼굴 landmark 및 blendshape 저장
- 참여자/세션 가명 코드
- 환경/주사용손 태그
- 카메라 영상 픽셀은 JSON에 저장하지 않음
- raw participant JSON 공개 Git 제외
- dataset audit 도구

### 학습/검증
- participant-group split 우선
- session-group split 차선
- stratified random fallback 및 경고
- grouped split 자동 smoke test
- LSTM / GRU 동일 split 비교
- validation macro-F1 우선 모델 선택
- held-out test 1회 최종 평가
- accuracy / macro-F1 / weighted-F1
- bootstrap 95% CI
- confusion matrix
- classification report
- test sample별 prediction CSV
- participant별 test metric
- 기타(unknown) 클래스 기반 false-accept 평가
- selective rejection threshold calibration

### v2 배포 모델 검증
- Keras → ONNX FP32 내보내기
- Keras ↔ ONNX FP32 held-out test parity
- ONNX FP32 → Dynamic INT8
- ONNX FP32 ↔ INT8 held-out test accuracy/F1 비교
- INT8 accuracy gate
- 브라우저 v2 FP32/INT8/WebGPU benchmark 페이지
- benchmark JSON 내보내기

### AAC 안전장치
- confidence + top1/top2 margin
- 낮은 확신도 자동 메시지 억제
- 상위 후보 선택 repair
- 직접 텍스트 입력 fallback
- 빠른 표현
- 인식 단어와 자연화 문장 구분
- 생성형 AI 없이 deterministic café decoder
- canonical label ID 기반 decoder
- 핵심 카페 문장 자동 테스트

### 양방향 의사소통
- 수어 → 문자/음성
- 상대방 음성 → 문자(STT)
- 브라우저 STT 미지원 처리
- STT 서버 처리 가능성 개인정보 안내

### 파일럿
- 고정 T01~T10 과제
- 기능적 성공
- AI 단독 성공
- 후보 repair 성공
- 직접입력 fallback 성공
- 수행시간 / 재시도 / 추론 latency
- 사용성 5점 설문
- JSON/CSV 대시보드
- 로그에 영상/raw landmark/음성/STT transcript 비저장
- 실제 결과 자동 근거 요약 도구

### 접근성/QA
- live region 및 상태 알림
- 버튼 aria-label/aria-pressed
- keyboard focus 표시
- reduced-motion 대응
- 모바일 반응형 UI
- 자동 CI syntax/feature/parity/split 검사
- 실제 기기 QA 체크리스트

## 실제 데이터가 있어야 완료

아래는 코드로 대신 만들거나 임의 수치를 생성할 수 없습니다.

1. 최소 여러 참여자의 실제 raw landmark 데이터 수집
2. v2 LSTM/GRU 실제 재학습
3. participant-held-out test 결과 확정
4. 실제 Chrome/Android/iPhone 브라우저 benchmark
5. 실제 참여자 T01~T10 파일럿
6. 실제 사용성 설문
7. 실측 수치로 공모전 결과 문단 확정

## 실제 데이터 수집 후 실행

1. data/raw 아래에 가명 참여자별 JSON 저장
2. dependencies 설치:
   pip install -r requirements-v2.txt
3. 전체 파이프라인:
   python tools/run_v2_pipeline.py --data "data/raw/**/*.json"
4. benchmark.html 및 benchmark-v2.html에서 실제 기기 결과 JSON 저장
5. 파일럿 및 설문 JSON을 reports/input 아래에 저장
6. 근거 요약:
   python tools/build_evidence_report.py

## 금지할 주장

- “한국수어 99% 번역 정확도”
- v1 내부 validation 98.91%를 실제 사용자 정확도로 표현
- 실제 파일럿 전 성공률/만족도 수치 작성
- INT8이 무조건 더 빠르다는 주장
- 얼굴 landmark 사용을 완전한 비수지문법 인식으로 표현
- 상황 특화 프로토타입을 범용 한국수어 번역기로 표현
