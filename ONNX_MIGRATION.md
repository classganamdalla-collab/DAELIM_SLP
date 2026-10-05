# 이음 ONNX/INT8 마이그레이션 실험

이 브랜치는 기존 `main`을 보존한 채 **TensorFlow.js FP32 → ONNX Runtime Web FP32 → ONNX Dynamic INT8**을 비교하기 위한 실험 브랜치입니다.

## 원칙

- 기존 TF.js 모델은 기준선(baseline)으로 유지합니다.
- ONNX가 실제로 더 빠르고, 수치 일치성과 실제 인식 성능이 허용 범위 안일 때만 최종 런타임으로 채택합니다.
- `validation-report.json`의 무작위 합성 입력 결과는 **수치 일치성/런타임 smoke test**일 뿐, 실제 수어 인식 정확도가 아닙니다.
- 실제 정확도는 별도 검증 데이터와 타인 파일럿에서 측정해야 합니다.

## 자동 빌드

GitHub Actions의 **Build ONNX variants**가 다음을 수행합니다.

1. 기존 `model/tfjs_model/model.json`을 ONNX FP32로 변환
2. Dynamic INT8 양자화
3. FP32 ↔ INT8 수치 일치성 및 CPU 추론시간 smoke test
4. 아래 결과물을 브랜치에 자동 커밋

```
model/onnx/sign_language_fp32.onnx
model/onnx/sign_language_int8.onnx
model/onnx/validation-report.json
```

변환은 현재 tf2onnx의 TFJS 입력 경로를 사용합니다. TFJS 변환은 tf2onnx에서도 실험적 기능이므로, 브라우저의 `benchmark.html`에서 기존 TF.js 출력과 ONNX FP32 출력의 일치성을 반드시 확인합니다.

## 브라우저 벤치마크

정적 서버/Vercel에서:

```
/benchmark.html
```

을 열고 **벤치마크 실행**을 누릅니다.

비교 항목:

- TF.js FP32
- ONNX Runtime Web WASM FP32
- ONNX Runtime Web WASM INT8
- 평균/중앙값/p95 추론시간
- ONNX 모델 크기
- 동일 합성 입력에 대한 출력 오차와 argmax 일치 여부

## 채택 기준

ONNX INT8은 아래 조건을 모두 만족할 때 최종 앱에 적용하는 것을 권장합니다.

1. 실제 사용자/검증 데이터에서 클래스별 성능 저하가 미미함
2. 브라우저 벤치마크에서 평균뿐 아니라 p95 지연시간도 개선됨
3. 모바일/노트북에서 오류 없이 동작
4. 모델 로딩 시간과 메모리 사용량이 악화되지 않음

## 다음 단계

이 실험이 성공하면 별도 작업으로:

- 추론 런타임 추상화(TF.js/ONNX fallback)
- 양손 특징 벡터 재설계
- 새 데이터 수집 및 재학습
- confusion matrix / precision / recall / macro-F1 평가
- 실제 카페 역할극 파일럿

을 진행합니다.
