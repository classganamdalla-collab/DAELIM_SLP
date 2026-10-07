# 공모전 실측 근거 집계

이 디렉터리는 실제 파일럿·브라우저 benchmark 결과를 로컬에서만 모아 공모전 결과 작성에 쓰기 위한 구조입니다.

원시 로그와 기기 정보는 공개 저장소에 올리지 않도록 gitignore 처리되어 있습니다.

## 파일 배치

reports/input/pilot/ 아래에 파일럿 로그와 설문 JSON을 넣고,
reports/input/benchmark/ 아래에 benchmark.html에서 저장한 JSON을 넣습니다.

## 집계

python tools/build_evidence_report.py

생성 결과:
- reports/local/evidence-summary.md
- reports/local/evidence-summary.json

v2 학습 결과가 model/v2/metrics.json에 존재하면 held-out test 결과도 함께 포함합니다.
실제 파일이 없는 항목은 아직 없음으로 표시하며 수치를 임의 생성하지 않습니다.
