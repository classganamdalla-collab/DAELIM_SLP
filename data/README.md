# 데이터 수집 폴더

실제 참여자/사용자 원시 JSON은 **공개 GitHub 저장소에 올리지 않습니다.**

권장 구조:

```
data/
  raw/
    P01_session01.json
    P01_session02.json
    P02_session01.json
```

수집기에서는 실명 대신 `P01`, `P02` 같은 가명 코드를 사용하세요.

## 재현 가능한 평가를 위한 최소 원칙

- 같은 사람이 여러 번 수행한 표본이 train/validation/test에 무작위로 섞이지 않도록 합니다.
- 가능하면 참여자 단위로 분리합니다.
- 조명, 거리, 속도, 카메라 각도를 다양화합니다.
- 각 라벨의 표본 수를 균형 있게 유지합니다.
- `기타` 라벨은 임의 동작을 폭넓게 모으되, 하나의 의미 있는 수어처럼 해석하지 않습니다.
- 원본 동영상/얼굴 영상은 연구 목적상 꼭 필요하지 않다면 저장하지 않습니다.

`training/train_v2.py`는 참여자 코드가 충분하면 participant-group split을 우선 사용하고, 그렇지 못할 때만 stratified random split으로 fallback합니다. fallback 결과는 일반화 성능으로 과장해서 해석하면 안 됩니다.
