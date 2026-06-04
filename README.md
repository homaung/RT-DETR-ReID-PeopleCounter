# RT-DETR-ReID-PeopleCounter

RT-DETR(Real-Time Detection Transformer)로 **사람을 탐지**하고, 각 사람의 **보행자 속성(나이·성별·상의색·하의색)** 을 함께 예측하는 **단일 모델**입니다.

탐지된 사람마다 두 정보를 함께 보고 속성을 읽습니다.

1. **디코더 query 임베딩** (256-d) — RT-DETR가 그 사람에 매칭한 query
2. **박스 영역 원본 픽셀** — 박스로 원본 픽셀을 잘라 작은 conv(RegionStem)로 가공한 256-d

> 핵심: 탐지용으로 압축된 query 임베딩만으로는 **색 정보가 뭉개집니다**. 박스 영역의 **원본 픽셀을 다시 조회**(RegionStem)해 query에 concat하니, 단일 모델·단일 forward로 색(up/down)까지 정확히 잡습니다.

---

## 인식 속성

| 속성 | 클래스 수 | 값 |
|------|:---:|------|
| age | 3 | young / adult / old |
| gender | 2 | male / female |
| up_color | 11 | black, blue, brown, green, grey, orange, pink, purple, red, white, yellow |
| down_color | 11 | (상동) |

> 색상은 원래 13색(+mixture, other)이었으나 라벨을 오염시키던 **mixture/other를 제거**하고 11색으로 정리했습니다. 색 미지정 샘플은 `-1`(ignore)로 처리되어 해당 속성만 학습/평가에서 제외됩니다.

---

## 성능 (UPAR val, held-out 3000장 실측)

| 속성 | 정확도 | 평가 표본 |
|------|:---:|:---:|
| age (3-way) | **96.93%** | n=3000 |
| gender (2-way) | **81.27%** | n=3000 |
| up_color (11-way) | **76.43%** | n=2800 |
| down_color (11-way) | **69.98%** | n=1329 |
| **mean** | **81.15%** | — |

> 추론 셀 `RUN_MODE="benchmark"`로 재현됩니다(val.csv held-out, 학습 미사용, GPU 61 it/s). down_color 표본이 작은 건 하의색 미지정(`-1`) 샘플이 평가에서 제외되기 때문입니다.
>
> 같은 데이터에서 query 임베딩만 쓰던 이전(head-only) 대비 **mean 66% → 81%**, 특히 색상이 크게 향상(up 53→76, down 42→70)되었습니다. color는 11-way 멀티클래스 정확도이며, 이진 per-color(mA)로 환산하면 더 높습니다.

---

## 구조

```
입력 이미지
   │  (동결 RT-DETR backbone + 인코더 + 디코더)
   ├─► dec_boxes (사람 박스) + query 임베딩 q(256)
   │                                   │ 사람 박스 ↔ query 매칭(argmin)
   └─► 박스로 원본 픽셀 crop ─► RegionStem(작은 conv 4층) ─► region(256)
                                                            │
                          concat[q(256), region(256)]=512 ─► trunk ─► 4 헤드
                                                            └─► age / gender / up / down
```

- **동결**: backbone · 인코더 · 디코더 (RT-DETR 원본, 탐지 성능 보존)
- **학습**: `RegionStem` + `trunk` + 속성 헤드 4개 (작음 → 안정적)
- 손실: per-속성 CrossEntropy, `ignore_index=-1`(미지정 색 제외), label smoothing, + warmup + grad clip
- 학습 경로는 fp32 (amp nan 회피), backbone forward는 `no_grad`/autocast로 빠르게

---

## 데이터셋 (UPAR)

Market-1501 + PA-100K 통합. 어노테이션은 [speckean/upar_dataset](https://github.com/speckean/upar_dataset)의 `dataset_all.pkl`.

**파이프라인 (노트북 자체완결)**

1. Drive 마운트 → `archive (1).zip`(PA-100K)/`archive (2).zip`(Market) → `/content/dataset` 추출
2. UPAR pkl 다운로드 → 색/속성 변환(`upar_to_label`)
3. merge: 절대경로 라벨 csv 생성 → train/val 9:1 분할
4. **Phase 1A** — 동결 RT-DETR로 각 이미지의 사람 query 박스/임베딩 추출 + **박스 필터**(objectness ≥ 0.6 미만 제외) → `query_cache_v0_3`
5. 학습 시 **박스 area 0.7~0.9**(전신이 깔끔히 들어온 것)만 사용 → 잘린 사람의 속성 노이즈 제외

---

## 학습 가속 — 동결부 출력 사전계산 캐시 (Phase 1A)

RT-DETR backbone·인코더·디코더는 **학습 내내 동결**이라 같은 이미지에 대해 매 epoch 똑같은 값을 낸다. 그래서 이 무거운 forward를 **데이터셋 1회만 돌려 출력을 Drive에 미리 저장**해 두고, 학습 루프는 가벼운 모듈만 갱신한다.

**미리 저장하는 것** (`query_cache_v0_3/{train,val}.pt`, ~70MB):

| 항목 | 내용 | 학습에서의 쓰임 |
|---|---|---|
| `X` | 사람 query 256-d 임베딩(fp16) | 헤드 입력 query 벡터 |
| `boxes` | 동결모델 pseudo-GT 박스(cxcywh) | query↔박스 매칭 · roi 영역 |
| `Y` | 4속성 라벨 | 손실 타깃 |
| objectness | query 사람 점수 | **박스 필터**(≥0.6) · **area 0.7~0.9** 선별 |

**효과**
- 데이터셋 전체에 대한 RT-DETR 트랜스포머 forward를 **epoch마다 반복하지 않고 1회로 끝냄** → 학습 시간 대폭 단축.
- 박스 품질 필터·전신 박스 선별이 **사전 확정**돼 학습 루프가 단순·경량(작은 `RegionStem`+헤드만 역전파).
- 한 번 만들면 Drive에 영구 저장 → 재학습·하이퍼파라미터 탐색 때 **추출 단계 스킵**(`FORCE_REEXTRACT=False`).

> 참고: region-branch는 색 증강(ColorJitter/flip)으로 픽셀이 바뀌므로 `RegionStem` 입력용으로는 backbone을 다시 통과하지만(동결·`no_grad`, 역전파 없음), 캐시가 박스 필터·pseudo-GT·매칭을 사전 확정해 주는 덕에 학습 루프는 여전히 가볍다. 증강을 끄면 `X`를 그대로 써 backbone 호출까지 생략할 수 있다.

---

## 실행

`RT-DETR_ReID.ipynb` 한 노트북에서 위→아래 순서대로 실행합니다 (Colab 기준).

| 단계 | 셀 |
|---|---|
| 셋팅 | 마운트 → 이미지 추출 → pkl 다운로드 → 색 변환 → merge → split |
| 박스 | Phase 1A (박스 필터 + pseudo-GT 박스 추출) |
| 학습 | RegionStem + 헤드 학습 → `checkpoints/region_attr/best.pt` |
| 추론 | 테스트 준비(데모 영상) → 추론 |

### 추론 모드 (`RUN_MODE`)

| 모드 | 카운팅 단위 | 설명 |
|---|---|---|
| `benchmark` | — | val.csv(held-out)로 속성별 정확도 측정 |
| `image` | **이미지 파일당 1행** | 폴더 내 이미지마다 사람 수·필터 매칭 수를 `image_counts.csv`에 1줄씩 로그, 오버레이 저장 |
| `video` | **영상 전체 누적** | RT-DETR 탐지 + ByteTrack 추적 + 속성 누적 투표 → **고유 인물 누적 카운트**, 매칭 영상/`video_counts.csv` 저장 |
| `webcam` | **실시간 누적** | Colab 웹캠, 프레임마다 누적 고유 카운트 갱신 표시, 로그 CSV |

> `image`/`video`는 `SOURCE`(파일 또는 폴더)의 확장자로 자동 분기됩니다. 폴더에 이미지·영상이 섞여 있으면 각각 처리됩니다.

---

## 특정 속성 선별 피플 카운터

이 SW의 핵심 기능. **원하는 속성을 정의하면 그 조건에 맞는 사람만 카운팅**합니다.

추론 셀 상단 `ATTR_FILTER` 한 줄로 카운팅 대상을 지정합니다.

```python
ATTR_FILTER = None                                  # 전체 사람 카운팅
ATTR_FILTER = {"gender": "female"}                  # 여성만
ATTR_FILTER = {"up_color": "red"}                   # 빨간 상의만
ATTR_FILTER = {"gender": "male", "up_color": "black", "age": "adult"}  # 복합 조건(AND)
```

| 속성 키 | 정의 가능한 값 |
|---|---|
| `age` | young / adult / old |
| `gender` | male / female |
| `up_color` | black, blue, brown, green, grey, orange, pink, purple, red, white, yellow |
| `down_color` | (상동 11색) |

**카운팅 방식**
- **탐지** — 원본 RT-DETR로 사람을 탐지(검증된 detector)
- **속성** — 박스별 query 임베딩 + 박스 원본 픽셀 조회(RegionStem) → 4속성 예측
- **추적·안정화** — ByteTrack으로 동일 인물에 `track_id` 부여, **시간 누적 다수표**로 한 사람의 속성을 한 값으로 고정(깜빡임 제거)
- **카운트** — `ATTR_FILTER` 조건을 **한 번이라도 통과한 고유 track_id 수**를 누적 → 같은 사람 중복 집계 없음

오버레이는 `Filter / In frame(현재 화면) / TOTAL counted(누적 고유)` 3줄을 표시하고, 매칭 인물은 초록 박스, 비매칭은 회색 얇은 박스로 구분합니다.

---

## 개발 과정

### 0. 문제 정의
단일 카메라에서 사람을 탐지하고 **속성(나이·성별·상의색·하의색)** 을 읽어, **특정 속성 조건에 맞는 사람만 선별 카운팅**하는 피플 카운터. 탐지와 속성인식을 별도 2모델로 돌리지 않고, **검증된 RT-DETR 탐지기를 그대로 살리면서 속성 가지를 얹는 단일 모델**로 설계해 추론 속도·일관성을 확보했다.

### 1. 데이터셋 구축 (UPAR)
- Market-1501 + PA-100K를 통합한 UPAR 어노테이션(`dataset_all.pkl`) 사용, 4속성으로 변환.
- **색 라벨 정제**: 원래 13색(+mixture, other)이 라벨을 오염시켜 **mixture/other 제거 → 11색**으로 축소. 미지정 색은 `-1`(ignore)로 두어 해당 속성만 학습/평가에서 제외.
- train/val 9:1 분할, 절대경로 csv로 관리.

### 2. 박스 품질 필터 (Phase 1A)
- 동결 RT-DETR로 각 이미지의 사람 query 박스·임베딩을 미리 추출해 캐시.
- **objectness ≥ 0.6** 미만 박스 제외, 학습 시 **박스 면적 0.7~0.9**(전신이 깔끔히 들어온 것)만 사용 → 잘린 사람의 속성 노이즈 차단.

### 3. 모델 구조 반복 (핵심 시행착오)
1. **head-only (query만)**: query 임베딩(256)만 헤드에 연결 → 안정적이나 **색 정확도 낮음**(up 53 / down 42). 탐지용으로 압축된 query엔 색 정보가 뭉개짐을 확인.
2. **디코더/박스헤드까지 같이 학습**: 워밍업된 헤드를 디코더 학습이 흔들어 붕괴(§4).
3. **2-stage (ResNet50 크롭 분류 + 탐지기 결합)**: 학습이 끝내 수렴 안 됨(loss 폭발/nan) → 폐기.
4. **region-branch (최종 채택)**: query는 그대로 두고, **박스 영역 원본 픽셀을 다시 잘라(roi_align) 작은 conv(RegionStem)로 256-d를 새로 뽑아** query와 concat → 헤드. "색은 원본 픽셀에 있다"는 가설이 적중.

### 4. 학습 붕괴 진단
- 증상: val mean 20%, age 7% 등 랜덤 이하로 추락.
- 진단: (1) FocalLoss + 역빈도 class weight = 불균형 보정 **이중 적용** → 희귀클래스 과예측, (2) weighted-CE로 바꿔도 붕괴 → **class weight 자체가 원인**, (3) **디코더/박스헤드 동시 학습이 워밍업된 헤드를 파괴**.
- 교훈: **검증된 탐지부는 동결**, 작은 학습 모듈만 얹는 게 안정적.

### 5. 최종 아키텍처
동결 RT-DETR(backbone+encoder+decoder) → query q(256) + 박스 원본픽셀 RegionStem(256) → concat(512) → trunk → 4헤드. 손실은 per-속성 CE(`ignore_index=-1`) + label smoothing + warmup + CosineAnnealing + grad clip 5.0, fp32 경로(amp nan 회피).

### 6. 결과
head-only 대비 **mean 66 → 81%**, 색상 대폭 향상(up 53→76, down 42→70). 검증된 탐지부를 살린 단일 모델·단일 forward로 속도와 정확도를 동시에 확보.

### 7. 피플 카운터
탐지(RT-DETR) → 속성(region-branch) → 추적(ByteTrack) → 시간 누적 다수표 → **선별 카운팅**. `ATTR_FILTER`로 조건(AND)을 정의하고, 같은 사람은 `track_id`로 중복 제거해 누적한다.

---

## 학습/배포 메모

- 사전학습 가중치는 `rtdetr-l.pt`(ultralytics)를 자동 다운로드합니다.
- 체크포인트(`best.pt`)에는 학습한 `stem`/`head`만 저장됩니다(backbone은 추론 시 원본에서 로드).
- API 키(Kaggle/W&B 등)는 공개 전 반드시 마스킹하세요.
