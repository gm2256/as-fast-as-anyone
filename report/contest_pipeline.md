대회 우승 및 상위권 진입을 위한 **가장 효율적인 실전 개발 로드맵 순서**입니다. 무작정 데이터 전처리부터 크게 벌리기보다는, **'베이스라인 확보 ➔ 핵심 데이터 구축 ➔ 파인튜닝 ➔ 고도화/앙상블'** 순서로 빠르게 검증 루프(Feedback Loop)를 돌리는 것이 핵심입니다.

---

## 대회 우승 전략 파이프라인 (추천 개발 순서)

```
[ Step 1. 베이스라인 설치 & 파이프라인 검증 ]
   ↓
[ Step 2. 데이터 샘플링 & 타깃 데이터셋 구축 ]
   ↓
[ Step 3. Behavioral Cloning (BC) 사전 학습 ]
   ↓
[ Step 4. 커스텀 보상 함수 설계 & RL Fine-tuning ]
   ↓
[ Step 5. 모델 앙상블 & Post-Processing (후처리) ]

```

---

### Step 1: 베이스라인 구축 및 파이프라인 검증 (1순위)

전체 데이터 전처리나 복잡한 모델을 짜기 전에, **대회 측이 제공한 코드 환경이 내 컴퓨팅 환경(RTX 4090 등)에서 정상 작동하는지 확인**하는 최우선 과제입니다.

1. **환경 세팅**: `uv`를 이용해 Ubuntu 24.04, JAX, Waymax, V-Max 가상환경 구축.


2. **Sanity Check (맛보기 학습)**: 샘플 데이터 몇 개만 넣고 기본 SAC/PPO 베이스라인 스크립트를 1~2 에포크 돌려 학습/평가/제출 파일 생성 루프 확인.


3. **제출 및 평가 테스트**: `WaymaxActorCore` 규격에 맞는 체크포인트 추출 및 추론 속도($T_{infer}$) 기준점(Benchmark) 확보.



---

### Step 2: 전략적 데이터 샘플링 및 전처리 (2순위)

26만 Scene(428GB)을 한 번에 다 돌리는 것은 시간 낭비입니다. **학습 시간을 1/5로 줄이면서 성능을 유지할 핵심 데이터셋**을 먼저 추출합니다.

1. **TFRecord ➔ pkl 변환**: 제공된 코드로 pkl 데이터 세팅.


2. **시나리오 필터링 (Hard/Corner-Case Mining)**:
* **직진 위주 단순 데이터 제거**: 단순 차선 유지 시나리오 비율 축소.
* **고난도 데이터 추출**: 곡선/교차로(Yaw-rate가 높은 경우), 급가감속(가속도 변화가 큰 경우), 주변 동적 객체(차량/보행자) 밀집 지역 우선 추출.

3. **빠른 검증용 Dataset 분리**:
* **Quick-Train Set (1만~2만 Scene)**: 모델 및 보상 함수 빠른 실험용 (1~2시간 이내 학습 가능).
* **Final-Train Set (5만~10만 Scene)**: 최종 제출용 고난도 위주 데이터셋.

---

### Step 3: 모델 선정 및 Imitation Learning (BC) 사전 학습 (3순위)

강화학습(RL)을 Random Initialization 상태에서 시작하면 탐색 단계에서 수많은 충돌이 발생하고 학습이 늦어집니다.

1. **모델 아키텍처 확정**:
* 입력: 과거 N-frame 센서/궤적 히스토리 + 정밀지도 Vector Map.
* 네트워크: Temporal Context(GRU/Transformer) + Feature Extractor.
* *주의*: PDM 기준 1.5배 추론 속도 컷오프를 고려해 경량 아키텍처 유지.


2. **Behavioral Cloning (BC) Pre-training**:
* Step 2에서 정제한 데이터의 Ground Truth 궤적으로 Policy Network를 Supervised Learning으로 먼저 학습.
* 이 단계만 완성해도 차량이 충돌 없이 유연하게 도로를 달리는 수준에 도달함.


---

### Step 4: Custom Reward Engineering & Safe RL Fine-tuning (4순위)

사전 학습된 가중치 위에서 **대회 평가 공식에 특화된 RL 파인튜닝**을 적용해 점수를 극대화합니다.

1. **파인튜닝 전용 Safety Technique 적용**:
* **Critic Warm-up**: Actor 가중치는 고정한 채, Critic(Q-Network)만 2,000~5,000 step 먼저 학습시켜 가중치 파괴 방지.
* **Replay Buffer Warm-start**: Expert Trajectory 데이터로 Replay Buffer를 채운 뒤 시작.


2. **대회 수식 맞춤형 커스텀 보상 함수 적용**:
* **Hard Penalty**: 충돌(`no_collision=0`) 및 도로 이탈(`on_road=0`) 시 대형 음수 보상 부여.

* **Progress (70%)**: 목적지 방향 진행 속도 및 Distance-to-goal 가중치.

* **Comfort (30%)**: 급조향(Yaw-rate), 급제동/급가속(Jerk) 제곱합 감점.

3. **PPO / SAC RL Fine-tuning 진행** (Quick-Train Set으로 실험 후 Final-Train Set에 적용).



---

### Step 5: 모델 앙상블 & 안전 후처리 (Post-Processing) (5순위 - 우승 결정타)

리더보드 상위권 다툼 및 최고점 달성을 위한 최종 마무리 단계입니다.

1. **다양성 기반 앙상블 (Ensemble)**:
* 서로 다른 Random Seed / 보상 가중치를 달리한 Top-K Checkpoints(3~5개)의 Action/Trajectory 평균(Action Averaging) 또는 Q-value 가중 평균(Critic-Weighted Ensemble) 수행.


2. **Safety & Comfort Guardrail (후처리)**:
* **Action Smoothing**: Low-Pass Filter(LPF) 적용 $a_t' = \alpha a_t + (1-\alpha) a_{t-1}$ 시켜 Jerk 감점 최소화.
* **Kinematics Clipping**: 차량 물리적 한계를 벗어나는 출력값 강제 제한.


3. **Inference Optimization**:
* ONNX / TensorRT 변환 등으로 RTX 4090 기준 추론 시간을 28ms 이내로 단축시켜 실시간성 컷오프 완전 방지.



---

### 💡 핵심 요약

1. **베이스라인 돌리기** ➔ 2. **어려운 데이터만 2만 개 선별하기** ➔ 3. **모범 답안으로 사전 학습(BC)하기** ➔ 4. **Critic 먼저 데운 후 RL 파인튜닝하기** ➔ 5. **3~5개 모델 앙상블 + LPF 후처리하기** 순서로 진행하는 것이 우승에 가장 가까운 최적 로드맵입니다!