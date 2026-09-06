# dxchallenge_planning_eval

DX 챌린지(motion planning) 평가 프로그램. 참가자 제출물(planner)을 rideflux
데이터셋 위에서 시뮬레이션하고 **rideflux score**를 채점한다.
pure Waymax 기반이며 V-Max에 의존하지 않는다 (필요한 metric은
`challenge_metrics/`에 V-Max에서 복사).

## 환경 및 추가 파이썬 패키지 (requirements.txt)

권장 시스템: **Ubuntu 24.04 LTS (x86_64)** + **최신 NVIDIA 드라이버** (번들된
CUDA 13 휠이 드라이버 >= 580을 요구). 시스템에 CUDA/cuDNN을 설치할 필요는
없다 — CUDA 유저스페이스 라이브러리는 전부 파이썬 휠로 설치된다.

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone 후 설치
git clone <this-repository>
cd dxchallenge_planning_eval
uv sync
```

`uv sync`는 Python 3.12(`.python-version`에 고정, 없으면 자동 다운로드)로
`.venv`를 만들고, `uv.lock`에 기록된 정확한 의존성 버전(CUDA 13 런타임 포함
JAX)을 설치한다.

평가 환경은 이렇게 `pyproject.toml` + `uv.lock`으로 고정되어 있다. 기본 제공:
jax(CUDA)==0.11.0, flax==0.12.8, numpy==2.5.1, tensorflow-cpu==2.21.0,
waymax(고정 rev)와 그 의존성(optax, chex 등).

기본 환경에 없는 패키지가 필요하면 제출 디렉토리에 `requirements.txt`를
포함하면 된다. 평가 직전에 다음 절차로 설치된다:

```bash
uv sync                                     # 기본 평가 환경 구성 (pyproject.toml + uv.lock)
uv add -r <submission>/requirements.txt     # 참가자 패키지 설치 (충돌 시 여기서 거부)
```

- **jax 스택·flax·numpy는 `==`로 고정된 직접 의존성**이라, 다른 버전을
  요구하는 requirements는 설치 단계에서 해석 실패로 거부된다. 그 외 패키지는
  참가자 패키지가 요구하면 버전이 조정될 수 있다.
- 본인 재현성을 위해 가능하면 `패키지==버전`으로 정확히 핀할 것.
- 제출 전 셀프 체크: 이 repo에서 `uv sync` 후
  `uv add -r <제출 디렉토리>/requirements.txt`를 돌려보면 주최측과 동일한
  환경을 미리 확인할 수 있다.
- 순수 파이썬 패키지는 requirements 없이 제출 디렉토리에 패키지 폴더를
  통째로 넣어도 된다 (제출 디렉토리가 `sys.path` 앞에 추가되므로 import 됨).
- 예시: `submission_example_vmax_sac/requirements.txt` (einops).

## 실행

```bash
CUDA_VISIBLE_DEVICES=0 uv run evaluate.py \
    --path_dataset /path/to/validation.tfrecord@???? \
    --submission <submission>
```

결과: stdout 요약 + `<output_dir>/evaluation_episodes.csv`(시나리오별) +
`evaluation_results.txt`(평균). `--output_dir` 기본값은 `results/<제출물 이름>`.

제출물 로딩·jit 컴파일을 포함한 전체 평가가 `--time_limit`(**기본 1800초 =
30분**) 안에 끝나지 않으면 실패로 처리된다 (`TIME LIMIT EXCEEDED` 출력 후
비정상 종료, 결과 파일 없음). 0 이하를 주면 제한이 없다.

평가는 항상 **lockstep batch**로 돈다: actor의 `init`/`select_action`을
batch에 vmap하므로 호출 하나하나는 여전히 unbatched state를 받는다
(per-scenario 의미는 동일). 따라서 **제출물은 JAX-traceable해야 한다**
(아래 제약 섹션 참조; batch 안에서 예외가 나면 그 batch 전체가 0점 처리됨).
시나리오 순회는 waymax 기본 파이프라인을 사용한다. 단일 파일 데이터셋(공식
평가셋)에서는 레코드 순서 = 파일 내 순서로 고정된다. `@N` shard 데이터셋에서는
병렬 interleave 때문에 순회 순서가 파일순이 아닐 수 있으나, 모든 레코드가
정확히 한 번씩 평가되고 episode별 점수는 순서·batch 구성과 무관하므로 전체
평균 점수는 동일하다. 단, batch 크기가 다르면 XLA가 다른 커널을 컴파일해 per-step 수치가
미세하게 달라질 수 있다.

## 시뮬레이션 구성

- `PlanningAgentEnvironment`: ego(SDC)만 참가자 planner가 제어, 나머지 객체는
  logged trajectory 재생.
- ego dynamics: `InvertibleBicycleModel(normalize_actions=True)`
  (V-Max 베이스라인과 동일).
- episode: 9초(91 step) 중 warmup 11 step 후 80 step 전체 시뮬레이션.
  **충돌(overlap) 또는 도로이탈(offroad_in_box)이 한 번이라도 발생하면
  해당 시나리오는 0점** (점수식의 곱셈 게이트).

## 참가자에게 주어지는 입력 (expert log 차단)

매 step, planner는 `waymax.datatypes.SimulatorState`를 받는다. 단:

- `log_trajectory`는 **전체가 invalid 처리 + 값 0으로 소거**된다. 유일한
  예외: **ego의 마지막 logged 위치(x, y)는 goal로 남겨진다**. 
  추출 예시는 `submission_example_const_vel/actor.py`의 `get_goal_xy` 참조.
- 관측은 `sim_trajectory`로 한다: warmup 히스토리(step 0~10)와 지금까지의
  rollout이 들어 있고, 비-ego 객체는 logged 궤적을 그대로 재생한다.
  현재 timestep 이후는 invalid.
- `log_traffic_light`는 현재 timestep까지 보인다.
- observation 정의 / feature extraction은 전적으로 참가자 코드 몫이다.

## 제출물 형식

디렉토리 하나. 필수 파일은 `actor.py`:

```python
def create_actor(submission_dir: str) -> waymax.agents.actor_core.WaymaxActorCore:
    ...
```

- `WaymaxActorCore`를 상속(또는 `actor_core_factory` 사용)한 planner를 반환.
- **`init`/`select_action`은 JAX-traceable해야 한다** (아래 제약 섹션).
- weight 파일 등은 같은 디렉토리에 두고 `submission_dir` 기준으로 로드.
- (선택) 모듈 상수 `BATCH_SIZE = N`을 선언하면 그 batch 크기로 평가된다
  (선언 없으면 64). 채점은 **RTX Pro 5000 Blackwell (48GB)** 에서 수행되므로 본인 모델이
  감당할 수 있는 값을 선언할 것 — 너무 크면 OOM으로 제출이 실패하고, 너무
  작으면 느려져 평가 시간 제한(30분)에 걸릴 수 있다.
- `select_action(params=None, state, actor_state, rng)`이 반환하는
  `WaymaxActorOutput.action`:
  - `data`: float32 `(2,)` = (가속도, 조향), 각각 [-1, 1]
    (내부적으로 ±6.0 m/s², ±0.3 curvature로 스케일). 범위 밖 값(±inf 포함)은
    [-1, 1]로 clip되고, NaN 성분은 0.0으로 치환된다.
  - `valid`: bool `(1,)`
- 예시: `submission_example_const_vel/` (등속 주행 planner),
  `submission_example_vmax_sac/` (V-Max SAC baseline).

## 채점

시나리오(에피소드)마다 전체 80 step에 대해 집계:

| metric           | per-episode 집계          |
| ---------------- | ------------------------- |
| `progress_ratio` | 마지막 step 값 (logged 경로 대비 진행률) |
| `comfort`        | step 평균 (nuPlan 6개 임계값 만족 비율) |
| `overlap`        | max (충돌 여부 0/1)       |
| `offroad_in_box` | max (도로이탈 여부 0/1)   |

```
rideflux_score = (7·clip(progress_ratio,0,1) + 3·comfort)/10 × (1−overlap) × (1−offroad_in_box)
```

최종 점수 = 전체 시나리오의 rideflux_score 평균. 참가자 코드가 특정
시나리오에서 예외를 던지면 해당 시나리오는 0점 처리(`error` 컬럼 표시).

제한시간(30분) 초과하면 0점처리: testset 대략 5만 시나리오 (베이스라인모델 5분30초 @RTX4090)

## 제약: JAX-traceable 코드

평가기는 actor를 `jit(vmap(...))` 안에서 호출한다. 즉 `init`/`select_action`은
JAX가 tracer(값 없이 shape/dtype만 있는 추상 입력)로 녹화할 수 있는 코드여야
한다.

제출물 안에서 직접 `jax.jit`를 걸 필요는 없다 — 평가기가 `select_action`을
`jit(vmap(...))`으로 감싸므로 내부 jit은 있어도 인라인되어 의미가 없다.
