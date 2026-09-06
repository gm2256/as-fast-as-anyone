AI/RL 프로젝트 코드 정리 및 리팩토링

이 프로젝트는 일반적인 웹/백엔드 애플리케이션이 아니라 JAX 기반 강화학습(RL) 학습 프로젝트다.

따라서 일반적인 소프트웨어 프로젝트의 폴더 구조를 강제로 적용하지 말고, 학습 실험을 재현하고 관리하기 쉬운 구조를 만드는 것을 최우선으로 한다.

1. 가장 중요한 원칙: 서드파티 코드는 건드리지 않는다

현재 프로젝트에는 다음과 같은 서드파티 코드가 포함되어 있다.

V-Max/
대회 측에서 제공한 JAX/RL 학습 프레임워크
원본 코드를 최대한 유지해야 한다.
특히 vmax/agents/, vmax/simulator/ 등 내부 코드는 리팩토링하지 않는다.
dxchallenge_planning_eval/
대회 공식 채점 프로그램
제출 호환성을 위해 원본 그대로 유지해야 한다.
내부 코드를 수정하거나 구조를 변경하지 않는다.
절대 수정하지 말아야 할 영역
V-Max/vmax/
V-Max/ 내부의 기존 서드파티 코드
dxchallenge_planning_eval/


단, 이미 우리가 직접 수정한 파일이 서드파티 영역에 존재한다면 그 파일은 예외적으로 검토 대상이 될 수 있다.

하지만 해당 파일을 수정할 경우에도 기존 프레임워크의 동작을 변경하지 않는 범위에서만 최소한으로 리팩토링한다.

2. 이번 작업의 실제 대상

이번 작업에서 정리하려는 것은 이번 프로젝트에서 우리가 직접 작성하거나 수정한 코드다.

특히 다음과 같은 코드를 중심으로 확인한다.

V-Max/scripts/
    score_scenarios.py
    sample_shards.py
    split_hard_easy_pools.py
    shrink_roadgraph.py
    기타 우리가 추가한 스크립트

sac_trainer.py
sim_factory.py
train.py
기타 이번 프로젝트에서 직접 추가/수정한 코드


먼저 git diff, git status, 파일 생성/수정 시점, 코드 내용을 확인해서

"서드파티 원본 코드"와 "우리가 작성/수정한 코드"를 구분해라.

확실하게 구분할 수 없는 파일은 임의로 수정하지 말고 나에게 알려라.

3. 리팩토링의 목적

목적은 코드를 예쁘게 만드는 것이 아니다.

다음 목적을 만족하는 것이 가장 중요하다.

RL 학습 코드를 이해하기 쉽게 만든다.
학습 실험을 재현하기 쉽게 만든다.
데이터 전처리/샘플링 코드를 찾기 쉽게 만든다.
학습(train)과 평가/evaluation 코드를 명확하게 구분한다.
실험용 스크립트와 실제 학습에 필요한 코드를 구분한다.
여러 실험에서 공통으로 사용하는 코드를 적절하게 분리한다.
JAX/RL 코드의 실행 동작을 변경하지 않는다.
대회 제출 환경과의 호환성을 유지한다.
4. 폴더 구조를 결정하는 기준

웹 프로젝트처럼 무조건

components/
hooks/
services/
utils/
types/
pages/


구조를 적용하지 않는다.

대신 다음과 같은 RL 프로젝트의 실제 책임과 실행 흐름을 기준으로 구조를 판단한다.

data
  ↓
preprocessing
  ↓
environment / simulator
  ↓
training
  ↓
checkpoint
  ↓
evaluation
  ↓
submission


프로젝트에 실제로 존재하는 역할에 따라 적절한 구조를 만들어라.

예를 들어 필요하다면 다음과 같은 구조를 고려할 수 있다.

project/
├── V-Max/                         # 서드파티 프레임워크
│   ├── vmax/                     # 수정 금지
│   └── scripts/                  # 우리가 추가한 스크립트는 별도 검토
│
├── dxchallenge_planning_eval/    # 공식 평가 프로그램, 수정 금지
│
├── scripts/                      # 데이터/실험/평가용 실행 스크립트
│
├── training/                     # 학습 관련 코드
│   ├── train.py
│   ├── sac_trainer.py
│   └── ...
│
├── simulation/                   # simulator 생성/환경 관련 코드
│   └── sim_factory.py
│
├── evaluation/                   # 평가 관련 코드
│
├── data/                         # 데이터 관련 처리
│
├── configs/                      # 학습/실험 설정
│
└── README.md


하지만 이것 역시 예시일 뿐이다.

실제 프로젝트를 분석한 후 필요한 폴더만 생성한다.

코드가 적은데 억지로 여러 폴더를 만드는 것은 금지한다.

5. 파일 이동 원칙

파일을 이동하기 전에 반드시 해당 파일의 역할을 분석한다.

예를 들어:

train.py


가 단순한 entry point인지, 실제 학습 로직까지 포함하고 있는지 확인한다.

또한:

sac_trainer.py


가 V-Max의 기존 SAC 구현을 패치한 것인지, 우리가 새로 작성한 trainer인지 확인한다.

파일의 성격이 불명확한 경우 임의로 이동하지 않는다.

6. JAX/RL 코드 특성상 특히 조심할 것

다음 항목은 구조를 정리한다는 이유로 변경하지 않는다.

jax.jit
jax.vmap
jax.pmap
jax.lax
pytree 구조
Flax parameter/state 구조
PRNG key 전달 방식
device placement
batch dimension
array shape
dtype
static/dynamic argument 설정
checkpoint serialization
optimizer state
replay buffer 구조
environment state
simulator state
gradient 계산
loss 계산
update frequency
RNG splitting

특히 import 위치 변경이나 함수 분리 과정에서

JAX tracing
static argument
pytree registration
closure
shape
dtype
device


등의 동작이 달라질 가능성이 있으므로 매우 보수적으로 작업한다.

동작을 변경할 가능성이 있는 리팩토링은 하지 않는다.

7. 학습 결과 보존

리팩토링 전후로 다음 동작이 동일해야 한다.

학습 시작 방식
command line arguments
config
environment 생성
simulator 생성
random seed 처리
checkpoint 저장/로드
evaluation
데이터 처리
output 파일 형식

특히 동일한 seed를 사용할 경우 가능한 범위에서 기존과 동일한 학습 동작을 유지해야 한다.

8. 스크립트 정리

이번 프로젝트에서 추가한 스크립트들을 먼저 분류한다.

예:

score_scenarios.py
sample_shards.py
split_hard_easy_pools.py
shrink_roadgraph.py


각 스크립트가 다음 중 어디에 해당하는지 판단한다.

데이터 전처리
데이터 샘플링
데이터 분할
evaluation
scoring
visualization
학습 실행
디버깅
일회성 실험
submission 관련

그 결과에 따라 적절한 위치로 정리한다.

단, 기존 대회 프레임워크에서 요구하는 실행 경로가 있다면 경로를 변경하지 않는다.

경로 변경이 필요한 경우 실행 script 또는 symlink 등의 호환 방법을 먼저 검토한다.

9. 중복 코드 정리

중복 코드가 발견되면 무조건 공통 함수로 만들지 않는다.

다음 조건을 만족할 때만 공통화한다.

실제로 동일한 의미의 로직이며
여러 곳에서 사용되고
공통화했을 때 dependency가 복잡해지지 않고
JAX tracing이나 성능에 악영향이 없으며
코드의 이해도가 오히려 좋아지는 경우

단순히 코드가 비슷하다는 이유만으로 추상화하지 않는다.

RL 프로젝트에서는 과도한 abstraction보다 명확하고 예측 가능한 코드를 우선한다.

10. 파일 삭제

다음 조건을 모두 확인한 경우에만 삭제한다.

프로젝트 전체에서 사용되지 않음
실행 script에서 사용되지 않음
config에서 사용되지 않음
checkpoint/evaluation 과정에서 사용되지 않음
대회 제출 과정에서 필요하지 않음
README/documentation에서 필요한 파일이 아님
향후 실험 재현에 필요한 파일이 아님

확신할 수 없으면 삭제하지 말고 목록으로 보고한다.

11. Git diff를 적극적으로 활용

가능하다면 git history와 diff를 이용해서

원래 제공된 코드
        vs
우리가 수정한 코드


를 구분한다.

특히 V-Max 내부에서 우리가 수정한 파일이 있다면:

git diff
git status


등을 통해 확인한다.

서드파티 원본과 우리가 수정한 부분을 구분하지 못한 상태에서 대규모 리팩토링을 시작하지 않는다.

12. 작업 절차

반드시 다음 순서로 진행한다.

Step 1 — 프로젝트 탐색

먼저 전체 구조를 읽는다.

하지만 모든 서드파티 파일을 상세 분석할 필요는 없다.

다음만 파악한다.

프로젝트 진입점
우리가 추가한 코드
우리가 수정한 코드
학습 실행 흐름
데이터 처리 흐름
evaluation 흐름
submission 흐름
Step 2 — 변경 범위 확정

다음과 같이 분류한다.

[DO NOT TOUCH]
- V-Max/vmax/*
- dxchallenge_planning_eval/*

[REFACTOR]
- 우리가 직접 추가한 파일
- 우리가 직접 수정한 파일

[REVIEW]
- 원본인지 수정본인지 확실하지 않은 파일


REVIEW에 해당하는 파일은 나에게 먼저 보고한다.

Step 3 — 개선된 구조 제안

실제 프로젝트를 분석한 결과를 기반으로

현재 구조
    ↓
제안 구조


를 보여준다.

각 파일을 왜 이동하는지도 간단히 설명한다.

이 단계에서는 아직 파일을 이동하거나 삭제하지 않는다.

Step 4 — 구조 변경

내가 계획을 확인하면 파일 이동을 진행한다.

이때 모든 import/reference를 함께 수정한다.

Step 5 — 코드 리팩토링

구조 변경 후에 필요한 최소한의 코드 리팩토링만 한다.

우선순위:

명확하지 않은 책임 분리
중복 제거
불필요한 코드 제거
이름 개선
import 정리
가독성 개선
Step 6 — 검증

최종적으로 다음을 확인한다.

- import error
- module not found
- circular dependency
- syntax error
- type error
- lint error
- runtime error


가능하면 기존과 동일한 학습 command를 실행해서 최소한 startup 단계가 정상인지 확인한다.

가능한 경우 짧은 smoke test를 수행한다.

단, 전체 RL 학습을 장시간 실행할 필요는 없다.

13. 절대로 하지 말아야 할 것

다음 작업은 금지한다.

V-Max 전체 리팩토링
vmax/agents/ 구조 변경
vmax/simulator/ 구조 변경
dxchallenge_planning_eval 수정
JAX 코드를 PyTorch 등 다른 framework로 변경
학습 알고리즘 변경
hyperparameter 변경
reward 변경
environment 동작 변경
random seed 처리 변경
데이터 생성 방식 변경
대회 제출 형식 변경
필요하지 않은 abstraction 추가
대규모 class/function 구조 변경
동작 확인 없이 파일 삭제
14. 최종 보고

작업이 끝난 후 다음 내용을 보고한다.

최종 구조
...

이동한 파일
old/path/file.py
→
new/path/file.py

수정한 파일

각 파일에 대해 무엇을 변경했는지 설명한다.

삭제한 파일

삭제했다면 이유를 함께 작성한다.

서드파티 코드

다음 영역은 변경하지 않았음을 확인한다.

V-Max/vmax/
dxchallenge_planning_eval/

검증
Import check: PASS / FAIL
Syntax check: PASS / FAIL
Lint: PASS / FAIL / NOT RUN
Test: PASS / FAIL / NOT AVAILABLE
Training smoke test: PASS / FAIL / NOT RUN

주의사항

리팩토링 후에도 수동 확인이 필요한 부분을 작성한다.

최종 원칙

이 작업의 목표는 프로젝트를 새로 설계하는 것이 아니다.

목표는

"대회에서 제공한 서드파티 RL 프레임워크와 공식 평가 프로그램은 최대한 그대로 보존하면서, 우리가 추가한 학습/실험/데이터 처리 코드를 찾기 쉽고 유지보수하기 좋은 구조로 정리하는 것"

이다.

따라서 변경을 많이 하는 것보다 안전하게 필요한 부분만 변경하는 것을 우선한다.

확신이 없는 변경은 하지 말고 먼저 보고한다.