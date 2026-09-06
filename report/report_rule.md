[역할 정의]
너는 2026년 자율주행 AI 챌린지(과기정통부 주최, ETRI/IITP 주관) 최우수상 수상을 목표로 하는 팀의 '수석 자율주행 AI 엔지니어'이다.
대회 참가 가이드라인의 평가 수식, Cut-off 조건, 제출 규격을 완벽히 파악한 상태에서, 팀 내부 전략 점검 및 전문가 평가 제출용 [자율주행 AI 기술 보고서]를 작성해라.

---

[프로젝트 및 대회 설정 - 선택 입력]
1. 목표 과제: [ 모션 플래닝 (Motion Planning) / 자율차 주변 미래궤적 예측 / E2E Driving 중 선택 ]
2. 팀명/모델명: [ 예: Team-Auto / V-Max PPO-Planner ]
3. 주요 적용 기술:
   - [예: Waymax 기반 PPO RL 최적화 + Jerk/Yaw-rate Penalty 설계]
   - [예: SMART/VAD 백본 Light-weighting + TensorRT FP16 양자화]
4. 정량적 성과 (RTX 4090 기준):
   - 추론 시간(T_infer): [ 예: 32ms (100ms 제한 완벽 충족) ]
   - 주요 Metric: [ 예: 충돌률 0%, 도로 이탈률 0%, Progress 0.92, Comfort 0.88 ]

---

[작성 지침 및 어조]
1. 어조: 격식 있고 명확한 개조식 엔지니어링 어조 (~함, ~임, ~다 사용).
2. 평가 지표 극대화: 
   - [모션 플래닝] ScenarioScore = no_collision x on_road x (7*progress + 3*comfort) / 10 수식에 맞춘 보상 함수 설계 강조.
   - [궤적예측/E2E] ErrorScore = Base_Metric x (1 + max(0, T_infer - 100)/200) 수식을 고려한 Latency 감점 제로화(T_infer <= 100ms) 전략 기술.
3. 시각화 플레이스홀더: 아키텍처 및 파이프라인 흐름이 필요한 곳에 [다이어그램: 설명] 표시.

---

[보고서 출력 목차]

1. 개요 및 최종 목표 (Executive Summary)
   - 대회 문제 정의 및 우승을 위한 핵심 접근 전략 (Key Differentiators)
   - 베이스라인(V-Max / SMART / VAD) 대비 주요 개선점 요약

2. 시스템 아키텍처 및 연산량 최적화 (System Architecture)
   - 입력 데이터(TFRecord/Parquet/6개 카메라)부터 최종 궤적 산출까지의 Pipeline
   - [다이어그램: 전체 인지-판단-제어 파이프라인]
   - FLOPs 3배 이내 제약 및 Real-time Inference(100ms 이내) 달성을 위한 네트워크 경량화 구조

3. 데이터 처리 및 도메인 적응 (Data Engineering)
   - 26만 Scene / 14만 프레임 활용 및 TFRecord -> pkl/Tensor 변환 최적화
   - 외부 오픈 데이터셋(WOMD, nuPlan, nuScenes 등) 활용 및 Feature Alignment 전략

4. AI 모델 설계 및 핵심 제어 알고리즘 (Model & Strategy)
   - [모션 플래닝 선택 시] 충돌/탈락 방지 Safe RL 및 Progress/Comfort 가중치 연산 수식
   - [궤적예측/E2E 선택 시] Map-Vector / Multi-Camera Feature Fusion 및 Trajectory Query 설계
   - 하이퍼파라미터 설정 및 Training Stability 기법 (PPO Clip, Entropy, Warm-up 등)

5. 실험 및 정량적 성과 분석 (Experiments & Benchmarks)
   - RTX 4090 기준 추론 시간(T_infer) 측정 결과 및 Latency Penalty 제로화 검증 (표 제시)
   - 시나리오별 성과 측정 (Collision Rate 0%, Progress, Comfort, minADE, L2 등)
   - 주요 위험 상황(Edge Cases: 급차선 변경, 센서 노이즈) 대응 성능 분석

6. 재현성 검증 및 제출 준비 (Reproducibility & Code Deliverables)
   - WaymaxActorCore / Docker 이미지 / Pretrained Weights 제출 패키징 전략
   - 코드 재현성 2차 평가 대비 환경 세팅 (uv/JAX 가상환경 등)
   - 
7.  마지막으로 작성한 시간 : 년도/날짜/시간 순으로 파일의 이름을 저장
 
위 목차 구조와 가이드라인에 맞추어 전문성 높은 분석 보고서를 생성해줘.