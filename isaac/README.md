# Isaac - 강화학습 환경

Doosan E0509 박스 스태킹 강화학습 환경 (IsaacLab Extension)

## 설치

### 사전 요구사항

- Ubuntu 22.04
- NVIDIA GPU (RTX 3070 이상 권장)
- NVIDIA Isaac Sim 4.x
- Isaac Lab 설치 완료

### Extension 설치

```bash
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .
```

## 사용법

### 학습

```bash
${ISAACLAB_PATH}/isaaclab.sh -p scripts/train.py --task RoboExpress-BoxStacking-v0
```

### 테스트

```bash
${ISAACLAB_PATH}/isaaclab.sh -p scripts/play.py --checkpoint checkpoints/model.pt
```

## 폴더 구조

```
isaac/
├── exts/robo_express/      # IsaacLab extension
│   ├── tasks/              # RL 환경 정의
│   ├── robots/             # 로봇 설정
│   └── assets/             # USD 모델
├── scripts/                # 학습/실행 스크립트
├── configs/                # 학습 설정
└── checkpoints/            # 학습된 모델
```
