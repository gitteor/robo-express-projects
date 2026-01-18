# Robo Express

Doosan E0509 로봇팔과 RH-P12-RN 그리퍼를 활용한 강화학습 기반 물류 자동화 시스템

## 🎯 프로젝트 개요

IsaacLab 시뮬레이션 환경에서 박스 스태킹 작업을 강화학습으로 학습하고, 실제 Doosan 협동로봇에 배포하는 End-to-End 물류 자동화 프로젝트입니다.

## 🎥 Demo

<!-- 데모 영상/GIF 추가 예정 -->

## 🏗️ 시스템 구조

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Web UI    │────▶│  Inference  │────▶│    ROS2     │
│             │     │   Server    │     │   Control   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  IsaacLab   │     │  Doosan     │
                    │  Training   │     │  E0509      │
                    └─────────────┘     └─────────────┘
```

## 📁 Repository 구조

| 폴더 | 설명 |
|------|------|
| [isaac/](./isaac) | IsaacLab 강화학습 환경 및 학습 코드 |
| [web/](./web) | 웹 서비스 (주문 관리, 실시간 모니터링) |
| [ros2/](./ros2) | ROS2 로봇 제어 패키지 |
| [docs/](./docs) | 프로젝트 문서 |

## 🛠️ 기술 스택

- **Simulation**: NVIDIA Isaac Sim + Isaac Lab
- **RL Framework**: RSL-RL
- **Robot**: Doosan E0509 (6-axis collaborative robot)
- **Gripper**: ROBOTIS RH-P12-RN
- **Backend**: FastAPI, Python
- **Robot Control**: ROS2 Humble

## 🚀 Quick Start

각 폴더의 README를 참조하세요.

```bash
# 시뮬레이션 학습
cd isaac && cat README.md

# 웹 서비스
cd web && cat README.md

# 로봇 제어
cd ros2 && cat README.md
```

## 👥 Contributors

| 이름 | 역할 | GitHub |
|------|------|--------|
| 정인 | RL Environment, Sim2Real | [@gitteor](https://github.com/gitteor) |
| 팀원A | 역할 | [@id](https://github.com/) |
| 팀원B | 역할 | [@id](https://github.com/) |
| 팀원C | 역할 | [@id](https://github.com/) |
| 팀원D | 역할 | [@id](https://github.com/) |

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Doosan Robotics](https://www.doosanrobotics.com/)
- [ROBOTIS](https://www.robotis.com/)
