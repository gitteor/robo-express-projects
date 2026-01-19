# Robo Express

Doosan E0509 로봇팔과 RH-P12-RN 그리퍼를 활용한 강화학습 기반 물류 자동화 시스템

## 프로젝트 개요

IsaacLab 시뮬레이션 환경에서 박스 스태킹 작업을 강화학습으로 학습하고, 실제 Doosan 협동로봇에 배포하는 End-to-End 물류 자동화 프로젝트입니다.

## Demo

<!-- 데모 영상/GIF 추가 예정 -->

## 시스템 구조

```
[Training]                      [Inference & Control]

IsaacLab ──학습──▶ Model ─────▶ Web Server ◀────── RealSense
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
                    ROS2 Robot               Arduino
                  (Doosan E0509)            (컨베이어)
```

## Repository 구조

| 폴더 | 설명 |
|------|------|
| [IsaacLab/](./IsaacLab) | IsaacLab 강화학습 환경 및 학습 코드 |
| [web/](./web) | 웹 서비스 (주문 관리, 실시간 모니터링) |
| [ros2/](./ros2) | ROS2 로봇 제어 패키지 |
| [docs/](./docs) | 프로젝트 문서 |

## 기술 스택

- **Simulation**: NVIDIA Isaac Sim + Isaac Lab
- **RL Framework**: RSL-RL
- **Robot**: Doosan E0509 (6-axis collaborative robot)
- **Gripper**: ROBOTIS RH-P12-RN
- **Vision**: Intel RealSense
- **Backend**: FastAPI, Python
- **Robot Control**: ROS2 Humble
- **Conveyor**: Arduino

## Contributors

| 이름 | 역할 |
|------|------|
| 훈석 | 강화학습, 디지털 트윈 |
| 서린 | Sim2Real, 로봇 제어, 객체인식 |
| 정인 | 강화학습, Sim2Real, 멀티모달 |
| 현우 | ROS2 통신 |

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Doosan Robotics](https://www.doosanrobotics.com/)
- [ROBOTIS](https://www.robotis.com/)
