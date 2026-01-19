# E0509 Cube Stack 환경 설정 가이드

## ★★★ 반드시 수정해야 할 부분 ★★★

### 1단계: USD 파일 경로 설정
`e0509_cube_stack_env.py` 파일 상단에서:
```python
E0509_USD_PATH = "/home/shin/IsaacLab/source/isaaclab_assets/data/Robots/Doosan/E0509_rl_basic.usd"
```
→ 실제 USD 파일 경로로 수정

### 2단계: 관절 이름 확인 및 수정
Isaac Sim에서 USD 파일 열고 관절 이름 확인:
1. Isaac Sim 실행
2. File > Open > E0509_rl_basic.usd
3. Stage 창에서 관절 확인
4. 아래 부분들 수정:

```python
# E0509_CFG의 init_state.joint_pos
joint_pos={
    "joint_1": 0.0,      # ← 실제 이름으로
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": 0.0,
    "joint_5": 0.0,
    "joint_6": 0.0,
    "finger_joint": 0.04,  # ← 그리퍼 관절 이름
    "right_outer_knuckle_joint": 0.04,
}

# actuators의 joint_names_expr
actuators={
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["joint_[1-6]"],  # ← 패턴 수정
        ...
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint", "right_outer_knuckle_joint"],
        ...
    ),
}
```

### 3단계: EEF 링크 이름 수정
```python
# __init__ 메서드에서
self.hand_link_idx = self._robot.find_bodies("link_6")[0][0]
```
→ 실제 End-Effector 링크 이름으로 수정 (예: "tool0", "ee_link", "link6" 등)


## 설치 방법

```bash
# 1. 폴더 복사
cp -r e0509_cube_stack ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/

# 2. USD 파일 복사
cp E0509_rl_basic.usd ~/IsaacLab/source/isaaclab_assets/data/Robots/Doosan/

# 3. direct/__init__.py에 import 추가
# from .e0509_cube_stack import *
```


## 학습 실행

```bash
cd ~/IsaacLab

# 테스트 (GUI)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-E0509-CubeStack-Direct-v0 \
    --num_envs 4

# 본격 학습 (headless)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-E0509-CubeStack-Direct-v0 \
    --num_envs 4096 \
    --headless
```


## Franka vs E0509 차이점

| 항목 | Franka | E0509 |
|------|--------|-------|
| 팔 관절 | 7개 | 6개 |
| 행동 공간 | 9 (7+2) | 8 (6+2) |
| 그리퍼 | panda_finger | 그대로 사용 |


## 문제 해결

### "joint not found" 에러
→ joint_pos의 관절 이름이 USD와 다름. Isaac Sim에서 확인 후 수정.

### "body not found" 에러  
→ hand_link_idx의 링크 이름이 USD와 다름. 수정 필요.

### 로봇이 이상하게 움직임
→ 초기 관절 위치(joint_pos) 값 조정 필요.
