# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Doosan E0509 Cube Stack 강화학습 환경 v5
- 6축 로봇 + RH-P12-RN 그리퍼
- 그리퍼 1D 제어 (실제 로봇과 동일)
- 4개 관절 동기화 제어
- 그리퍼 열고 닫기: 규칙 기반으로 확실하게 동작
- 단계별 보상 시스템 (Staged Reward)
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.utils.math import subtract_frame_transforms
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, subtract_frame_transforms
import os

##############################################################################
# E0509 로봇 설정
##############################################################################

E0509_USD_PATH = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Robots/Doosan/E0509_rl_basic.usd")

E0509_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=E0509_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 1.5708,
            "joint_4": 0.0,
            "joint_5": 1.5708,
            "joint_6": 0.0,
            "rh_l1": 0.0,
            "rh_l2": 0.0,
            "rh_r1": 0.0,
            "rh_r2": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-3]"],
            velocity_limit=87.0,
            effort_limit=87.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[4-6]"],
            velocity_limit=12.0,
            effort_limit=87.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_l1", "rh_r1"],
            velocity_limit=2.0,
            effort_limit=100.0,
            stiffness=1000.0,
            damping=50.0,
        ),
    },
)


@configclass
class E0509CubeStackEnvCfg(DirectRLEnvCfg):
    """E0509 Cube Stack 환경 설정 v5"""

    episode_length_s = 12.0
    decimation = 2
    action_scale = 1.5
    ee_pos_action_scale = 0.20
    ee_rot_action_scale = 1.00

    osc_kp_task = [250.0, 250.0, 250.0, 80.0, 80.0, 80.0]

    action_space = 7
    observation_space = 22
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
    )

    table_height: float = 1.0
    table_thickness: float = 0.05
    table_stand_height: float = 0.03

    robot_cfg: ArticulationCfg = E0509_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.425, 0.0, 1.055),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.5708,
                "joint_4": 0.0,
                "joint_5": 1.5708,
                "joint_6": 0.0,
                "rh_l1": 0.0,
                "rh_l2": 0.0,
                "rh_r1": 0.0,
                "rh_r2": 0.0,
            },
        ),
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    table_stand_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TableStandA",
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.22, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.425, 0.0, 1.04)),
    )

    cubeA_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeA",
        spawn=sim_utils.CuboidCfg(
            size=(0.085, 0.12, 0.035),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.27, 0.17)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0425)),
    )

    cubeB_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeB",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # 물리 영향 제거
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 1.05)),
    )

    cubeA_size: float = 0.045  # 4.5cm (기존 3.5cm에서 +1cm)
    cubeA_width: float = 0.085
    cubeB_size: float = 0.05

    start_position_noise: float = 0.15
    start_rotation_noise: float = 0.5


class E0509CubeStackEnv(DirectRLEnv):
    """
    E0509 로봇 Cube Stack 환경 v5
    """

    cfg: E0509CubeStackEnvCfg

    def __init__(self, cfg: E0509CubeStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        self.robot_dof_lower_limits[1] = -0.5236
        self.robot_dof_upper_limits[1] = 1.1345

        self.robot_dof_targets = torch.zeros((self.num_envs, 10), device=self.device)

        self.hand_link_idx = self._robot.find_bodies("tool0")[0][0]
        self.finger_l_idx = self._robot.find_bodies("rh_p12_rn_l2")[0][0]
        self.finger_r_idx = self._robot.find_bodies("rh_p12_rn_r2")[0][0]

        self.arm_joint_ids = self._robot.find_joints(["joint_[1-6]"])[0]
        self.gripper_joint_ids = self._robot.find_joints(["rh_l1", "rh_r1"])[0]

        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="variable_kp",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_damping_ratio_task=1.0,
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            nullspace_control="none",
        )
        self.osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.device)

        self.task_frame_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.task_frame_pose_b[:, 3] = 1.0

        self.ee_goal_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_goal_quat_b[:, 0] = 1.0

        kp = torch.tensor(self.cfg.osc_kp_task, device=self.device, dtype=torch.float32)
        self.osc_kp_task = kp.unsqueeze(0).repeat(self.num_envs, 1)

        self.joint_centers = torch.mean(self._robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :], dim=-1)

        self.control_dt = self.cfg.sim.dt * self.cfg.decimation

        self.should_close_gripper = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.arm_joint_efforts = torch.zeros((self.num_envs, len(self.arm_joint_ids)), device=self.device)

        self._sync_ee_goal_with_current()

        self.table_height = self.cfg.table_height + self.cfg.table_thickness / 2.0
        self.cubeA_size = self.cfg.cubeA_size
        self.cubeA_width = self.cfg.cubeA_width
        self.cubeB_size = self.cfg.cubeB_size

        self.actions = torch.zeros((self.num_envs, 7), device=self.device)

        self.cubeA_initial_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cube_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cube_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.should_close_gripper = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.max_gripper_open = self.robot_dof_upper_limits[6] + self.robot_dof_upper_limits[7]

        # ✅ 기준(원래) 쿼터니안: 리셋에서 항상 [1,0,0,0]이므로 그대로 사용
        self.cubeA_ref_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)

        # Cube B 높이 추적 (랜덤화된 높이 저장)
        self.cubeB_height = torch.zeros(self.num_envs, device=self.device)
        self.cubeB_height[:] = self.table_height + self.cubeB_size / 2.0  # 기본값

        print(f"=== E0509 Cube Stack Env v6 ===")
        print(f"Action space: {self.cfg.action_space} (6 arm + 1 gripper)")
        print(f"Gripper: Rule-based open/close, RL controls arm positioning")

    def _quat_mul(self, q, r):
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = r.unbind(-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def _quat_from_rotvec(self, rotvec):
        theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp(min=1e-9)
        axis = rotvec / theta
        half = 0.5 * theta
        w = torch.cos(half)
        xyz = axis * torch.sin(half)
        return torch.cat([w, xyz], dim=-1)

    def _sync_ee_goal_with_current(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        root_pos_w = self._robot.data.root_pos_w[env_ids]
        root_quat_w = self._robot.data.root_quat_w[env_ids]
        ee_pos_w = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        ee_quat_w = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        self.ee_goal_pos_b[env_ids] = ee_pos_b
        self.ee_goal_quat_b[env_ids] = ee_quat_b

    def _compute_osc_states(self):
        ee_jacobi_idx = self.hand_link_idx - 1
        jacobian_w = self._robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, self.arm_joint_ids]
        mass_matrix = (
            self._robot.root_physx_view.get_generalized_mass_matrices()[:, self.arm_joint_ids, :][:, :, self.arm_joint_ids]
        )
        gravity = self._robot.root_physx_view.get_gravity_compensation_forces()[:, self.arm_joint_ids]

        jacobian_b = jacobian_w.clone()
        root_rot_matrix = matrix_from_quat(quat_inv(self._robot.data.root_quat_w))
        jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self.hand_link_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        ee_vel_w = self._robot.data.body_vel_w[:, self.hand_link_idx, :]
        root_vel_w = self._robot.data.root_vel_w
        relative_vel_w = ee_vel_w - root_vel_w
        ee_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, relative_vel_w[:, 0:3])
        ee_ang_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, relative_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

        ee_force_b = torch.zeros((self.num_envs, 3), device=self.device)

        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]

        return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, ee_force_b, joint_pos, joint_vel

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._table = RigidObject(self.cfg.table_cfg)
        self._table_stand = RigidObject(self.cfg.table_stand_cfg)
        self._cubeA = RigidObject(self.cfg.cubeA_cfg)
        self._cubeB = RigidObject(self.cfg.cubeB_cfg)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["table_stand"] = self._table_stand
        self.scene.rigid_objects["cubeA"] = self._cubeA
        self.scene.rigid_objects["cubeB"] = self._cubeB

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        clone = self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)

        self._update_gripper_rule()

        a_pos = self.actions[:, 0:3]
        a_rot = self.actions[:, 3:6]

        dpos = (self.cfg.ee_pos_action_scale * self.control_dt) * a_pos
        drot = (self.cfg.ee_rot_action_scale * self.control_dt) * a_rot
        dquat = self._quat_from_rotvec(drot)

        self.ee_goal_pos_b = self.ee_goal_pos_b + dpos
        self.ee_goal_quat_b = self._quat_mul(dquat, self.ee_goal_quat_b)
        self.ee_goal_quat_b = self.ee_goal_quat_b / torch.linalg.norm(self.ee_goal_quat_b, dim=-1, keepdim=True).clamp(min=1e-9)

        osc_command = torch.cat([self.ee_goal_pos_b, self.ee_goal_quat_b, self.osc_kp_task], dim=-1)

        jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, ee_force_b, joint_pos, joint_vel = self._compute_osc_states()

        self.osc.set_command(
            command=osc_command,
            current_ee_pose_b=ee_pose_b,
            current_task_frame_pose_b=self.task_frame_pose_b,
        )

        self.arm_joint_efforts = self.osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=self.joint_centers,
        )

        # gripper: rule-based position target
        grip_ratio = 0.95
        open_l1 = self.robot_dof_lower_limits[6].item()
        close_l1 = self.robot_dof_upper_limits[6].item()
        target_l1 = open_l1 + grip_ratio * (close_l1 - open_l1)

        open_r1 = self.robot_dof_lower_limits[7].item()
        close_r1 = self.robot_dof_upper_limits[7].item()
        target_r1 = open_r1 + grip_ratio * (close_r1 - open_r1)

        # ✅ should_close_gripper == False이면 open_l1/open_r1로 "완전 Open"
        self.robot_dof_targets[:, 6] = torch.where(self.should_close_gripper, target_l1, open_l1)
        self.robot_dof_targets[:, 7] = torch.where(self.should_close_gripper, target_r1, open_r1)

    def _update_gripper_rule(self):
        env_origins = self.scene.env_origins
        finger_l_pos = self._robot.data.body_pos_w[:, self.finger_l_idx, :] - env_origins
        finger_r_pos = self._robot.data.body_pos_w[:, self.finger_r_idx, :] - env_origins
        cubeA_pos = self._cubeA.data.root_pos_w - env_origins
        cubeB_pos = self._cubeB.data.root_pos_w - env_origins

        gripper_center = (finger_l_pos + finger_r_pos) / 2.0
        cube_xy = cubeA_pos[:, :2]
        gripper_center_xy = gripper_center[:, :2]
        xy_dist_to_cube = torch.norm(cube_xy - gripper_center_xy, dim=-1)
        
        # 그리퍼 중심이 큐브 중심과 XY 3cm 이내, 높이는 6cm 이내
        height_diff = torch.abs(gripper_center[:, 2] - cubeA_pos[:, 2])
        box_between_fingers = (xy_dist_to_cube < 0.03) & (height_diff < 0.06)

        # ✅ 릴리즈 래치: "팔렛트 위 정렬 + 목표 높이" 만족하면 강제로 OPEN
        # (방향 조건은 Open 트리거에서 제거: 요청사항 반영)
        cubeA_height = cubeA_pos[:, 2] - self.table_height
        target_stack_height = self.cubeB_size + self.cubeA_size / 2.0
        height_ok = torch.abs(cubeA_height - target_stack_height) < 0.03

        xy_dist_to_cubeB = torch.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], dim=-1)
        aligned_ok = xy_dist_to_cubeB < 0.05

        release_cond = self.cube_lifted & aligned_ok & height_ok
        self.cube_released |= release_cond

        # 릴리즈 이후에는 절대 다시 닫지 않음
        self.should_close_gripper = (~self.cube_released) & (box_between_fingers | self.cube_grasped)

    def _apply_action(self):
        self._robot.set_joint_effort_target(self.arm_joint_efforts, joint_ids=self.arm_joint_ids)

        gripper_targets = self.robot_dof_targets[:, [6, 7]]
        self._robot.set_joint_position_target(gripper_targets, joint_ids=self.gripper_joint_ids)

        self._robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        env_origins = self.scene.env_origins

        eef_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx, :]
        eef_quat_w = self._robot.data.body_quat_w[:, self.hand_link_idx, :]
        finger_l_pos_w = self._robot.data.body_pos_w[:, self.finger_l_idx, :]
        finger_r_pos_w = self._robot.data.body_pos_w[:, self.finger_r_idx, :]

        cubeA_pos_w = self._cubeA.data.root_pos_w
        cubeA_quat = self._cubeA.data.root_quat_w
        cubeB_pos_w = self._cubeB.data.root_pos_w

        eef_pos = eef_pos_w - env_origins
        finger_l_pos = finger_l_pos_w - env_origins
        finger_r_pos = finger_r_pos_w - env_origins
        cubeA_pos = cubeA_pos_w - env_origins
        cubeB_pos = cubeB_pos_w - env_origins

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        eef_pos_b, eef_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, eef_pos_w, eef_quat_w)

        goal_pos_error_b = self.ee_goal_pos_b - eef_pos_b

        gripper_pos = self._robot.data.joint_pos[:, 6:8]
        gripper_closed_amount = gripper_pos[:, 0] + gripper_pos[:, 1]
        gripper_open_normalized = 1.0 - (gripper_closed_amount / self.max_gripper_open)

        finger_distance = torch.norm(finger_l_pos - finger_r_pos, dim=-1, keepdim=True)

        obs = torch.cat(
            [
                cubeA_pos,  # 3
                cubeA_quat,  # 4
                cubeB_pos,  # 3
                eef_pos,  # 3
                eef_quat_w,  # 4
                gripper_open_normalized.unsqueeze(-1),  # 1
                goal_pos_error_b,  # 3
                finger_distance,  # 1
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        env_origins = self.scene.env_origins

        eef_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx, :]
        eef_quat = self._robot.data.body_quat_w[:, self.hand_link_idx, :]
        finger_l_pos_w = self._robot.data.body_pos_w[:, self.finger_l_idx, :]
        finger_r_pos_w = self._robot.data.body_pos_w[:, self.finger_r_idx, :]
        cubeA_pos_w = self._cubeA.data.root_pos_w
        cubeB_pos_w = self._cubeB.data.root_pos_w

        eef_pos = eef_pos_w - env_origins
        finger_l_pos = finger_l_pos_w - env_origins
        finger_r_pos = finger_r_pos_w - env_origins
        cubeA_pos = cubeA_pos_w - env_origins
        cubeB_pos = cubeB_pos_w - env_origins

        gripper_pos = self._robot.data.joint_pos[:, 6:10]
        gripper_closed_amount = gripper_pos[:, 0] + gripper_pos[:, 1]
        gripper_open = self.max_gripper_open - gripper_closed_amount

        # Approach 목표: 큐브 중심 (옆면을 잡아야 함)
        eef_to_cubeA = torch.norm(cubeA_pos - eef_pos, dim=-1)
        dl_to_cubeA = torch.norm(cubeA_pos - finger_l_pos, dim=-1)
        dr_to_cubeA = torch.norm(cubeA_pos - finger_r_pos, dim=-1)
        d_fingers_avg = (dl_to_cubeA + dr_to_cubeA) / 2.0

        finger_distance = torch.norm(finger_l_pos - finger_r_pos, dim=-1)

        gripper_center = (finger_l_pos + finger_r_pos) / 2.0
        gripper_center_to_cube = torch.norm(cubeA_pos - gripper_center, dim=-1)

        cubeA_height = cubeA_pos[:, 2] - self.table_height
        target_stack_height = self.cubeB_size + self.cubeA_size / 2.0

        qw, qx, qy, qz = eef_quat[:, 0], eef_quat[:, 1], eef_quat[:, 2], eef_quat[:, 3]
        gripper_z_component = 2.0 * (qx * qz + qw * qy)
        local_x_world_z = 2.0 * (qx * qz - qw * qy)

        cube_xy = cubeA_pos[:, :2]
        gripper_center_xy = gripper_center[:, :2]
        xy_dist_to_cube = torch.norm(cube_xy - gripper_center_xy, dim=-1)
        
        # 그리퍼 중심이 큐브 중심과 XY 3cm 이내, 높이는 6cm 이내
        height_diff_gripper_cube = torch.abs(gripper_center[:, 2] - cubeA_pos[:, 2])
        box_between_fingers = (xy_dist_to_cube < 0.03) & (height_diff_gripper_cube < 0.06)

        gripper_closed = self.should_close_gripper

        cube_near_gripper = gripper_center_to_cube < 0.05
        cube_grasped_now = gripper_closed & box_between_fingers & cube_near_gripper
        self.cube_grasped = self.cube_grasped | cube_grasped_now

        lifted_height = cubeA_height - self.cubeA_size / 2.0
        currently_holding = gripper_closed & (gripper_center_to_cube < 0.05)

        # v5 방식: 단순히 5cm → 8cm (+3cm)
        cube_lifted = self.cube_grasped & (lifted_height > 0.08)  # 8cm (기존 5cm + 3cm)
        self.cube_lifted = self.cube_lifted | cube_lifted

        xy_dist_to_cubeB = torch.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], dim=-1)
        aligned_over_cubeB = self.cube_lifted & (xy_dist_to_cubeB < 0.05)

        # ==========================
        # ✅ 큐브 쿼터니안 유지(정렬) 보상
        # - 리셋 시 cubeA_quat = [1,0,0,0] 이므로 이를 기준으로 "원래 자세 유지" 유도
        # - qdot = |<q, q_ref>| (0~1)
        # ==========================
        cubeA_quat = self._cubeA.data.root_quat_w  # (N,4) (w,x,y,z) assumed
        q_ref = self.cubeA_ref_quat.expand_as(cubeA_quat)
        qdot = torch.abs(torch.sum(cubeA_quat * q_ref, dim=-1)).clamp(0.0, 1.0)  # 1이면 동일

        # 스택 성공에 포함할 정도의 기준(원하면 조절)
        quat_ok = qdot > 0.98

        height_diff = torch.abs(cubeA_height - target_stack_height)

        # ✅ "놓기(강제 open 발동)"까지 포함한 성공 판정
        stack_success = aligned_over_cubeB & (height_diff < 0.02) & quat_ok & self.cube_released

        # ========== Stage 1: 접근 ==========
        approach_reward = 1.0 - torch.tanh(5.0 * d_fingers_avg)
        positioning_bonus = box_between_fingers.float() * 3.0

        # ========== Stage 2: 그립 (v5 방식) ==========
        grip_reward = cube_grasped_now.float() * 5.0
        stable_grip_reward = currently_holding.float() * (~self.cube_lifted).float() * 5.0

        # ========== Stage 3: 리프트 (v5 방식) ==========
        # 정렬 전까지만 lift 보상 (정렬 후에는 내려가야 함)
        lift_reward = currently_holding.float() * (~aligned_over_cubeB).float() * torch.tanh(10.0 * torch.clamp(lifted_height, min=0.0)) * 5.0
        lift_success_bonus = cube_lifted.float() * 10.0

        # ========== Stage 4: 이동 (v5 방식) ==========
        move_reward = self.cube_lifted.float() * (1.0 - torch.tanh(4.0 * xy_dist_to_cubeB)) * 12.0
        # 높이 유지는 정렬 전까지만 (정렬 후에는 내려가야 함)
        height_maintain_reward = self.cube_lifted.float() * (~aligned_over_cubeB).float() * (lifted_height > 0.03).float() * 0.5
        align_bonus = self.cube_lifted.float() * aligned_over_cubeB.float() * 8.0

        # ========== Stage 5: 놓기/스택 ==========
        # target_stack_height = cubeB_size + cubeA_size/2 (CubeA 중심이 CubeB 위)
        height_to_target = torch.abs(cubeA_height - target_stack_height)

        place_height_reward = aligned_over_cubeB.float() * (1.0 - torch.tanh(15.0 * height_to_target)) * 6.0

        # 쿼터니안 유지 보상(정렬될수록 1에 가까움)
        quat_reward = aligned_over_cubeB.float() * torch.clamp((qdot - 0.95) / (1.0 - 0.95), 0.0, 1.0) * 6.0

        # 놓기 준비 완료 보너스: 정렬 + 높이 + 자세(쿼터니안)
        stack_ready_bonus = (
            self.cube_lifted.float()
            * aligned_over_cubeB.float()
            * (height_to_target < 0.02).float()
            * quat_ok.float()
            * 6.0
        )
        
        # ★ Open 유도 보상: 정렬 + 높이 맞으면 그리퍼 Open에 보상
        ready_to_release = aligned_over_cubeB & (height_to_target < 0.03) & quat_ok
        open_reward = ready_to_release.float() * gripper_open * 20.0  # 그리퍼 열수록 보상
        
        # ★ Release 성공 보상 (cube_released 트리거 시)
        release_bonus = self.cube_released.float() * 25.0

        # 최종 성공 보상
        stack_success_bonus = stack_success.float() * 50.0  # 40 → 50

        # ========== 패널티 ==========
        flipped_penalty = (gripper_z_component > 0.0).float() * 5.0
        cos_down = (-local_x_world_z).clamp(-1.0, 1.0)
        tilted_penalty = (1.0 - cos_down).clamp(min=0.0) * 3.0  # 기울어짐 패널티 강화

        cube_dropped = self.cube_grasped & ~currently_holding & (lifted_height < 0.01) & ~stack_success
        drop_penalty = cube_dropped.float() * 10.0

        finger_min_height = torch.min(finger_l_pos[:, 2], finger_r_pos[:, 2])
        floor_collision = (finger_min_height < self.table_height + 0.03).float()  # 3cm 마진
        floor_penalty = floor_collision * 5.0

        # 목표 높이 근처인데 계속 잡고 있으면 패널티 (강화)
        hold_penalty_near_target = (
            self.cube_lifted.float()
            * aligned_over_cubeB.float()
            * (height_to_target < 0.03).float()
            * currently_holding.float()
            * 10.0  # 4 → 10
        )

        a_pos = self.actions[:, 0:3]
        a_rot = self.actions[:, 3:6]
        action_penalty = 0.002 * torch.sum(a_pos ** 2, dim=-1) + 0.0005 * torch.sum(a_rot ** 2, dim=-1)

        rewards = (
            2.0 * approach_reward
            + positioning_bonus
            + grip_reward
            + stable_grip_reward
            + lift_reward
            + lift_success_bonus
            + move_reward
            + height_maintain_reward
            + align_bonus
            + place_height_reward
            + quat_reward
            + stack_ready_bonus
            + open_reward
            + release_bonus
            + stack_success_bonus
            - flipped_penalty
            - tilted_penalty
            - drop_penalty
            - floor_penalty
            - action_penalty
            - hold_penalty_near_target
        )

        # TensorBoard용 전체 로그
        self.extras["log"] = {
            "reward/stage1_approach": (2.0 * approach_reward).mean().item(),
            "reward/stage1_positioning": positioning_bonus.mean().item(),
            "reward/stage2_grip": grip_reward.mean().item(),
            "reward/stage2_stable_grip": stable_grip_reward.mean().item(),
            "reward/stage3_lift": lift_reward.mean().item(),
            "reward/stage3_lift_success": lift_success_bonus.mean().item(),
            "reward/stage4_move": move_reward.mean().item(),
            "reward/stage4_height_maintain": height_maintain_reward.mean().item(),
            "reward/stage4_align_bonus": align_bonus.mean().item(),
            "reward/stage5_place_height": place_height_reward.mean().item(),
            "reward/stage5_quat": quat_reward.mean().item(),
            "reward/stage5_ready": stack_ready_bonus.mean().item(),
            "reward/stage5_open": open_reward.mean().item(),
            "reward/stage5_release": release_bonus.mean().item(),
            "reward/stage5_stack_success": stack_success_bonus.mean().item(),
            "penalty/flipped": flipped_penalty.mean().item(),
            "penalty/tilted": tilted_penalty.mean().item(),
            "penalty/drop": drop_penalty.mean().item(),
            "penalty/floor": floor_penalty.mean().item(),
            "penalty/holdinPallet": hold_penalty_near_target.mean().item(),
            "state/d_fingers_avg": d_fingers_avg.mean().item(),
            "state/gripper_open": gripper_open.mean().item(),
            "state/finger_distance": finger_distance.mean().item(),
            "state/xy_dist_to_cube": xy_dist_to_cube.mean().item(),
            "state/cubeA_height": cubeA_height.mean().item(),
            "state/lifted_height": lifted_height.mean().item(),
            "state/xy_dist_to_cubeB": xy_dist_to_cubeB.mean().item(),
            "state/eef_to_cubeA": eef_to_cubeA.mean().item(),
            "state/qdot": qdot.mean().item(),
            "progress/box_between_fingers": box_between_fingers.sum().item(),
            "progress/should_close_gripper": self.should_close_gripper.sum().item(),
            "progress/gripper_closed": gripper_closed.sum().item(),
            "progress/cube_grasped_now": cube_grasped_now.sum().item(),
            "progress/currently_holding": currently_holding.sum().item(),
            "progress/cube_grasped_count": self.cube_grasped.sum().item(),
            "progress/cube_lifted_count": self.cube_lifted.sum().item(),
            "progress/aligned_count": aligned_over_cubeB.sum().item(),
            "progress/cube_released": self.cube_released.sum().item(),
            "progress/stack_success_count": stack_success.sum().item(),
            "debug/xy_dist_to_cube": xy_dist_to_cube.mean().item(),
            "debug/height_diff_gripper_cube": height_diff_gripper_cube.mean().item(),
        }
        
        # 터미널 출력용 로그 (간소화)
        self.extras["episode"] = {
            "reward/stage1_approach": (2.0 * approach_reward).mean().item(),
            "reward/stage2_grip": grip_reward.mean().item(),
            "reward/stage3_lift": lift_reward.mean().item(),
            "reward/stage4_move": move_reward.mean().item(),
            "reward/stage5_stack_success": stack_success_bonus.mean().item(),
            "penalty/flipped": flipped_penalty.mean().item(),
            "penalty/tilted": tilted_penalty.mean().item(),
            "progress/cube_grasped_count": self.cube_grasped.sum().item(),
            "progress/cube_lifted_count": self.cube_lifted.sum().item(),
            "progress/stack_success_count": stack_success.sum().item(),
        }

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_origins = self.scene.env_origins

        eef_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx, :]
        eef_quat = self._robot.data.body_quat_w[:, self.hand_link_idx, :]
        cubeA_pos_w = self._cubeA.data.root_pos_w
        cubeB_pos_w = self._cubeB.data.root_pos_w

        finger_l_pos_w = self._robot.data.body_pos_w[:, self.finger_l_idx, :]
        finger_r_pos_w = self._robot.data.body_pos_w[:, self.finger_r_idx, :]

        eef_pos = eef_pos_w - env_origins
        cubeA_pos = cubeA_pos_w - env_origins
        cubeB_pos = cubeB_pos_w - env_origins

        finger_l_pos = finger_l_pos_w - env_origins
        finger_r_pos = finger_r_pos_w - env_origins
        finger_min_height = torch.min(finger_l_pos[:, 2], finger_r_pos[:, 2])

        cubeA_height = cubeA_pos[:, 2] - self.table_height
        target_height = self.cubeB_size + self.cubeA_size / 2.0

        xy_dist = torch.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], dim=-1)
        height_diff = torch.abs(cubeA_height - target_height)
        eef_to_cubeA = torch.norm(cubeA_pos - eef_pos, dim=-1)

        # ✅ 성공 조건: 정렬 + 높이 + 쿼터니안 유지 + 릴리즈(강제 오픈 발동)
        cubeA_quat = self._cubeA.data.root_quat_w
        q_ref = self.cubeA_ref_quat.expand_as(cubeA_quat)
        qdot = torch.abs(torch.sum(cubeA_quat * q_ref, dim=-1)).clamp(0.0, 1.0)
        quat_ok = qdot > 0.98

        stack_success = (xy_dist < 0.03) & (height_diff < 0.02) & quat_ok & self.cube_released

        cubeA_fallen = cubeA_pos[:, 2] < (self.table_height - 0.1)
        cubeB_fallen = cubeB_pos[:, 2] < (self.table_height - 0.1)
        finger_floor_collision = finger_min_height < (self.table_height + 0.03)  # 3cm 마진

        qw, qx, qy, qz = eef_quat[:, 0], eef_quat[:, 1], eef_quat[:, 2], eef_quat[:, 3]
        gripper_z = 2.0 * (qx * qz + qw * qy)
        gripper_flipped = gripper_z > 0.3  # 더 엄격하게 (0.5 → 0.3)

        eef_dist = torch.norm(eef_pos[:, :2] - torch.tensor([-0.45, 0.0], device=self.device), dim=-1)
        workspace_violation = eef_dist > 1.0

        terminated = stack_success | cubeA_fallen | cubeB_fallen | finger_floor_collision | gripper_flipped | workspace_violation
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("log", {})
        self.extras["log"].update(
            {
                "done/stack_success": stack_success.sum().item(),
                "done/cubeA_fallen": cubeA_fallen.sum().item(),
                "done/cubeB_fallen": cubeB_fallen.sum().item(),
                "done/floor_collision": finger_floor_collision.sum().item(),
                "done/gripper_flipped": gripper_flipped.sum().item(),
                "done/workspace_violation": workspace_violation.sum().item(),
                "done/timeout": (self.episode_length_buf >= self.max_episode_length - 1).sum().item(),
            }
        )

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += 0.1 * (torch.rand_like(joint_pos) - 0.5)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        joint_pos[:, 6:10] = self.robot_dof_lower_limits[6:10]
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        if hasattr(self, "arm_joint_efforts"):
            self.arm_joint_efforts[env_ids] = 0.0

        self._reset_cubes(env_ids)

        self.cube_grasped[env_ids] = False
        self.cube_lifted[env_ids] = False
        self.cube_released[env_ids] = False
        self.should_close_gripper[env_ids] = False

        self._robot.reset(env_ids)
        self._robot.update(self.cfg.sim.dt)

        self._sync_ee_goal_with_current(env_ids)

        self.osc.reset()

        super()._reset_idx(env_ids)

    def _reset_cubes(self, env_ids: Sequence[int]):
        env_origins = self.scene.env_origins[env_ids]
        num_resets = len(env_ids)

        # Cube B 위치 랜덤화: XY ±5cm, Z +0~5cm
        cubeB_pos_local = torch.zeros((num_resets, 3), device=self.device)
        cubeB_pos_local[:, 0] = 0.23 + 0.05 * torch.rand(num_resets, device=self.device)  # 0.23 ~ 0.28 (0~5cm)
        cubeB_pos_local[:, 1] = 0.05 * (2.0 * torch.rand(num_resets, device=self.device) - 1.0)  # -0.05 ~ 0.05
        cubeB_z_offset = 0.05 * torch.rand(num_resets, device=self.device)  # 0 ~ 0.05
        cubeB_pos_local[:, 2] = self.table_height + self.cubeB_size / 2.0 + cubeB_z_offset
        cubeB_pos = cubeB_pos_local + env_origins
        cubeB_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)

        self._cubeB.write_root_pose_to_sim(torch.cat([cubeB_pos, cubeB_quat], dim=-1), env_ids)
        self._cubeB.write_root_velocity_to_sim(torch.zeros((num_resets, 6), device=self.device), env_ids)

        # Cube B 높이 저장 (lift 보상 계산용)
        self.cubeB_height[env_ids] = cubeB_pos_local[:, 2]

        xmin, xmax = -0.13, 0.06
        ymin, ymax = -0.15, 0.15
        cubeA_pos_local = torch.zeros((num_resets, 3), device=self.device)
        cubeA_pos_local[:, 0] = xmin + (xmax - xmin) * torch.rand(num_resets, device=self.device)
        cubeA_pos_local[:, 1] = ymin + (ymax - ymin) * torch.rand(num_resets, device=self.device)
        cubeA_pos_local[:, 2] = self.table_height + self.cubeA_size / 2.0
        cubeA_pos = cubeA_pos_local + env_origins
        cubeA_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)

        self._cubeA.write_root_pose_to_sim(torch.cat([cubeA_pos, cubeA_quat], dim=-1), env_ids)
        self._cubeA.write_root_velocity_to_sim(torch.zeros((num_resets, 6), device=self.device), env_ids)

        self.cubeA_initial_pos[env_ids] = cubeA_pos_local
