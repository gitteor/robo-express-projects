# play2.py 기반 - 성공한 에피소드 CSV 저장
# play2.py와 완전히 동일한 루프, 기록 기능만 추가

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play and record successful episode.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--output_dir", type=str, default="sim2real_output")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
import csv

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


RAD2DEG = 180.0 / np.pi


class EpisodeRecorder:
    """에피소드 데이터 기록"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.timestamps = []
        self.joint_positions = []
        self.joint_velocities = []
        self.gripper_positions = []
        self.gripper_commands = []
        self.rewards = []
    
    def record(self, timestamp, joint_pos, joint_vel, gripper_pos, gripper_cmd, reward):
        self.timestamps.append(timestamp)
        self.joint_positions.append(joint_pos.copy())
        self.joint_velocities.append(joint_vel.copy())
        self.gripper_positions.append(gripper_pos)
        self.gripper_commands.append(gripper_cmd)
        self.rewards.append(reward)
    
    def save_csv(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp']
            header += [f'joint_{i}_pos_deg' for i in range(6)]
            header += [f'joint_{i}_vel_deg' for i in range(6)]
            header += ['gripper_pos', 'gripper_cmd']
            writer.writerow(header)
            
            for i in range(len(self.timestamps)):
                row = [self.timestamps[i]]
                row += list(self.joint_positions[i])
                row += list(self.joint_velocities[i])
                row += [self.gripper_positions[i], int(self.gripper_commands[i])]
                writer.writerow(row)
        print(f"💾 CSV 저장: {filepath}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and record successful episode."""
    
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    
    # 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    
    sim_env = env.unwrapped
    dt = sim_env.step_dt
    
    # === play2.py와 완전히 동일한 변수들 ===
    episode_count = 0
    episode_step = 0
    gripper_close_steps = 0
    total_reward = 0.0
    success_count = 0
    fail_count = 0
    print_interval = 6
    
    # === 추가: 에피소드 기록 ===
    recorder = EpisodeRecorder()
    successful_trajectory = None
    
    obs = env.get_observations()
    
    print("\n" + "=" * 80)
    print("🚀 Sim2Real Play - 성공한 에피소드 기록")
    print("=" * 80)
    
    # === play2.py와 완전히 동일한 루프 ===
    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        
        start_time = time.time()
        
        # play2.py와 완전히 동일!
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        
        # play2.py와 완전히 동일!
        done = bool(dones[0].item()) if hasattr(dones[0], 'item') else bool(dones[0])
        reward = float(rewards[0].item()) if hasattr(rewards[0], 'item') else float(rewards[0])
        total_reward += reward
        episode_step += 1
        
        # play2.py와 완전히 동일!
        joint_pos_rad = sim_env._robot.data.joint_pos[0, :6].detach().cpu().numpy()
        joint_vel_rad = sim_env._robot.data.joint_vel[0, :6].detach().cpu().numpy()
        joint_pos_deg = joint_pos_rad * RAD2DEG
        joint_vel_deg = joint_vel_rad * RAD2DEG
        
        # play2.py와 완전히 동일!
        gripper_joints = sim_env._robot.data.joint_pos[0, 6:8].detach().cpu().numpy()
        gripper_closed_amount = gripper_joints[0] + gripper_joints[1]
        max_gripper_open = float(sim_env.max_gripper_open.item()) if hasattr(sim_env.max_gripper_open, 'item') else float(sim_env.max_gripper_open)
        gripper_pos = gripper_closed_amount / max_gripper_open if max_gripper_open > 0 else 0.0
        
        # play2.py와 완전히 동일!
        if hasattr(sim_env, 'should_close_gripper'):
            gripper_cmd = bool(sim_env.should_close_gripper[0].item())
        else:
            gripper_cmd = gripper_pos > 0.5
        
        # play2.py와 완전히 동일!
        if gripper_cmd:
            gripper_close_steps += 1
        else:
            gripper_close_steps = 0
        
        # === 추가: 기록 ===
        timestamp = episode_step * dt
        recorder.record(timestamp, joint_pos_deg, joint_vel_deg, gripper_pos, gripper_cmd, reward)
        
        # 출력
        if step % print_interval == 0:
            gripper_status = "CLOSE" if gripper_cmd else "OPEN"
            print(f"[{step:4d}] Ep:{episode_count:2d} Step:{episode_step:3d} | "
                  f"G:{gripper_pos:.2f}({gripper_status}) cls:{gripper_close_steps:3d} | "
                  f"R:{reward:7.3f} | J0:{joint_pos_deg[0]:6.1f}°")
        
        # play2.py와 완전히 동일한 에피소드 종료 처리!
        if done:
            # play2.py와 동일한 성공 조건: 리워드 > 10
            if total_reward > 10:
                success_count += 1
                result = "✅ SUCCESS"
                
                # 성공한 에피소드 저장
                if successful_trajectory is None:
                    successful_trajectory = recorder
                    print(f"\n{'='*60}")
                    print(f"에피소드 {episode_count} 종료: {result}")
                    print(f"  스텝: {episode_step}, 총 리워드: {total_reward:.2f}")
                    print(f"  ✨ 이 에피소드를 CSV로 저장합니다!")
                    print(f"{'='*60}\n")
                    break  # 첫 성공 에피소드에서 종료
            else:
                fail_count += 1
                result = "❌ FAIL"
            
            print(f"\n{'='*60}")
            print(f"에피소드 {episode_count} 종료: {result}")
            print(f"  스텝: {episode_step}, 총 리워드: {total_reward:.2f}")
            print(f"  성공: {success_count}, 실패: {fail_count}")
            print(f"{'='*60}\n")
            
            # 리셋
            episode_count += 1
            episode_step = 0
            gripper_close_steps = 0
            total_reward = 0.0
            recorder = EpisodeRecorder()  # 새 에피소드용 리코더
        
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # 결과 저장
    if successful_trajectory is not None:
        csv_path = os.path.join(args_cli.output_dir, "episode_trajectory.csv")
        successful_trajectory.save_csv(csv_path)
        print(f"\n📊 성공한 에피소드 저장 완료!")
        print(f"   기록: {len(successful_trajectory.timestamps)}개")
        print(f"   시간: {successful_trajectory.timestamps[-1]:.2f}초")
    else:
        print(f"\n⚠️ {args_cli.max_steps}스텝 내에 성공한 에피소드를 찾지 못했습니다.")
    
    print(f"\n📊 최종 결과")
    print(f"  총 에피소드: {episode_count}")
    print(f"  성공: {success_count}, 실패: {fail_count}")
    if episode_count > 0:
        print(f"  성공률: {success_count/episode_count*100:.1f}%")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
