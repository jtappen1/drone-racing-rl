# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Persistent state for potential-based progress reward
        self._prev_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.env.num_lap_completed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._total_resets = 0

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains — roll/pitch
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # PID controller gains — yaw
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    # =========================================================================
    # REWARDS
    # =========================================================================
    def get_rewards(self) -> torch.Tensor:

        # -----------------------------------------------------------------
        # Gate Pass Detection
        # -----------------------------------------------------------------
        curr_x = self.env._pose_drone_wrt_gate[:, 0]
        curr_y = self.env._pose_drone_wrt_gate[:, 1]
        curr_z = self.env._pose_drone_wrt_gate[:, 2]
        prev_x = self.env._prev_x_drone_wrt_gate

        half_side = self.env._gate_model_cfg_data.gate_side / 2.0
        pass_half = half_side * 0.8

        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)

        # Velocity alignment with gate normal (inverted — normals point opposite to pass direction)
        gate_normal = self.env._normal_vectors[self.env._idx_wp]
        vel_dir = torch.nn.functional.normalize(self.env._robot.data.root_com_lin_vel_w, dim=1)
        alignment = torch.sum(vel_dir * gate_normal, dim=1)

        gate_passed = (
            (prev_x > 0.0)
            & (curr_x < 0.0)
            & (dist_to_gate < 1.0)
            & (torch.abs(curr_y) < pass_half)
            & (torch.abs(curr_z) < pass_half)
            & (alignment < -0.2)
        )
        ids_gate_passed = torch.where(gate_passed)[0]

        # Increment gate counter and advance waypoint
        self.env._n_gates_passed[ids_gate_passed] += 1
        self.env._idx_wp[ids_gate_passed] = (
            (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        )

        # Recompute pose wrt new gate to prevent false double-triggers
        if len(ids_gate_passed) > 0:
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3],
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = (
                self.env._pose_drone_wrt_gate[ids_gate_passed, 0]
            )

        # Update desired position for envs that passed a gate
        self.env._desired_pos_w[ids_gate_passed, :3] = self.env._waypoints[
            self.env._idx_wp[ids_gate_passed], :3
        ]

        # -----------------------------------------------------------------
        # Potential-based Progress Reward (dense)
        # -----------------------------------------------------------------
        distance_to_goal = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        # Sharper potential: divisor 1.5 instead of 3.0 gives stronger gradient
        progress = 1.0 - torch.tanh(distance_to_goal / 1.5)

        progress_diff = progress - self._prev_progress
        self._prev_progress = progress.clone()

        # Zero out progress diff for envs that just passed a gate
        # (distance jumps to new gate — would cause large negative reward)
        progress_diff[ids_gate_passed] = 0.0

        # -----------------------------------------------------------------
        # Gate Pass Reward (sparse)
        # -----------------------------------------------------------------
        gate_pass_reward = torch.zeros(self.num_envs, device=self.device)
        gate_pass_reward[ids_gate_passed] = 1.0

        # -----------------------------------------------------------------
        # Lap Completion Reward (sparse)
        # -----------------------------------------------------------------
        lap_completed_reward = torch.zeros(self.num_envs, device=self.device)
        curr_lap = self.env._n_gates_passed // self.env._waypoints.shape[0]
        lap_completed = (
            (self.env._idx_wp == 0)
            & (self.env._n_gates_passed >= self.env._waypoints.shape[0])
        )
        give_lap_reward = (curr_lap > self.env.num_lap_completed) & lap_completed
        self.env.num_lap_completed[give_lap_reward] = curr_lap[give_lap_reward].to(
            self.env.num_lap_completed.dtype
        )
        lap_completed_reward[give_lap_reward] = 1.0

        # -----------------------------------------------------------------
        # Crash Detection (dense)
        # -----------------------------------------------------------------
        contact_forces = self.env._contact_sensor.data.net_forces_w
        force_mag = torch.norm(contact_forces, dim=-1)
        crashed = (force_mag > 1e-8).any(dim=1).int()

        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # -----------------------------------------------------------------
        # Speed Reward (dense)
        # -----------------------------------------------------------------
        direction_to_gate = self.env._desired_pos_w[:, :3] - self.env._robot.data.root_link_pos_w
        direction_to_gate_norm = direction_to_gate / (torch.linalg.norm(direction_to_gate, dim=1, keepdim=True) + 1e-6)
        velocity_w = self.env._robot.data.root_com_lin_vel_w
        speed_toward_gate = torch.sum(velocity_w * direction_to_gate_norm, dim=1)
        speed_reward = torch.clamp(speed_toward_gate, min=0.0)

        # -----------------------------------------------------------------
        # Angular Velocity Penalty (dense)
        # -----------------------------------------------------------------
        ang_vel_b = self.env._robot.data.root_ang_vel_b
        ang_vel_penalty = torch.sum(ang_vel_b ** 2, dim=-1)


        # -----------------------------------------------------------------
        # Time Penalty (dense) — encourages speed
        # -----------------------------------------------------------------
        time_penalty = torch.ones(self.num_envs, device=self.device)

        # -----------------------------------------------------------------
        #  Altitude Penalty (dense) — prevents sitting on ground
        # -----------------------------------------------------------------
        gate_altitude = self.env._desired_pos_w[:, 2]  # current target gate height
        drone_altitude = self.env._robot.data.root_link_pos_w[:, 2]
        # altitude_error = torch.clamp(gate_altitude - drone_altitude, min=0.0)  # only penalize when below gate
        altitude_error = gate_altitude - drone_altitude # penalize when too high or too low now
        altitude_penalty = altitude_error ** 2

        # -----------------------------------------------------------------
        # Combine Rewards
        # -----------------------------------------------------------------
        if self.cfg.is_train:
            rewards = {
                "progress_goal": progress_diff * self.env.rew["progress_goal_reward_scale"],
                "gate_passed": gate_pass_reward * self.env.rew["gate_passed_reward_scale"],
                "crash": crashed * self.env.rew["crash_reward_scale"],
                "lap_passed": lap_completed_reward * self.env.rew["lap_passed_reward_scale"],
                "altitude": altitude_penalty * self.env.rew["altitude_reward_scale"],
                "speed": speed_reward * self.env.rew["speed_reward_scale"],
                #"gate_alignment": gate_alignment_reward * self.env.rew["gate_alignment_reward_scale"],
                "ang_vel_penalty": ang_vel_penalty * self.env.rew["ang_vel_penalty_reward_scale"],
                #"action_smoothness": action_smoothness_penalty * self.env.rew["action_smoothness_reward_scale"],
                "time_penalty": time_penalty * self.env.rew["time_penalty_reward_scale"],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew["death_cost"],
                reward,
            )

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()

        return reward

    # =========================================================================
    # OBSERVATIONS (22 dims, all body frame)
    # =========================================================================
    def get_observations(self) -> Dict[str, torch.Tensor]:

        # Linear velocity in body frame (3)
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b

        # Angular velocity in body frame (3)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        # Projected gravity vector in body frame (3)
        # Replaces quaternion — avoids q/-q ambiguity, only 3 dims
        quat_w = self.env._robot.data.root_quat_w
        rot_matrix = matrix_from_quat(quat_w)  # (num_envs, 3, 3)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        projected_gravity = torch.matmul(
            rot_matrix.transpose(1, 2),
            gravity_world.unsqueeze(-1),
        ).squeeze(-1)  # (num_envs, 3)

        # Current gate position in body frame (3)
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]
        gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            current_gate_pos_w,
        )

        # Current gate normal in body frame (3)
        gate_normal_w = self.env._normal_vectors[current_gate_idx]
        rot_matrix = matrix_from_quat(self.env._robot.data.root_quat_w)
        gate_normal_b = torch.matmul(
            rot_matrix.transpose(1, 2),
            gate_normal_w.unsqueeze(-1),
        ).squeeze(-1)  # (num_envs, 3)

        # Next gate position in body frame (3)
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            next_gate_pos_w,
        )

        # Previous actions (4)
        prev_actions = self.env._previous_actions

        # Total: 3 + 3 + 3 + 3 + 3 + 3 + 4 = 22 dims
        obs = torch.cat(
            [
                drone_lin_vel_b,       # (3) velocity
                drone_ang_vel_b,       # (3) body rates
                projected_gravity,     # (3) orientation
                gate_pos_b,            # (3) current gate in body frame
                gate_normal_b,         # (3) current gate normal in body frame
                next_gate_pos_b,       # (3) next gate in body frame
                prev_actions,          # (4) previous actions
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    # =========================================================================
    # RESET
    # =========================================================================
    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Robot reset
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]
            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)
            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # =================================================================
        # SPAWN CURRICULUM (training only)
        # =================================================================
        self.env.num_lap_completed[env_ids] = 0
        num_waypoints = self.env._waypoints.shape[0]
        self._total_resets += len(env_ids)

        if self.cfg.is_train:
            iteration = self.env.iteration

            # waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
    
            # # 30% of envs spawn behind gate 2 for powerloop practice
            # hard_mask = torch.rand(n_reset, device=self.device) < 0.3
            # waypoint_indices[hard_mask] = 2
            
            # x_local = torch.empty(n_reset, device=self.device).uniform_(-2.5, -0.5)
            # y_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            # z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
            # yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)

            if iteration < 150:
                # Learn to pass gate 0 from a fixed position
                waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
                x_local = -2.0 * torch.ones(n_reset, device=self.device)
                y_local = torch.zeros(n_reset, device=self.device)
                z_local = torch.zeros(n_reset, device=self.device)
                yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)

            else:
                # Spawn behind gate 0 with noise — goal is to chain gate 0 → gate 1
                waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
                x_local = torch.empty(n_reset, device=self.device).uniform_(-2.5, -0.5)
                y_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
                z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
                yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)

            # elif iteration < 650:
            #     # Spawn behind gate 1 — goal is to chain gate 1 → gate 2 (approach the powerloop)
            #     waypoint_indices = torch.ones(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            #     x_local = torch.empty(n_reset, device=self.device).uniform_(-2.0, -0.5)
            #     y_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            #     z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
            #     yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)

            # elif iteration < 900:
            #     # Spawn behind gate 2 — goal is gate 2 → gate 3 (the actual powerloop)
            #     waypoint_indices = 2 * torch.ones(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            #     x_local = torch.empty(n_reset, device=self.device).uniform_(-1.5, -0.3)
            #     y_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            #     z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
            #     yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)

            # elif iteration < 1100:
            #     # Spawn behind gate 0, 1, or 2 — chain through the powerloop from further back
            #     waypoint_indices = torch.randint(0, 3, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
            #     x_local = torch.empty(n_reset, device=self.device).uniform_(-2.5, -0.5)
            #     y_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
            #     z_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            #     yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.25, 0.25)

            # elif iteration < 1400:
            #     # Spawn at gates 0-5, learn the chicane section
            #     waypoint_indices = torch.randint(0, 6, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
            #     x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
            #     y_local = torch.empty(n_reset, device=self.device).uniform_(-0.7, 0.7)
            #     z_local = torch.empty(n_reset, device=self.device).uniform_(-0.4, 0.4)
            #     yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            # else:
            #     # Phase 6: full track
            #     waypoint_indices = torch.randint(0, num_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
            #     x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
            #     y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)
            #     z_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
            #     yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
        else:
            # Play mode: spawn behind the initial waypoint
            waypoint_indices = self.env._initial_wp * torch.ones(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)
            z_local = torch.zeros(n_reset, device=self.device)
            yaw_noise = torch.zeros(n_reset, device=self.device)

        # -----------------------------------------------------------------
        # Compute world-frame spawn position from gate-local offset
        # -----------------------------------------------------------------
        x0_wp = self.env._waypoints[waypoint_indices, 0]
        y0_wp = self.env._waypoints[waypoint_indices, 1]
        z_wp = self.env._waypoints[waypoint_indices, 2]
        theta = self.env._waypoints[waypoint_indices, -1]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_wp + z_local

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # Point drone towards the target gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise,
        )
        default_root_state[:, 3:7] = quat

        # Play mode: start from ground
        if not self.cfg.is_train:
            default_root_state[:, 2] = 0.05

        # -----------------------------------------------------------------
        # Write state to sim
        # -----------------------------------------------------------------
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids, :3] = self.env._waypoints[waypoint_indices, :3].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2]
            - self.env._robot.data.root_link_pos_w[env_ids, :2],
            dim=1,
        )
        self.env._n_gates_passed[env_ids] = 0

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset tracking variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3],
        )
        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        self.env._crashed[env_ids] = 0

        # Reset progress tracking for potential-based reward
        distance_to_goal = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids],
            dim=1,
        )
        self._prev_progress[env_ids] = 1.0 - torch.tanh(distance_to_goal / 3.0)