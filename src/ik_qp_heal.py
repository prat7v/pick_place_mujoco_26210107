#!/usr/bin/env python3
"""
QP-based IK for the 6-DOF HEAL arm
"""

import numpy as np
import mujoco
from qpsolvers import solve_qp


class HealQPIK:
    def __init__(self, model, data, ee_site_name="ee_site", lambda_damping=0.05):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.lambda_damping = lambda_damping
        self.dt = 0.005
        self.max_step = 0.5

    def get_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return np.vstack([jacp[:, :6], jacr[:, :6]])

    def get_pose_error(self, target_pos, target_quat):
        current_pos = self.data.site_xpos[self.ee_site_id].copy()
        err_pos = target_pos - current_pos
        current_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self.ee_site_id])
        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, target_quat, current_quat)
        return np.concatenate([err_pos, err_rot * 0.3])

    def solve(self, target_pos, target_quat):
        J = self.get_jacobian()
        err = self.get_pose_error(target_pos, target_quat)
        P = J.T @ J + self.lambda_damping ** 2 * np.eye(6)
        q = -J.T @ err

        # Joint limit constraints (simple velocity limits)
        G = np.vstack([np.eye(6), -np.eye(6)])
        h = np.hstack([np.full(6, self.max_step), np.full(6, self.max_step)])

        q_dot = solve_qp(P, q, G, h, solver="osqp", verbose=False)
        if q_dot is None:
            q_dot = np.zeros(6)
        return q_dot

    def move_to(self, target_pos, target_quat, n_steps=100):
        for _ in range(n_steps):
            q_dot = self.solve(target_pos, target_quat)
            self.data.qpos[:6] += q_dot * self.dt
            for i in range(6):
                if self.model.jnt_limited[i]:
                    lo, hi = self.model.jnt_range[i]
                    self.data.qpos[i] = np.clip(self.data.qpos[i], lo, hi)
            mujoco.mj_forward(self.model, self.data)
