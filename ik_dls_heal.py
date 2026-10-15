#!/usr/bin/env python3
"""
Damped Least Squares IK for the 6-DOF HEAL arm
"""

import numpy as np
import mujoco


class HealDLSIK:
    def __init__(self, model, data, ee_site_name="ee_site", lambda_damping=0.03):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.lambda_damping = lambda_damping
        self.dt = 0.005
        self.max_step = 0.6

    def get_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        return J

    def get_pose_error(self, target_pos, target_quat):
        current_pos = self.data.site_xpos[self.ee_site_id].copy()
        err_pos = target_pos - current_pos

        current_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self.ee_site_id])

        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, target_quat, current_quat)
        return np.concatenate([err_pos, err_rot])

    def solve(self, target_pos, target_quat):
        J = self.get_jacobian()
        err = self.get_pose_error(target_pos, target_quat)
        # Reduce orientation influence
        err[3:] *= 0.3
        JTJ = J.T @ J
        damp = self.lambda_damping ** 2 * np.eye(6)
        q_dot = np.linalg.solve(JTJ + damp, J.T @ err)
        q_dot = np.clip(q_dot, -self.max_step, self.max_step)
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
