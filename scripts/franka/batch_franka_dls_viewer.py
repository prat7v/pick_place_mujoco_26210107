#!/usr/bin/env python3
import sys, time
sys.path.insert(0, ".")
import numpy as np
import mujoco, mujoco.viewer
from src.environment import PickPlaceEnv
from src.ik_dls import DLSIKController

N_EPISODES = 20
env = PickPlaceEnv()
ik = DLSIKController(env.model, env.data, lambda_damping=0.05)
down_quat = np.array([0.0, 1.0, 0.0, 0.0])
finger_ids = [env.model.nu - 2, env.model.nu - 1]

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for ep in range(N_EPISODES):
        seed = 2000 + ep
        cube_pos, _ = env.reset(seed=seed)
        print(f"\n=== Franka DLS Episode {ep} (seed {seed}) ===")

        for _ in range(20):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.02)

        target = cube_pos.copy()
        target[2] = env.table_top_z + 0.15
        for _ in range(100):
            ik.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        target[2] = env.table_top_z + 0.03
        for _ in range(80):
            ik.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        print("  closing")
        for i in range(60):
            open_amt = max(0.0, 0.04 * (1.0 - i / 40))
            for fid in finger_ids:
                env.data.ctrl[fid] = open_amt
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        target[2] = env.table_top_z + 0.15
        for _ in range(80):
            ik.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        target = env.tray_pos.copy()
        target[2] = env.table_top_z + 0.15
        for _ in range(100):
            ik.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        target[2] = env.table_top_z + 0.03
        for _ in range(60):
            ik.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        for fid in finger_ids:
            env.data.ctrl[fid] = 0.04
        for _ in range(30):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        final, _ = env.get_cube_pose()
        offset = np.linalg.norm(final[:2] - env.tray_pos[:2])
        print(f"  DLS offset: {offset:.3f} m")

    print("Franka DLS finished.")
    while viewer.is_running():
        mujoco.mj_step(env.model, env.data)
        viewer.sync()
