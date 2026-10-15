#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, ".")
import numpy as np
import mujoco, mujoco.viewer
from src.environment_heal import HealPickPlaceEnv
from src.mink_heal import HealMinkController

N_EPISODES = 20
env = HealPickPlaceEnv()
ctrl = HealMinkController(env.model, env.data)
ctrl.ee_task.position_cost = 5.0
down_quat = np.array([1.0, 0.0, 0.0, 0.0])

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for ep in range(N_EPISODES):
        seed = 1000 + ep
        cube_pos, _ = env.reset(seed=seed)
        print(f"\n=== Mink Episode {ep} (seed {seed}) ===")

        for _ in range(20):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.02)

        target = cube_pos.copy()
        target[2] = env.table_top_z + 0.22
        for _ in range(100):
            ctrl.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        target[2] = env.table_top_z + 0.06
        for _ in range(80):
            ctrl.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        print("  closing")
        for i in range(60):
            open_amt = max(0.0, 0.04 * (1.0 - i / 40))
            env.set_gripper(open_amt)
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        target[2] = env.table_top_z + 0.18
        for _ in range(80):
            ctrl.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        target = env.tray_pos.copy()
        target[2] = env.table_top_z + 0.18
        for _ in range(100):
            ctrl.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        target[2] = env.table_top_z + 0.05
        for _ in range(60):
            ctrl.move_to(target, down_quat, n_steps=1)
            viewer.sync()
            time.sleep(0.01)

        env.set_gripper(0.04)
        for _ in range(30):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        final, _ = env.get_cube_pose()
        offset = np.linalg.norm(final[:2] - env.tray_pos[:2])
        print(f"  Mink offset: {offset:.3f} m")

    print("Mink finished.")
    while viewer.is_running():
        mujoco.mj_step(env.model, env.data)
        viewer.sync()
