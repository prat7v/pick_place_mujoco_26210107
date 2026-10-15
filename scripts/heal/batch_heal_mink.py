#!/usr/bin/env python3
"""
Small batch logging for HEAL Mink (kinematic attachment).
Runs N episodes and records metrics.
"""

import sys, os, time, json
sys.path.insert(0, ".")
import numpy as np
import mujoco
from src.environment_heal import HealPickPlaceEnv
from src.mink_heal import HealMinkController

N_EPISODES = 20
LOG_PATH = "logs/heal_mink_batch.json"
os.makedirs("logs", exist_ok=True)

env = HealPickPlaceEnv()
ctrl = HealMinkController(env.model, env.data)
ctrl.ee_task.position_cost = 5.0

down_quat = np.array([1.0, 0.0, 0.0, 0.0])
logs = []

print(f"Running {N_EPISODES} episodes...")

for ep in range(N_EPISODES):
    seed = 1000 + ep
    cube_pos, cube_yaw = env.reset(seed=seed)
    t0 = time.time()

    # Approach
    target = cube_pos.copy()
    target[2] = env.table_top_z + 0.20
    for _ in range(120):
        ctrl.move_to(target, down_quat, n_steps=1)

    # Descend
    target[2] = env.table_top_z + 0.06
    for _ in range(80):
        ctrl.move_to(target, down_quat, n_steps=1)

    # Attach
    env.grasp_offset = np.array([0.0, 0.0, -0.04])
    env.grasped = True
    ee = env.data.site_xpos[env.ee_site_id].copy()
    qadr = env.cube_joint_id
    env.data.qpos[qadr:qadr+3] = ee + env.grasp_offset
    env.data.qvel[qadr:qadr+6] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Lift + move to tray
    target = env.tray_pos.copy()
    target[2] = env.table_top_z + 0.18
    for _ in range(100):
        ee = env.data.site_xpos[env.ee_site_id].copy()
        env.data.qpos[qadr:qadr+3] = ee + env.grasp_offset
        env.data.qvel[qadr:qadr+6] = 0.0
        ctrl.move_to(target, down_quat, n_steps=1)

    # Place
    target[2] = env.table_top_z + 0.05
    for _ in range(60):
        ee = env.data.site_xpos[env.ee_site_id].copy()
        env.data.qpos[qadr:qadr+3] = ee + env.grasp_offset
        env.data.qvel[qadr:qadr+6] = 0.0
        ctrl.move_to(target, down_quat, n_steps=1)

    # Release
    env.grasped = False
    mujoco.mj_forward(env.model, env.data)

    final, _ = env.get_cube_pose()
    elapsed = time.time() - t0
    offset = np.linalg.norm(final[:2] - env.tray_pos[:2])
    success = offset < 0.15  # 15 cm tolerance

    entry = {
        "episode": ep,
        "seed": seed,
        "start_xy": cube_pos[:2].tolist(),
        "yaw": float(cube_yaw),
        "final_xy": final[:2].tolist(),
        "offset_m": float(offset),
        "success": bool(success),
        "time_s": round(elapsed, 2),
    }
    logs.append(entry)
    print(f"Ep {ep:02d} | offset={offset:.3f} m | success={success} | time={elapsed:.1f}s")

# Summary
successes = sum(1 for e in logs if e["success"])
avg_time = np.mean([e["time_s"] for e in logs])
avg_offset = np.mean([e["offset_m"] for e in logs])

summary = {
    "n_episodes": N_EPISODES,
    "success_rate": successes / N_EPISODES,
    "mean_time_s": round(avg_time, 2),
    "mean_offset_m": round(avg_offset, 3),
    "episodes": logs,
}

with open(LOG_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("\n===== SUMMARY =====")
print(f"Success rate : {summary['success_rate']*100:.1f}%")
print(f"Mean time    : {summary['mean_time_s']} s")
print(f"Mean offset  : {summary['mean_offset_m']} m")
print(f"Log saved to {LOG_PATH}")
