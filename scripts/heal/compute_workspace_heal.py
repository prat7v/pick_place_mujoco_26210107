#!/usr/bin/env python3
"""
Phase 1: Sample and visualise HEAL workspace
"""

import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt

XML_PATH = "/home/pratyaksh/ITR_mujoco_fk_lab/robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Use the end-effector site
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_center")
if site_id < 0:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_site_eff")

print(f"Using site id: {site_id}")

N = 8000
positions = []

np.random.seed(42)
for _ in range(N):
    # Sample within joint limits
    q = np.zeros(model.nq)
    for i in range(model.nq):
        low, high = model.jnt_range[i]
        if high > low:
            q[i] = np.random.uniform(low, high)
        else:
            q[i] = np.random.uniform(-np.pi, np.pi)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    positions.append(data.site_xpos[site_id].copy())

positions = np.array(positions)

print("\nWorkspace bounds (metres):")
print(f"  X: {positions[:,0].min():.3f}  to  {positions[:,0].max():.3f}")
print(f"  Y: {positions[:,1].min():.3f}  to  {positions[:,1].max():.3f}")
print(f"  Z: {positions[:,2].min():.3f}  to  {positions[:,2].max():.3f}")

# Plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(positions[:,0], positions[:,1], positions[:,2], s=1, alpha=0.3)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("HEAL Task Workspace")

ax2 = fig.add_subplot(122)
ax2.scatter(positions[:,0], positions[:,1], s=1, alpha=0.3)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Top-down view")
ax2.set_aspect("equal")

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/heal_workspace.png", dpi=150)
print("\nPlot saved to plots/heal_workspace.png")
plt.show()
