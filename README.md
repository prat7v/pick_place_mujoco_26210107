# Tutorial 4: Pick and Place in MuJoCo with IK

Implements a complete pick-and-place pipeline for two robots:
- **Franka Emika Panda** (7-DOF + parallel-jaw gripper)
- **HEAL** (6-DOF + three-jaw gripper)

Three inverse-kinematics methods are evaluated: Mink, Damped Least Squares (DLS), and Quadratic Programming (QP).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
