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
/home/pratyaksh/ITR_mujoco_fk_lab/robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml
src/
  environment.py          # Franka scene
  environment_heal.py     # HEAL scene
  mink_controller.py      # Franka Mink
  ik_dls.py               # Franka DLS
  ik_qp.py                # Franka QP
  mink_heal.py            # HEAL Mink
scripts/
  frank/                  # Franka batch runners
  heal/                   # HEAL batch runners
plots/                    # Workspace visualisations
report/                   # Evaluation reports
videos/                   # Successful demonstrations
Running
Franka
Bashpython scripts/frank/batch_franka_mink_viewer.py
python scripts/frank/batch_franka_dls_viewer.py
python scripts/frank/batch_franka_qp_viewer.py
HEAL
Bashpython scripts/heal/batch_heal_mink_viewer.py
Results Summary








































RobotMethodEpisodesSuccess (≤0.05 m)NotesFrankaMink202Physical graspFrankaDLS202Identical to MinkFrankaQP202Identical to MinkHEALMink——See report
On the Franka the three IK methods produced identical outcomes. The limiting factor is physical grasping with the parallel-jaw gripper, not the choice of IK solver.
Failure Analysis

Most Franka failures occur because the cube is not securely held after the fingers close.
Successful episodes correspond to favourable cube poses relative to the approach direction.
Improving contact parameters (friction, compliance) or using a more robust grasp strategy would raise the success rate.

Requirements

Python ≥ 3.10
mujoco
mink
numpy

See requirements.txt.

Save the file. This covers both robots, how to run the scripts, and a short results summary.




Save the file. This covers b
