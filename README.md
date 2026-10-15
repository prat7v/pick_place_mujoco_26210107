# Tutorial 4: Pick and Place in MuJoCo with IK

This repository implements a complete pick-and-place pipeline in MuJoCo for two robots:

- **Franka Emika Panda** (7-DOF arm + parallel-jaw gripper)
- **HEAL** (6-DOF arm + three-jaw gripper)

The pipeline is evaluated with three inverse-kinematics methods:
- Mink (off-the-shelf library)
- Damped Least Squares (DLS)
- Quadratic Programming (QP)

The work covers workspace analysis, table design, randomised cube poses, pick-and-place execution, and failure analysis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### HEAL model path
The HEAL XML is expected at:
```
/home/pratyaksh/ITR_mujoco_fk_lab/robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml
```
Update the path inside `src/environment_heal.py` if your location differs.

## Project Structure

```
pick_place_mujoco/
├── README.md
├── requirements.txt
├── src/
│   ├── environment.py          # Franka scene (table, cube, tray)
│   ├── environment_heal.py     # HEAL scene (table, cube, tray, three-jaw gripper)
│   ├── mink_controller.py      # Franka Mink controller
│   ├── ik_dls.py               # Franka DLS controller
│   ├── ik_qp.py                # Franka QP controller
│   └── mink_heal.py            # HEAL Mink controller
├── scripts/
│   ├── frank/
│   │   ├── batch_franka_mink_viewer.py
│   │   ├── batch_franka_dls_viewer.py
│   │   └── batch_franka_qp_viewer.py
│   └── heal/
│       └── batch_heal_mink_viewer.py
├── plots/
│   ├── frank_workspace.png
│   └── heal_workspace.png
├── report/
│   ├── Franka_Batch_Evaluation_Report.docx
│   └── HEAL_Batch_Evaluation_Report.docx
└── videos/
    └── (successful demonstration recordings)
```

## Running the Experiments

### Franka Emika Panda
```bash
python scripts/frank/batch_franka_mink_viewer.py
python scripts/frank/batch_franka_dls_viewer.py
python scripts/frank/batch_franka_qp_viewer.py
```

### HEAL
```bash
python scripts/heal/batch_heal_mink_viewer.py
```

Each script runs 20 randomised episodes (seeds 2000–2019) and prints the final XY offset of the cube from the tray centre.

## Results Summary

| Robot  | Method | Episodes | Success (≤ 0.05 m) | Notes                        |
|--------|--------|----------|--------------------|------------------------------|
| Franka | Mink   | 20       | 2                  | Physical grasp               |
| Franka | DLS    | 20       | 2                  | Identical to Mink            |
| Franka | QP     | 20       | 2                  | Identical to Mink            |
| HEAL   | Mink   | —        | —                  | See HEAL report for details  |

On the Franka the three IK methods produced identical final offsets for every seed. This shows that the inverse-kinematics solver is not the limiting factor once contact is made.

## Failure Analysis

- **Dominant failure mode (Franka):** The parallel-jaw gripper fails to hold the cube reliably after the fingers close. The cube is either never grasped or is dislodged during the subsequent lift.
- **Successful episodes:** Seeds 2000 and 2003 produced clear successes (offset ≤ 0.05 m). These correspond to favourable cube positions and orientations relative to the finger approach direction.
- **Marginal case:** Seed 2004 ended 0.16 m from the tray centre.
- **HEAL:** Results are documented separately in the HEAL evaluation report.

Improving contact parameters (higher friction, softer compliance) or adopting a more robust grasp strategy would be the most direct way to raise the success rate on the Franka.

## Requirements

- Python ≥ 3.10
- mujoco
- mink
- numpy

Full list in `requirements.txt`.

## Author

Debojit Das  
Tutorial 4 – Pick and Place in MuJoCo with IK
