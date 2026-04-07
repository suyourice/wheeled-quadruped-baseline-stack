# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ArticulationCfg for the Unitree Go2-W (wheeled quadruped) robot."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


def _resolve_usd_path() -> str:
    """Return absolute path to go2w.usd, relative to repository root (cwd)."""
    rel = "assets/go2w/urdf/go2w_description/go2w.usd"
    return os.path.join(os.getcwd(), rel)


# ---------------------------------------------------------------------------
# Go2-W robot configuration
#
# Joint structure (16 actuated):
#   revolute  (×12): {FL|FR|RL|RR}_{hip|thigh|calf}_joint
#   continuous (×4): {FL|FR|RL|RR}_foot_joint  ← wheels
#
# URDF limits:
#   hip   : ±1.0472 rad,  23.7 Nm,  30.1 rad/s
#   thigh : varies per leg (front -1.57~3.49, rear -0.52~4.54), 23.7 Nm
#   calf  : -2.7227~-0.83776 rad (negative range!), 35.55 Nm, 20.07 rad/s
#   wheel : continuous (no limit), 23.7 Nm, 30.1 rad/s
# ---------------------------------------------------------------------------

GO2W_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_resolve_usd_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Calculated standing height:
        #   h = thigh_len*cos(0.4) + calf_len*cos(0.44) + wheel_r
        #     = 0.213*0.921 + 0.2264*0.905 + 0.086 = 0.487 m  (+0.013 margin)
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*_hip_joint": 0.0,
            # Thigh 0.4 rad: foot x-offset from hip ≈ -0.013 m (nearly vertical).
            #   foot_x = 0.213*sin(0.4) + 0.2264*sin(0.4-0.84) = -0.013 m
            # Base CoM is at x=+0.021 m (slightly forward) → symmetric stance is stable.
            "F[LR]_thigh_joint": 0.4,
            "R[LR]_thigh_joint": 0.4,
            # Calf near upper limit (-0.84 rad) for maximum leg extension.
            ".*_calf_joint": -0.84,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Leg joints (position-controlled via implicit PD)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=40.0,    # Nm/rad — tune as needed
            damping=2.0,       # Nms/rad
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0,
        ),
        # Wheel joints (velocity-controlled; stiffness=0 → pure torque/velocity)
        # Strategy A (legs only): set effort_limit_sim=0 to lock wheels.
        # Strategy B (hybrid): set stiffness=0, damping>0 for velocity drive.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            stiffness=0.0,     # no position stiffness → free-spinning
            damping=0.5,
            effort_limit_sim=23.7,
            velocity_limit_sim=30.1,
        ),
    },
)
