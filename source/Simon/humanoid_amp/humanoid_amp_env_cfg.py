# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation: int = MISSING

    # spaces
    observation_space = 81
    action_space = 28
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 81

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "torso"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )


@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
 
@configclass
class HumanoidAmpWalkYoungEnvCfg(HumanoidAmpEnvCfg):
    """Humanoid AMP environment config for elderly walking."""
    
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
    decimation = 1
    
    # Modified simulation parameters with smaller timestep for stability
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )


@configclass
class HumanoidAmpWalkNormalEnvCfg(HumanoidAmpEnvCfg):
    """Humanoid AMP environment config for elderly walking."""
    
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
    decimation = 2
    
    # Modified simulation parameters with smaller timestep for stability
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )
@configclass
class HumanoidAmpWalkOldEnvCfg(HumanoidAmpEnvCfg):
    """Humanoid AMP environment config for elderly walking."""
    
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
    decimation = 4
    
    # Modified robot parameters to simulate elderly movement - with more stable settings
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # Replace velocity_limit with velocity_limit_sim to address the warning
                velocity_limit_sim=70.0,  # More conservative velocity limit
                stiffness=45.0,  # Add moderate joint stiffness for stability
                damping=3.0,  # Moderate damping - not too high to avoid instability
            ),
        },
    )
    
    # Modified simulation parameters with smaller timestep for stability
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )
    
@configclass
class HumanoidAmpWalkReallyOldEnvCfg(HumanoidAmpEnvCfg):
    """Humanoid AMP environment config for elderly walking."""
    
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
    decimation = 6
    
    # Modified robot parameters to simulate elderly movement - with more stable settings
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # Replace velocity_limit with velocity_limit_sim to address the warning
                velocity_limit_sim=40.0,  # More conservative velocity limit
                stiffness=45.0,  # Add moderate joint stiffness for stability
                damping=5.0,  # Moderate damping - not too high to avoid instability
            ),
        },
    )
    
    # Modified simulation parameters with smaller timestep for stability
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )