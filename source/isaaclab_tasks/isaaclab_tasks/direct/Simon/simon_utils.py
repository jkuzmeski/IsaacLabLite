"""Utility functions for the Humanoid AMP environment."""

def get_observation_dimensions():
    """
    Returns a dictionary mapping observation dimensions to their meaning.
    
    Let me break down the observation space for you based on the compute_obs function in your code:

    DOF Positions (dimensions 0-27):
        28 joint positions
    DOF Velocities (dimensions 28-55):
        28 joint velocities
    Root Height (dimension 56):
        Z-coordinate of the reference body (torso)
    Root Orientation (dimensions 57-62):
        Tangent vector (x,y,z): 3D
        Normal vector (x,y,z): 3D
        These are derived from the quaternion through quaternion_to_tangent_and_normal()
    Root Linear Velocity (dimensions 63-65):
        Linear velocity (x,y,z) of the reference body
    Root Angular Velocity (dimensions 66-68):
        Angular velocity (x,y,z) of the reference body
    Key Body Positions (dimensions 69-80):
        Relative positions of 4 key bodies (right_hand, left_hand, right_foot, left_foot)
        Each position is 3D (x,y,z), so 4*3 = 12 dimensions
        These positions are relative to the root body position (torso)
    
    
    
    The HumanoidAmpEnv observation space consists of 81 dimensions structured as follows:
    """
    
    # Count the DOFs (28 for HUMANOID_28_CFG)
    num_dofs = 28
    
    # Create the dimension mapping
    dim_map = {}
    
    # 1. Joint positions (28 dimensions)
    start_idx = 0
    for i in range(num_dofs):
        dim_map[start_idx + i] = f"dof_position[{i}]"
    
    # 2. Joint velocities (28 dimensions)
    start_idx += num_dofs
    for i in range(num_dofs):
        dim_map[start_idx + i] = f"dof_velocity[{i}]"
    
    # 3. Root body height (1 dimension)
    start_idx += num_dofs
    dim_map[start_idx] = "root_height"
    
    # 4. Root body orientation - tangent and normal vectors (6 dimensions)
    start_idx += 1
    dim_map[start_idx + 0] = "root_tangent_x"
    dim_map[start_idx + 1] = "root_tangent_y"
    dim_map[start_idx + 2] = "root_tangent_z"
    dim_map[start_idx + 3] = "root_normal_x"
    dim_map[start_idx + 4] = "root_normal_y"
    dim_map[start_idx + 5] = "root_normal_z"
    
    # 5. Root body linear velocity (3 dimensions)
    start_idx += 6
    dim_map[start_idx + 0] = "root_lin_vel_x"
    dim_map[start_idx + 1] = "root_lin_vel_y"
    dim_map[start_idx + 2] = "root_lin_vel_z"
    
    # 6. Root body angular velocity (3 dimensions)
    start_idx += 3
    dim_map[start_idx + 0] = "root_ang_vel_x"
    dim_map[start_idx + 1] = "root_ang_vel_y"
    dim_map[start_idx + 2] = "root_ang_vel_z"
    
    # 7. Key body positions (4 bodies × 3 dimensions = 12 dimensions)
    # These are relative to root position (torso)
    start_idx += 3
    bodies = ["right_hand", "left_hand", "right_foot", "left_foot"]
    for i, body in enumerate(bodies):
        dim_map[start_idx + i*3 + 0] = f"{body}_pos_rel_x"
        dim_map[start_idx + i*3 + 1] = f"{body}_pos_rel_y"
        dim_map[start_idx + i*3 + 2] = f"{body}_pos_rel_z"
    
    return dim_map

def extract_observation_components(obs):
    """
    Extract meaningful components from the observation vector.
    
    Args:
        obs: A vector of shape (81,) or a batch of observations
        
    Returns:
        A dictionary of observation components
    """
    if len(obs.shape) > 1:
        # Handle batched observations
        components = {}
        components["dof_positions"] = obs[:, :28]
        components["dof_velocities"] = obs[:, 28:56]
        components["root_height"] = obs[:, 56:57]
        components["root_orientation"] = obs[:, 57:63]
        components["root_linear_velocity"] = obs[:, 63:66]
        components["root_angular_velocity"] = obs[:, 66:69]
        components["key_body_positions"] = obs[:, 69:81].reshape(-1, 4, 3)
    else:
        # Handle single observation
        components = {}
        components["dof_positions"] = obs[:28]
        components["dof_velocities"] = obs[28:56]
        components["root_height"] = obs[56:57]
        components["root_orientation"] = obs[57:63]
        components["root_linear_velocity"] = obs[63:66]
        components["root_angular_velocity"] = obs[66:69]
        components["key_body_positions"] = obs[69:81].reshape(4, 3)
    
    return components
