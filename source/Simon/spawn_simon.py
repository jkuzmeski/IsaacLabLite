# Import necessary modules
import torch
from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab_assets import HUMANOID_28_CFG
from isaaclab.assets import Articulation

# Launch the simulator (similar to your spawn_simon.py)
app_launcher = AppLauncher()
simulation_app = app_launcher.app

# Create a simulation context
sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)

# Create the humanoid articulation asset
# We'll use a simple path rather than the environment regex
humanoid_cfg = HUMANOID_28_CFG.replace(prim_path="/World/Robot")
humanoid = Articulation(humanoid_cfg)

# Play the simulator to initialize physics
sim.reset()
sim.step()

# Now you can access joint and body information
print(f"Number of joints: {humanoid.num_joints}")
print(f"Joint names (ordered): {humanoid.joint_names}")
print(f"Number of bodies: {humanoid.num_bodies}")
print(f"Body names: {humanoid.body_names}")

# Access joint position limits
joint_limits = humanoid.data.joint_pos_limits[0]  # First instance
print("\nJoint position limits:")
for i, name in enumerate(humanoid.joint_names):
    print(f"  {name}: [{joint_limits[i][0]:.3f}, {joint_limits[i][1]:.3f}]")

# Access default joint positions
default_positions = humanoid.data.default_joint_pos[0]  # First instance
print("\nDefault joint positions:")
for i, name in enumerate(humanoid.joint_names):
    print(f"  {name}: {default_positions[i]:.3f}")

# To get the height, you can use the root position or analyze body positions
# After stepping the simulation:
root_height = humanoid.data.root_pos_w[0, 2]  # Z-coordinate of root position
print(f"\nRoot height: {root_height:.3f}")

# To get total height, you could find the highest body part
body_positions = humanoid.data.body_pos_w[0]  # Get positions of all bodies
max_height = body_positions[:, 2].max().item()
min_height = body_positions[:, 2].min().item()
total_height = max_height - min_height
print(f"Approximate total height: {total_height:.3f}")

# Continue simulation for a few steps
for _ in range(10):
    sim.step()

# Close the simulator when done
simulation_app.close()