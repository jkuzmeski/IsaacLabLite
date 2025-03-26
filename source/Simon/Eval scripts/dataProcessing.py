# This script is created to compare different simulation results with different simulation parameters.
# The script reads the simulation results from the files and plots the results in a graph.

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




young_raw_data = pd.read_csv(fr'C:\Users\jkuzm\Isaac\IsaacLabLite\logs\skrl\humanoid_amp_walk\2025-03-26_15-07-02_amp_torch_test_normal\joint_data.csv')
# only include columns 0:29
young_joint_data = young_raw_data.iloc[0:153,1:29]
#convert data from radians to degrees
young_joint_data = young_joint_data*(180/np.pi)

young_right_ankle_x = young_joint_data.iloc[0:153, 23]
young_right_knee_x = young_joint_data.iloc[0:153, 20]
young_right_hip_x = young_joint_data.iloc[0:153, 13]


# old data
old_raw_data = pd.read_csv(fr'C:\Users\jkuzm\Isaac\IsaacLabLite\logs\skrl\humanoid_amp_walk\2025-03-26_15-42-23_amp_torch_test_old\joint_data.csv')
# only include columns 0:29
old_joint_data = old_raw_data.iloc[0:153,1:29]
#convert data from radians to degrees
old_joint_data = old_joint_data*(180/np.pi)

old_right_ankle_x = old_joint_data.iloc[0:153, 23]
old_right_knee_x = old_joint_data.iloc[0:153, 20]
old_right_hip_x = old_joint_data.iloc[0:153, 13]


reference_data = pd.read_csv(fr'C:\Users\jkuzm\Isaac\IsaacLabLite\source\Simon\humanoid_amp\motions\walk\humanoid_walk_dof_positions.csv')
reference_data = reference_data*(180/np.pi)

#plot the data
plt.figure()
plt.plot(young_right_ankle_x, label='right_ankle_sim_young')
plt.plot(old_right_ankle_x, label='right_ankle_sim_old')
plt.plot(reference_data['right_ankle_y'], label='right_ankle_ref')
plt.title('Right Ankle Flex/Ext')
plt.legend()
plt.show()

plt.figure()
plt.plot(young_right_knee_x, label='right_knee_sim_young')
plt.plot(old_right_knee_x, label='right_knee_sim_old')
plt.plot(reference_data['right_knee'], label='right_knee_ref')
plt.title('Right Knee Flex/ExtX')
plt.legend()
plt.show()

plt.figure()
plt.plot(young_right_hip_x, label='right_hip_sim_young')
plt.plot(old_right_hip_x, label='right_hip_sim_old')
plt.plot(reference_data['right_hip_y'], label='right_hip_ref')
plt.title('Right Hip Flex/Ext')
plt.legend()
plt.show()


# to trouble shoot i want to make a plot of each column in the joint data and compare it to the right knee data of the reference data

# for i in range(0, len(joint_data.columns)):
#     if 'amp' in joint_data.columns[i]:
#             checkdata = joint_data.iloc[:, i]
#             #plot the data
#             plt.figure()
#             plt.plot(checkdata, label='checkdata')
#             plt.plot(reference_data['left_hip_x'], label='left_hip_ref')
#             plt.title('Column ' + str(i))
#             plt.legend()
#             plt.show()

