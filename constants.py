# ARMS_LENGTHS = [187.2, 117.5, 80.4 + 119]  # mm Third arm is last arm + gripper
import numpy as np

# ARMS_LENGTHS = np.array([0.9, 0.3, 0.3])  # mm Third arm is last arm + gripper
ARMS_LENGTHS = np.array([187.2, 117.5, 80.4 + 119]) / 1000  # mm Third arm is last arm + gripper
ARM_WIDTH = 0.05
TOTAL_ARM_LENGTH = np.sum(ARMS_LENGTHS)
ZERO_POS_BASE = (44 + 100) / 1000  # mm Base servo plus base

INITIAL_CONFIG_Q = np.array([0, np.pi * 0.8, -np.pi * 0.8]) # Initial robot angles
INITIAL_CONFIG_SERVO = np.array([0, np.pi * 0.8, -np.pi * 0.8]) # Initial robot angles