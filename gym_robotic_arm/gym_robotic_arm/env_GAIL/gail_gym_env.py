from abc import abstractmethod

import os
import gym
from camera import Camera
from gym import spaces
import numpy as np

from gym_robotic_arm.constants import ARMS_LENGTHS, TOTAL_ARM_LENGTH

from gym_robotic_arm.dynamic_model import RobotArm3dof


class GailRobotArm(gym.Env):
    # need to copy code from https://github.com/TianhongDai/hindsight-experience-replay
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, robot_controller):
        self.robot_controller = robot_controller

        # initiate camera
        self.camera = Camera(1)
        self.camera.show_feed_continuous()
        
        # camera resolution to initiate observation space
        # laptop cam resolution and gripper/clamp cam resolution
        self.res_l = [self.camera.laptop_cam_res[0], self.camera.laptop_cam_res[1]]
        self.res_c = [self.camera.clamp_cam_res[0], self.camera.clamp_cam_res[1]]

        # initiate obseration space
        # code from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
        self.observation_space = spaces.Box(
            low=0, high=255, shape=((self.res_c[0], self.res_c[1], 3), 
            (self.res_l[0], self.res_l[1], 3)), dtype=np.uint8
        )

        # define actions space
        # action: {1: base joints, 2: 1st hanging joint, 3: second hanging joint,
        #          4: gripper rotation , 5: gripper close, 6: stepper rotation}
        self.action_space = spaces.Box(low=-1,high=1,shape=(6,),dtype=np.float32)
        
        self._max_episode_steps = 500
        self.episode_step = 0
        self.sim = None


    def step(self, action):
        """"
        action: numpy.ndarray containing 3 floats (x,z,gripper_pos)
            x,z [-1,1] movement of the endpoint (gripper)
            gripper_pos [0,1] 0 gripper fully open and 1 fully closed
        """
        #  todo still need actions defined for z rotation and gripper
        self.robot_controller.move_endpoint_xz(action[:2])
        
        state = self.camera.return_cam_obs()

        reward = self._compute_reward()
        
        # todo improve checking when episode is done
        done = self.episode_step >= self._max_episode_steps  # Done is true means the episode is over.
        self.episode_step += 1
        info = {'is_success': False}

        return np.array(state, dtype=np.float32), reward, done, info  # obs, reward, done, info

    def reset(self):
        # todo Reset environment and return new observation
        self.robot_controller.reset()
        self.episode_step = 0

        # initiate obseration space
        # code from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
        self.observation_space = spaces.Box(
            low=0, high=255, shape=((self.res_c[0], self.res_c[1], 3), (self.res_l[0], self.res_l[1], 3)), dtype=np.uint8
        )

        # define actions space
        # action: {1: base joints, 2: 1st hanging joint, 3: second hanging joint,
        #          4: gripper rotation , 5: gripper close, 6: stepper rotation}
        self.action_space = spaces.Box(low=-1,high=1,shape=(6,),dtype=np.float32)
        return

    def _compute_reward(self):
        # todo compute reward based on observation and robot model

        # Basic reward enforcing robot to go to certain goal_x
        #end_p = self.robot_controller.FK_end_p()
        #goal_x = TOTAL_ARM_LENGTH - 0.1
        #return -abs(goal_x - end_p[0])
        return

    @abstractmethod
    def _get_observation(self):
        return self.camera.return_cam_obs()

    @abstractmethod
    def render(self, mode='human', height=100, width=100):
        raise NotImplementedError()
        # height and width are used to center crop the image.
        # No need to return

    def seed(self, seed=None):
        # todo could implement seed when using random numbers
        pass

    def close(self):
        pass

    def _render_callback(self):
        # Required by RL repo
        pass