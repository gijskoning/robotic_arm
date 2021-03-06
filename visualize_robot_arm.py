"""
Robot Dynamics and Control Assignment 2e: kinematic task-priority control
-------------------------------------------------------------------------------
DESCRIPTION:
4-DOF planar robot arm model with shoulder and elbow joints. The code includes
simulation environment and visualisation of the robot.

The robot is a classic position-controlled robot:
- Measured joint angle vector q is provided each sample time.
- Calculate the joint velocity dq in each sample time to be sent to the robot.

Important variables:
q[0] -> shoulder joint configuration
q[1] -> first elbow joint configuration
q[2] -> second elbow joint configuration
q[3] -> third elbow joint configuration
p[0] -> endpoint x position
p[1] -> endpoint y position

TASK:
Make the robot track a given endpoint reference trajectory with the primary
endpoint (end of the kinematic chain) by using a kinematic control (i.e., PID
controller). This robot structure has two redundant degrees of freedom. Use
null-space control to make the second elbow (secondary endpoint) track position
[0,0] as a secondary task to the primary endpoint task.
-------------------------------------------------------------------------------


INSTURCTOR: Luka Peternel
e-mail: l.peternel@tudelft.nl

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pygame
from numpy import sin, cos

from visualization_util import DISPLAY, WINDOW_SCALE

'''SIMULATION'''


#
# # REFERENCE TRAJETORY
# ts = T / dt  # trajectory size
# xt = np.linspace(-2, 2, int(ts))
# yt1 = np.sqrt(1 - (abs(xt) - 1) ** 2)
# yt2 = -3 * np.sqrt(1 - (abs(xt) / 2) ** 0.5)
#
# x = np.concatenate((xt, np.flip(xt, 0)), axis=0)
# y = np.concatenate((yt1, np.flip(yt2, 0)), axis=0)
#
# pr = np.array((x / 10 + 0.0, y / 10 + 0.1))  # reference endpoint trajectory

class Display:

    def __init__(self, dt, arm_lengths, start_pos):
        # initialise real-time plot with pygame
        self.arm_lengths = arm_lengths

        self.xc, self.yc = DISPLAY.get_rect().center  # window center
        pygame.display.set_caption('robot arm')

        self.font = pygame.font.Font('freesansbold.ttf', 12)  # printing text font and font size
        self.text = self.font.render('robot arm', True, (0, 0, 0), (255, 255, 255))  # printing text object
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (10, 10)  # printing text position with respect to the top-left corner of the window

        self.clock = pygame.time.Clock()  # initialise clock

        # SIMULATION PARAMETERS
        dts = dt * 1  # desired simulation step time (NOTE: it may not be achieved)
        self.T = 3  # total simulation time

        self.FPS = int(1 / dts)  # refresh rate

        # scaling
        self.window_scale = WINDOW_SCALE  # conversion from meters to pixels

        self.start = start_pos

    def render(self, q, goal):
        # real-time plotting
        DISPLAY.fill((255, 255, 255))  # clear window

        l1, l2, l3 = self.arm_lengths
        # update individual link position
        x0, y0 = self.start
        xbase, ybase = [0,0]
        # print(q[0])
        # print(l1)
        x1 = x0 + l1 * np.cos(q[0])
        y1 = y0 + l1 * np.sin(q[0])
        # print("xy",x1,y1)
        x2 = x1 + l2 * np.cos(q[0] + q[1])
        y2 = y1 + l2 * np.sin(q[0] + q[1])
        x3 = x2 + l3 * np.cos(q[0] + q[1] + q[2])
        y3 = y2 + l3 * np.sin(q[0] + q[1] + q[2])
        # x4 = x3 + l4 * np.cos(q[0] + q[1] + q[2] + q[3])
        # y4 = y3 + l4 * np.sin(q[0] + q[1] + q[2] + q[3])
        window_scale = self.window_scale
        xc, yc = self.xc, self.yc

        # xy_list = list(zip([xbase, x0, x1], [ybase, y0, y1]))
        xy_list = list(zip([xbase, x0, x1, x2, x3], [ybase, y0, y1, y2, y3]))

        xy_list_lines = xy_list.copy()
        points = np.array(xy_list_lines)
        #
        points *= window_scale
        points[:, 1] *= -1
        points[:] += np.array([xc, yc])
        pygame.draw.lines(DISPLAY, (0, 0, 255), False, points, 3)
        xc,yc = DISPLAY.get_rect().center


        def draw_points(xy, color=(0, 0, 0)):
            for x, y in xy:
                pygame.draw.circle(DISPLAY, color,
                                   (int(window_scale * x) + xc, int(-window_scale * y) + yc),
                                   5)

        draw_points(xy_list[0:-1])
        draw_points(xy_list[-1:], color=(255, 0, 0))

        # pygame.draw.circle(window, (255, 0, 0), (int(window_scale * x4) + xc, int(-window_scale * y4) + yc),
        #                    3)  # draw hand / endpoint
        pygame.draw.circle(DISPLAY, (0, 255, 0),
                           (int(window_scale * goal[0]) + xc, int(-window_scale * goal[1]) + yc),
                           3)  # draw reference position


        text = self.font.render("FPS = " + str(round(self.clock.get_fps())), True, (0, 0, 0), (255, 255, 255))
        DISPLAY.blit(text, self.textRect)

    def tick(self):
        self.clock.tick(self.FPS)


def plot_state(state):
    state = np.array(state)

    plt.figure(1)
    plt.subplot(211)
    plt.title("JOINT SPACE BEHAVIOUR")
    plt.plot(state[:, 0], state[:, 4], "b", label="shoulder")
    plt.plot(state[:, 0], state[:, 5], "r", label="elbow1")
    plt.plot(state[:, 0], state[:, 6], "g", label="elbow2")
    # plt.plot(state[:, 0], state[:, 7], "m", label="elbow3")
    plt.legend()
    plt.ylabel("dq [rad/s]")

    plt.subplot(212)
    plt.plot(state[:, 0], state[:, 1], "b", label="shoulder")
    plt.plot(state[:, 0], state[:, 2], "r", label="elbow1")
    plt.plot(state[:, 0], state[:, 3], "g", label="elbow2")
    # plt.plot(state[:, 0], state[:, 4], "m", label="elbow3")
    plt.legend()
    plt.ylabel("q [rad]")
    plt.xlabel("t [s]")

    plt.tight_layout()

    plt.figure(2)
    plt.title("ENDPOINT SPACE BEHAVIOUR")
    plt.plot(0, 0, "ok", label="shoulder")
    plt.plot(state[:, 9], state[:, 10], label="trajectory")
    plt.plot(state[0, 9], state[0, 10], "xg", label="start point")
    plt.plot(state[-4, 7], state[-4, 10], "+r", label="end point")
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()

    plt.tight_layout()
