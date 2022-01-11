# SIMULATION PARAMETERS
import numpy as np
import pygame
from pygame import K_RIGHT, K_LEFT, K_UP, K_DOWN

from gym_robotic_arm.gym_robotic_arm.constants import ARMS_LENGTHS, TOTAL_ARM_LENGTH, ZERO_POS_BASE, INITIAL_CONFIG_Q, ARM_WIDTH, \
    INITIAL_CONFIG_SERVO, \
    CONTROL_DT

from gym_robotic_arm.gym_robotic_arm.dynamic_model import RobotArm3dof, PIDController
from serial import SerialException

from sim_utils import length, config_to_polygon_pygame, check_collision, config_to_polygon, arm_to_polygon
from visualization_util import draw_rectangle_from_config, DISPLAY
from visualize_robot_arm import Display
from gym_robotic_arm.gym_robotic_arm.arduino_communication import ArduinoControl
from gym_robotic_arm.gym_robotic_arm.camera import Camera
from gym_robotic_arm.gym_robotic_arm.env_GAIL.log_save import DataLog, Save_Expert


dt = CONTROL_DT
# ROBOT     PARAMETERS
x0 = 0.0  # base x position
y0 = 0.0  # base y position

# PID CONTROLLER PARAMETERS
Kp = 15  # proportional gain
Ki = 0.3  # integral gain
Kd = 0.1  # derivative gain


def keyboard_control(dt, goal):
    step_size = dt * 0.1
    goal = goal.copy()
    pygame.event.get()  # refresh keys
    keys = pygame.key.get_pressed()
    if keys[K_LEFT]:
        goal[0] -= step_size
    if keys[K_RIGHT]:
        goal[0] += step_size

    if keys[K_UP]:
        goal[1] += step_size
    if keys[K_DOWN]:
        goal[1] -= step_size
    return goal


def cap_goal(goal):
    local_goal = goal - robot_base
    l = length(local_goal)

    if l > TOTAL_ARM_LENGTH:
        shorter_local_goal = local_goal / l * TOTAL_ARM_LENGTH
        return shorter_local_goal + robot_base
    return goal


if __name__ == '__main__':

    l = ARMS_LENGTHS

    arduino = True
    arduino_control = None
    if arduino:
        try:
            arduino_control = ArduinoControl()
        except IOError as e:
            print(e)
    print("arduino_control",arduino_control)
    robot_base = np.array([0, ZERO_POS_BASE])
    local_endp_start = np.array([0.3, 0])

    robot_arm = RobotArm3dof(l=ARMS_LENGTHS, reset_q=INITIAL_CONFIG_Q, arduino_control=arduino_control)
    q = robot_arm.q

    controller = PIDController(kp=15, ki=0.1, kd=0.1)

    t = 0.0  # time

    state = []  # state vector
    p = robot_base + local_endp_start
    goal = robot_base + local_endp_start

    display = Display(dt, ARMS_LENGTHS, start_pos=robot_base)
    step = 0
    sent = 2

    # init camera
    camera = Camera(1)
    Camera.show_feed_continuous()
    datalogger = None

    while True:
        display.render(q, goal)

        goal = keyboard_control(dt, goal)
        goal = cap_goal(goal)

        # Control
        local_goal = goal - robot_base

        # F_end can be replaced with RL action. array[2]
        F_end = controller.control_step(robot_arm.FK_end_p(), local_goal, dt)

        p, q, dq = robot_arm.move_endpoint_xz(F_end, step)
        t += dt

        # Render
        for pol in robot_arm.arm_regions:
            pol = [xy + robot_base for xy in pol]
            draw_rectangle_from_config(pol)

        #if you want to save camera state, uncomment:
        #camera.save_image()

        # save state
        state.append([t, q[0], q[1], q[2], dq[0], dq[1], dq[2], p[0], p[1]])

        # try to keep it real time with the desired step time
        display.tick()
        pygame.display.flip()  # update display
        step += 1

        # save log, uncomment to create a GAIL expert dataset 
        '''observation = Camera.return_cam_obs()
        action = F_end
        datalogger = DataLog()
        datalogger.append_to_log(observation, action)
        
        #make button for datalogger.save with tkinter
        datalogger.save'''