import numpy as np
import math

from numpy import sin, cos


class robot_arm_3dof:

    def __init__(self, l):
        self.l = l  # link length

        self.q = np.array([0.0, 0.0, 0.0])  # joint position
        self.dq = np.array([0.0, 0.0, 0.0])  # joint velocity
        self.tau = np.array([0.0, 0.0, 0.0])  # joint torque

    # forward kinematics (until the second elbow, i.e., secondary endpoint)
    def FK2(self):
        p = np.zeros([2])  # endpoint position
        '''*********** Student should fill in ***********'''
        l = self.l
        q = self.q
        p[0] = l[0] * cos(q[0]) + l[1] * cos(q[0] + q[1])
        p[1] = l[0] * sin(q[0]) + l[1] * sin(q[0] + q[1])
        '''*********** Student should fill in ***********'''
        return p

    # Jacobian matrix (until the second elbow, i.e., secondary endpoint)
    def Jacobian2(self):
        '''*********** Student should fill in ***********'''
        l = self.l
        q = self.q
        J = np.array([[-l[0] * sin(q[0]) - l[1] * sin(q[0] + q[1]), -l[1] * sin(q[0] + q[1])],
                      [l[0] * cos(q[0]) + l[1] * cos(q[0] + q[1]), l[1] * cos(q[0] + q[1])]])
        '''*********** Student should fill in ***********'''
        return J

    # forward kinematics (until the end of the chain, i.e., primary endpoint)
    def FK4(self):
        p = np.zeros([2])  # endpoint position
        '''*********** Student should fill in ***********'''
        l = self.l
        q = self.q
        p[0] = l[0] * cos(q[0]) + l[1] * cos(q[0] + q[1]) + l[2] * cos(q[0] + q[1] + q[2])
        p[1] = l[0] * sin(q[0]) + l[1] * sin(q[0] + q[1]) + l[2] * sin(q[0] + q[1] + q[2])
        '''*********** Student should fill in ***********'''
        return p

    # Jacobian matrix (until the end of the chain, i.e., primary endpoint)
    def Jacobian4(self):
        '''*********** Student should fill in ***********'''
        l = self.l
        q = self.q
        dxq1 = -l[2] * sin(q[0] + q[1])
        dyq1 = l[2] * cos(q[0] + q[1])
        dxq2 = -l[2] * sin(q[0] + q[1] + q[2])
        dyq2 = l[2] * cos(q[0] + q[1] + q[2])
        J = np.array(
            [[-l[0] * sin(q[0]) + dxq1 + dxq2, dxq1 + dxq2, dxq2],
             [l[0] * cos(q[0]) + dyq1 + dyq2, dyq1 + dyq2, dyq2]])
        '''*********** Student should fill in ***********'''
        return J

    # inverse kinematics (until joint 2)
    def IK2(self, p):
        q = np.zeros([2])
        r = np.sqrt(p[0] ** 2 + p[1] ** 2)
        q[1] = np.pi - math.acos((self.l[0] ** 2 + self.l[1] ** 2 - r ** 2) / (2 * self.l[0] * self.l[1]))
        q[0] = math.atan2(p[1], p[0]) - math.acos((self.l[0] ** 2 - self.l[1] ** 2 + r ** 2) / (2 * self.l[0] * r))

        return q

    # state change
    def state(self, q, dq):
        self.q = q
        self.dq = dq


class Controller:

    def __init__(self, kp=1, ki=0, kd=0):
        self.lambda_coeff = 0.001

        self.i = 0  # loop counter
        self.se = 0.0  # integrated error
        self.state = []  # state vector
        self.last_p_end = np.array([0, 0])
        self.lastp_2 = np.array([0, 0])
        self.se = 0
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

    def pid_control(self, model, goal, dt):
        # KINEMATIC CONTROL
        p_end = model.FK4()
        error = goal - p_end
        d_p = p_end - self.last_p_end

        # F_end can be replaced with a reinforcement learning action
        F_end = self.Kp * error + self.Ki * self.se * dt - self.Kd * d_p / dt

        self.se += error
        self.last_p_end = p_end

        return F_end

    def control(self, model, goal, F_end):
        # KINEMATIC CONTROL
        J_end = model.Jacobian4()
        p_end = model.FK4()
        # error2 = - p2
        # d_p2 = p2 - self.last_p_2

        # F_2 = self.Kp * error2 + self.Ki * self.se * dt - self.Kd * d_p2 / dt

        def J_robust(_J):
            _Jt = _J.transpose()
            damp_identity = self.lambda_coeff * np.identity(len(_J))
            return _Jt @ np.linalg.inv(_J @ _Jt + damp_identity)

        J_end_robust = J_robust(J_end)
        # J_2_robust = J_2_robust
        # null_space_velocity = np.concatenate((J_2_robust @ F_2, np.zeros(1)))
        # null_space_control = (np.identity(len(J_end[0])) - J_end_robust @ J_end) @ null_space_velocity

        dq = J_end_robust @ F_end  # + null_space_control
        print("goal, p_end, F_end, dq", goal, p_end, F_end)

        p = p_end
        return p, dq


if __name__ == '__main__':
    from constants import ARMS_LENGTHS

    l = ARMS_LENGTHS
    model = robot_arm_3dof(l)
    controller = Controller()
