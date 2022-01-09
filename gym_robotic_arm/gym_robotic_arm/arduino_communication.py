import math

import numpy as np
import time

from serial import Serial


# for wsl: https://devblogs.microsoft.com/commandline/connecting-usb-devices-to-wsl/
# sudo chmod 666 /dev/ttyS4
class ArduinoControl:

    def __init__(self, port='COM6'):
        print("trying port", port)

        # initiate arduino communication
        self.arduino = Serial(port=port, baudrate=115200, timeout=.1)

    def set_servo(self, q, debug=False):
        """
        :param q: numpy array for each joint angle
        :return:
        """
        q = q / math.pi * 100
        q = q.astype(int)
        q = np.clip(q, 0, 100)
        s1, s2, s3 = q.astype(int)
        # self.write_message(f"0:{s1},1:{s2},2:{s3}", debug)
        #write servo angles, servo identifier:{angle}
        self.write_message(f"0:{s1},1:{s2}", debug)

    def write_message(self, x, debug=False):
        if debug:
            print("Sending: ", x)
        _bytes = bytes(str(x), 'utf-8')
        print("Length bytes: ", len(_bytes))
        self.arduino.write(_bytes)
        time.sleep(0.05)
        data = self.arduino.readline()
        if debug and data is not None:
            print("data:", data)
        return data

    # while True:
    #     num = input("Enter a number: ") # Taking input from user
    #     value = write_read(num)
    #     print(value) # printing the value
    # Step 4: Arduino Code

    # :param action: [x,z,z_rot,gripper_pos] floats

    def sent_action(self, action, debug=False):
        """
        # :param action: [q1,q2,q3,q4] floats q4 is gripper
        """
        # [360-660]
        self.set_servo(action[:3], debug)


if __name__ == '__main__':
    # test code
    arduino_control = ArduinoControl()
    arduino_control.sent_action("test")
