#!/usr/bin/env python3

import numpy as np
import os
import rospy
import miro2 as miro
from AudioEngine import DetectAudioEngine
from std_msgs.msg import Int16MultiArray
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geometry_msgs.msg import Twist, TwistStamped
import time
from scipy.signal import find_peaks
from miro2.lib import wheel_speed2cmd_vel


class SoundLocalizer:
    def __init__(self, mic_distance=0.1):
        self.mic_distance = mic_distance

        self.averaging = False
        self.rotating = False
        self.x_len = 40000
        self.no_of_mics = 4
        self.input_mics = np.zeros((self.x_len, self.no_of_mics))

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
                                         Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)

        # publishers
        self.pub_push = rospy.Publisher(topic_base_name + "/core/mpg/push", miro.msg.push, queue_size=0)
        self.pub_wheels = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)

        # prepare push message
        self.msg_push = miro.msg.push()
        self.msg_push.link = miro.constants.LINK_HEAD
        self.msg_push.flags = (miro.constants.PUSH_FLAG_NO_TRANSLATION + miro.constants.PUSH_FLAG_VELOCITY)

        # time
        self.msg_wheels = TwistStamped()

        self.left_ear_data = np.flipud(self.input_mics[:, 0])
        self.right_ear_data = np.flipud(self.input_mics[:, 1])
        self.head_data = np.flipud(self.input_mics[:, 2])
        self.tail_data = np.flipud(self.input_mics[:, 3])

        # Running average stuff
        self.t1_values = []
        self.t2_values = []

        print("init success")

    @staticmethod
    def block_data(data, block_size=500):
        # Calculate the number of blocks
        num_of_blocks = len(data) // block_size

        blocks = []

        for i in range(num_of_blocks):
            start = i * block_size
            end = start + block_size

            block = data[start:end]

            # Add the block to the list of blocks
            blocks.append(block)

        return np.array(blocks)

    @staticmethod
    def find_high_peaks(audio_data):
        peaks, _ = find_peaks(audio_data, height=0.6)

        return peaks

    @staticmethod
    def create_block(index, data, block_size=500):
        # take the data around an index and create a block half of block size before and after the index
        block = data[index - block_size // 2:index + block_size // 2]

        return block

    def process_data(self):

        # get the high points
        peak_l = self.find_high_peaks(self.left_ear_data)
        peak_r = self.find_high_peaks(self.right_ear_data)
        peak_t = self.find_high_peaks(self.tail_data)

        # find a common points
        # Convert to sets
        set_l_peak = set(peak_l)
        set_r_peak = set(peak_r)
        set_t_peak = set(peak_t)

        # Try to find common high points and convert to blocks
        try:
            common_high_points = set_l_peak.intersection(set_r_peak, set_t_peak)

            common_values_l = [self.left_ear_data[point] for point in common_high_points]
            common_values_r = [self.right_ear_data[point] for point in common_high_points]
            common_values_t = [self.tail_data[point] for point in common_high_points]

            # Calculate the sum of values for each common high point
            # By doing this we can find the common high point with the largest accumulative value
            sum_values = [self.left_ear_data[point] + self.right_ear_data[point] + self.tail_data[point] for point in
                          common_high_points]

            # Find the index of the maximum sum
            max_index = np.argmax(sum_values)

            # Get the common high point with the largest accumulative value
            max_common_high_point = list(common_high_points)[max_index]

            threshold = 600
            # check that common values reach threshold
            if max(common_values_l) < threshold or max(common_values_r) < threshold or max(common_values_t) < threshold:
                return None

            # Get block around max common high point
            max_common_block_l = self.create_block(max_common_high_point, self.left_ear_data)
            max_common_block_r = self.create_block(max_common_high_point, self.right_ear_data)
            max_common_block_t = self.create_block(max_common_high_point, self.tail_data)

            x_l_r = xco = np.correlate(max_common_block_l, max_common_block_r, mode='same')
            x_l_t = xco = np.correlate(max_common_block_l, max_common_block_t, mode='same')
            x_r_t = xco = np.correlate(max_common_block_r, max_common_block_t, mode='same')

            t1 = np.cos(np.argmax(x_l_r) * 343) / .1
            t2 = np.cos(np.argmax(x_l_t) * 343) / .25

            print(t1, t2)

            return t1, t2
        except Exception as e:
            print("No common high points")
            return None, None

    def callback_mics(self, data):

        global av1, av2
        if not self.rotating:
            # data for angular calculation
            # data for display
            data = np.asarray(data.data)
            # 500 samples from each mics
            data = np.transpose(data.reshape((self.no_of_mics, 500)))  # after this step each row is a sample and each
            # column is the mag. at that sample time for each mic
            data = np.flipud(data)  # flips as the data comes in reverse order
            self.input_mics = np.vstack((data, self.input_mics[:self.x_len - 500, :]))
            self.left_ear_data = np.flipud(self.input_mics[:, 0])
            self.right_ear_data = np.flipud(self.input_mics[:, 1])
            self.head_data = np.flipud(self.input_mics[:, 2])
            self.tail_data = np.flipud(self.input_mics[:, 3])

            t1, t2, = None, None
            try:
                if not self.rotating:
                    t1, t2 = self.process_data()
            except Exception as e:
                t1 = None
                t2 = None

            # running average stuff
            if not t1 is None and not self.averaging:  # if theres a value through and we are averaging start tracking
                self.t1_values.append(t1)
                self.t2_values.append(t2)

            if t1 is None and self.averaging and len(self.t1_values) > 0:
                try:

                    av1, av2 = np.average(self.t1_values), np.average(self.t2_values)
                    print('avgs', av1, av2)
                except Exception as e:
                    print(e)
                    print(self.t1_values, self.t2_values)

                print('turning')
                self.averaging = False
                an = self.estimate_angle(av1, av2)
                self.turn_to_sound(an)
                time.sleep(2)
                self.t1_values = []
                self.t2_values = []

            self.averaging = t1 is None and not self.averaging  # sets averaging to true if none and not already averaging

        return None

    def estimate_angle(self, t1, t2):
        # t1 lr, t2 l, back

        # pos on right ear
        # neg on left ear
        # pos if in front 
        # negative if behind 

        angle = 0
        if t2 < 0:  # then the sound is coming from behind
            angle += 180
        if t2 < 0:
            angle += 0
        if t1 > 0:  # then the sound is coming from the right
            angle += 45
        if t1 < 0:  # then the sound is coming from the left
            angle += -45

        print(angle)
        print(np.deg2rad(angle))
        return np.deg2rad(angle)

    def turn_to_sound(self, azimuth):
        print("trying to turn")
        self.rotating = True
        # Turn to the sound source
        tf = 2
        t0 = 0
        counter = 0
        while t0 <= tf:
            counter += 1

            # self.drive(v*2,v*2)
            self.msg_wheels.twist.linear.x = 0.0
            self.msg_wheels.twist.angular.z = azimuth / 2

            # test output
            # self.msg_wheels.twist.angular.z = 0.0

            self.pub_wheels.publish(self.msg_wheels)
            rospy.sleep(0.01)
            t0 += 0.01

        self.rotating = False


# Example of using the class
if __name__ == '__main__':
    print("Initialising")
    rospy.init_node('sound_localizer')
    AudioEng = DetectAudioEngine()
    localizer = SoundLocalizer()
    direction = localizer.process_data()

    rospy.spin()  # Keeps Python from exiting until this node is stopped
