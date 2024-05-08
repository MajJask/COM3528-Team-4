#!/usr/bin/env python3

import numpy as np
import os
import rospy
import miro2 as miro
import geometry_msgs
from AudioEngine import DetectAudioEngine
from std_msgs.msg import Int16MultiArray
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geometry_msgs.msg import Twist, TwistStamped
import time
from scipy.signal import find_peaks


class SoundLocalizer:
    def __init__(self, mic_distance=0.1):
        self.mic_distance = mic_distance
        self.speed_of_sound = 343  # Speed of sound in m/smport miro2 as miro

        self.x_len = 40000
        self.no_of_mics = 4
        self.input_mics = np.zeros((self.x_len, self.no_of_mics))
        # print(self.input_mics)

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
                                         Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)

        # publishers
        self.pub_push = rospy.Publisher(topic_base_name + "/core/mpg/push", miro.msg.push, queue_size=0)
        self.pub_wheels = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)

        self.left_ear_data = np.flipud(self.input_mics[:, 0])
        self.right_ear_data = np.flipud(self.input_mics[:, 1])
        self.head_data = np.flipud(self.input_mics[:, 2])
        self.tail_data = np.flipud(self.input_mics[:, 3])

        self.tmp = []
        self.thresh = 0
        self.thresh_min = 0.03

        # Running average stuff
        self.t1_values = []  # for each period of direction calulation we populate these lists
        self.t2_values = []  # then once no more high points are foudn we will use these values to calculate averge values

        print("init success")

    def block_data(self, data, block_size=500):
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

    def find_high_peaks(self, audio_data):
        peaks, _ = find_peaks(audio_data, height=0.5)

        return peaks

    def create_block(self, index, data, block_size=500):
        # take the data around an index and create a block half of block size before and after the index
        block = data[index - block_size // 2:index + block_size // 2]

        return block

    def cast_peak_data(self, high_point, sequence):
        # take the high point and cast it to the sequence
        peak_block = self.create_block(high_point, sequence)
        return peak_block

    def gcc(self, mic1, mic2):
        pad = np.zeros(len(mic1))
        s1 = np.hstack([mic1, pad])
        s2 = np.hstack([pad, mic2])
        f_s1 = np.fft.fft(s1)
        f_s2 = np.fft.fft(s2)
        f_s2c = np.conj(f_s2)
        f_s = f_s1 * f_s2c
        denom = np.abs(f_s)
        denom[denom == 0] = 1e-10
        f_s /= denom
        correlation = np.fft.ifft(f_s)
        delay = np.argmax(np.abs(correlation)) - len(mic1)
        print("Acquired delay: " + str(delay))
        return delay

    def process_data(self):

        # get the high points
        print(self.left_ear_data)
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
            sum_values = [self.left_ear_data[point] + self.right_ear_data[point] + self.tail_data[point] for point in
                          common_high_points]

            # Find the index of the maximum sum
            max_index = np.argmax(sum_values)

            # Get the common high point with the largest accumulative value
            max_common_high_point = list(common_high_points)[max_index]

            threshold = 500
            # check that common values reach threshold
            if max(common_values_l) < threshold or max(common_values_r) < threshold or max(common_values_t) < threshold:
                return None

            # Get block around max common high point
            max_common_block_l = self.create_block(max_common_high_point, self.left_ear_data)
            max_common_block_r = self.create_block(max_common_high_point, self.right_ear_data)
            max_common_block_t = self.create_block(max_common_high_point, self.tail_data)

            delay_left_right = self.gcc(max_common_block_l, max_common_block_r)
            x_l_r = xco = np.correlate(max_common_block_l, max_common_block_r, mode='same')
            delay_left_tail = self.gcc(max_common_block_l, max_common_block_t)
            x_l_t = xco = np.correlate(max_common_block_l, max_common_block_t, mode='same')
            delay_right_tail = self.gcc(max_common_block_r, max_common_block_t)
            x_r_t = xco = np.correlate(max_common_block_r, max_common_block_t, mode='same')

            t1  = np.cos(np.argmax(x_l_r) * 343)/(.1)
            t2 = np.cos(np.argmax(x_l_t) * 343)/(.25)
            print(np.average(90-t1, t2))   

            return t1 , t2

            # Convert delays to angles using small angle approximation
            # angle_left_right = (delay_left_right / self.speed_of_sound) * self.mic_distance
            # angle_left_tail = (delay_left_tail / self.speed_of_sound) * self.mic_distance
            # angle_right_tail = (delay_right_tail / self.speed_of_sound) * self.mic_distance

            # # Simple average of angles as a naive triangulation approach
            # estimated_direction = np.mean([angle_left_right, angle_left_tail, angle_right_tail])
        except Exception as e:
            print("No common high points")
            return None, None

    def callback_mics(self, data):
        # data for angular calculation
        self.audio_event = AudioEng.process_data(data.data)

        # data for dynamic thresholding
        data_t = np.asarray(data.data, 'float32') * (1.0 / 32768.0)
        data_t = data_t.reshape((4, 500))
        self.head_data = data_t[2][:]
        if self.tmp is None:
            self.tmp = np.hstack((self.tmp, np.abs(self.head_data)))
        elif (len(self.tmp) < 10500):
            self.tmp = np.hstack((self.tmp, np.abs(self.head_data)))
        else:
            # when the buffer is full
            self.tmp = np.hstack((self.tmp[-10000:], np.abs(self.head_data)))
            # dynamic threshold is calculated and updated when new signal come
            self.thresh = self.thresh_min + AudioEng.non_silence_thresh(self.tmp)

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
        
        t1, t2 = self.process_data()
        if t1 is None and t2 is None:  # then there are no high points (sound not produced)
            if 




# Example of using the class
if __name__ == '__main__':
    print("Initialising")
    rospy.init_node('sound_localizer')
    AudioEng = DetectAudioEngine()
    localizer = SoundLocalizer()
    direction = localizer.process_data()
    

    rospy.spin()  # Keeps Python from exiting until this node is stopped
