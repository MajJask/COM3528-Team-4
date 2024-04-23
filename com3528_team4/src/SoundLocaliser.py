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

class SoundLocalizer:
    def __init__(self, mic_distance=0.1):
        self.mic_distance = mic_distance
        self.speed_of_sound = 343  # Speed of sound in m/smport miro2 as miro

        self.x_len = 40000
        self.no_of_mics = 4
        self.input_mics = np.zeros((self.x_len, self.no_of_mics))
        #print(self.input_mics) 

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

        print("init success")

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
        print("left ear:", self.left_ear_data)
        delay_left_right = self.gcc(self.left_ear_data, self.right_ear_data)
        delay_left_tail = self.gcc(self.left_ear_data, self.tail_data)
        delay_right_tail = self.gcc(self.right_ear_data, self.tail_data)

        # Convert delays to angles using small angle approximation
        angle_left_right = (delay_left_right / self.speed_of_sound) * self.mic_distance
        angle_left_tail = (delay_left_tail / self.speed_of_sound) * self.mic_distance
        angle_right_tail = (delay_right_tail / self.speed_of_sound) * self.mic_distance

        # Simple average of angles as a naive triangulation approach
        estimated_direction = np.mean([angle_left_right, angle_left_tail, angle_right_tail])
        print("Got direction")
        return estimated_direction

    def callback_mics(self, data):
        # data for angular calculation
        self.audio_event = AudioEng.process_data(data.data)

        # data for dynamic thresholding
        data_t = np.asarray(data.data, 'float32') * (1.0 / 32768.0)
        data_t = data_t.reshape((4, 500))    
        self.head_data = data_t[2][:]
        if self.tmp is None:
            self.tmp = np.hstack((self.tmp, np.abs(self.head_data)))
        elif (len(self.tmp)<10500):
            self.tmp = np.hstack((self.tmp, np.abs(self.head_data)))
        else:
            # when the buffer is full
            self.tmp = np.hstack((self.tmp[-10000:], np.abs(self.head_data)))
            # dynamic threshold is calculated and updated when new signal come
            self.thresh = self.thresh_min + AudioEng.non_silence_thresh(self.tmp)

        # data for display
        data = np.asarray(data.data)
        # 500 samples from each mics
        data = np.transpose(data.reshape((self.no_of_mics, 500)))
        data = np.flipud(data)
        self.input_mics = np.vstack((data, self.input_mics[:self.x_len-500,:]))
        self.left_ear_data = np.flipud(self.input_mics[:, 0])
        self.right_ear_data = np.flipud(self.input_mics[:, 1])
        self.head_data = np.flipud(self.input_mics[:, 2])
        self.tail_data = np.flipud(self.input_mics[:, 3])
        print(self.process_data())

        

# Example of using the class
if __name__ == '__main__':
    print ("Initialising")
    rospy.init_node('sound_localizer')
    AudioEng = DetectAudioEngine()
    localizer = SoundLocalizer()
    direction = localizer.process_data()
    
    
    
    rospy.spin()  # Keeps Python from exiting until this node is stopped


