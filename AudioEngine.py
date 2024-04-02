#!/usr/bin/env python3

"""
MiRo orienting towards a sound and approaching it, set in the context of the game "Marco Polo"
"""

from math import radians
import math
import os
import numpy as np
import rospy
import miro2 as miro
import geometry_msgs
from node_detect_audio_engine import DetectAudioEngine
from sensor_msgs.msg import Range
from std_msgs.msg import Int16MultiArray, UInt16MultiArray
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, TwistStamped
import time


class AudioClient():

    def __init__(self):
        # Microphone Parameters
        # Number of points to display
        self.x_len = 40000
        # number of microphones coming through on topic
        self.no_of_mics = 4

        # Generate figure for plotting mics
        self.fig = plt.figure()
        self.fig.suptitle("Microphones")  # Give figure title

        # HEAD
        self.head_plot = self.fig.add_subplot(4, 1, 3)
        self.head_plot.set_ylim([-33000, 33000])
        self.head_plot.set_xlim([0, self.x_len])
        self.head_xs = np.arange(0, self.x_len)
        self.head_plot.set_xticklabels([])
        self.head_plot.set_yticks([])
        self.head_plot.grid(which="both", axis="x")
        self.head_plot.set_ylabel("Head", rotation=0, ha="right")
        self.head_ys = np.zeros(self.x_len)
        self.head_line, = self.head_plot.plot(self.head_xs, self.head_ys, linewidth=0.5, color="g")

        # LEFT EAR
        self.left_ear_plot = self.fig.add_subplot(4, 1, 1)
        self.left_ear_plot.set_ylim([-33000, 33000])
        self.left_ear_plot.set_xlim([0, self.x_len])
        self.left_ear_xs = np.arange(0, self.x_len)
        self.left_ear_plot.set_xticklabels([])
        self.left_ear_plot.set_yticks([])
        self.left_ear_plot.grid(which="both", axis="x")
        self.left_ear_plot.set_ylabel("Left Ear", rotation=0, ha="right")
        self.left_ear_ys = np.zeros(self.x_len)
        self.left_ear_line, = self.left_ear_plot.plot(self.left_ear_xs, self.left_ear_ys, linewidth=0.5, color="b")

        # RIGHT EAR
        self.right_ear_plot = self.fig.add_subplot(4, 1, 2)
        self.right_ear_plot.set_ylim([-33000, 33000])
        self.right_ear_plot.set_xlim([0, self.x_len])
        self.right_ear_xs = np.arange(0, self.x_len)
        self.right_ear_plot.set_xticklabels([])
        self.right_ear_plot.set_yticks([])
        self.right_ear_plot.grid(which="both", axis="x")
        self.right_ear_plot.set_ylabel("Right Ear", rotation=0, ha="right")
        self.right_ear_ys = np.zeros(self.x_len)
        self.right_ear_line, = self.right_ear_plot.plot(self.right_ear_xs, self.right_ear_ys, linewidth=0.5, color="r")

        # Tail
        self.tail_plot = self.fig.add_subplot(4, 1, 4)
        self.tail_plot.set_ylim([-33000, 33000])
        self.tail_plot.set_xlim([0, self.x_len])
        self.tail_xs = np.arange(0, self.x_len)
        self.tail_plot.set_yticks([])
        self.tail_plot.set_xlabel("Samples")
        self.tail_plot.grid(which="both", axis="x")
        self.tail_plot.set_ylabel("Tail", rotation=0, ha="right")
        self.tail_ys = np.zeros(self.x_len)
        self.tail_line, = self.tail_plot.plot(self.tail_xs, self.tail_ys, linewidth=0.5, color="c")

        self.fig.subplots_adjust(hspace=0, wspace=0)

        self.input_mics = np.zeros((self.x_len, self.no_of_mics))

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # save previous head data
        self.tmp = []

        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
                                         Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)

        self.sub_sonar = rospy.Subscriber(topic_base_name + "/sensors/sonar",
                                          Range, self.callback_sonar, queue_size=1)

        # publishers
        self.pub_push = rospy.Publisher(topic_base_name + "/core/mpg/push", miro.msg.push, queue_size=1)
        self.pub_wheels = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=1)
        self.kinematic_pub = rospy.Publisher(topic_base_name + "/control/kinematic_joints", JointState, queue_size=1)
        self.pub_tone = rospy.Publisher(topic_base_name + "/control/tone", UInt16MultiArray, queue_size=1)

        # prepare push message
        self.msg_push = miro.msg.push()
        self.msg_push.link = miro.constants.LINK_HEAD
        self.msg_push.flags = (miro.constants.PUSH_FLAG_NO_TRANSLATION + miro.constants.PUSH_FLAG_VELOCITY)

        # status flags
        self.audio_event = None
        self.orienting = False
        self.action_time = 1  # secs
        self.thresh = 0.05

        # time
        self.frame_p = None
        self.msg_wheels = TwistStamped()
        self.controller = miro.lib.PoseController()
        self.cmd_vel = miro.lib.DeltaPose()

        # dynamic threshold
        self.thresh = 0
        self.thresh_min = 0.03

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