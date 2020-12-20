#!/usr/bin/env python

import rospy
import math
import time


from control import *
from gps import *
import cv2
import time

from kobuki_project.msg import Status
from geometry_msgs.msg import Twist



publisher_status = rospy.Publisher('kobuki_status', Status, queue_size=1)
publisher_action=rospy.Publisher('/kobuki_velocity',Twist,queue_size=1)

status=Status()
twist=Twist()

mission_Range = 1
# GPS 생성
gps = gps(port='/dev/ttyUSB0',
          brate=115200,
          gps_filename='cilabReversePath.txt',
          mode='trace',
        #   mode='record',/';
          degree_mode='gps',
          start_range_num=mission_Range,
          imu_port='COM3',
          plot_flag=True)

# ############################################################
# def state():
#     rate = rospy.Rate(10)
#     start=time.time()
#     ###################
#     ####움직이는부분~~##########
#     P=0.3   # path planning 정도
#     safety_distance=0.5
#     twist.linear.x=0.3
#     angular_path=0
#     angular_obstacle=0  
#     obstacle_coefficient=0.5    ### 장애물 피하는 정도
#     ########angular_path value setting
#     Range, STEER = gps.gps_tracer()
#     if type(STEER) == int:
#         angular_path=-STEER/2000

#     twist.angular.z=P*angular_path+(1-P)*angular_obstacle/10
#     publisher_action.publish(twist)
#     ###########################
#     rate.sleep()
#     print('time:',time.time()-start)
#     print('FPS: %.3f' %(1/(time.time()-start)))

def think():
    rospy.init_node('think')
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
      P=0.3   # path planning 정도
      safety_distance=0.5
      twist.linear.x=0.7
      angular_path=0
      angular_obstacle=0  
      obstacle_coefficient=0.5    ### 장애물 피하는 정도
      ########angular_path value setting
      Range, STEER = gps.gps_tracer()
      if type(STEER) == int:
        angular_path=-STEER/1000

      twist.angular.z=angular_path
      publisher_action.publish(twist)
      rate.sleep()

if __name__ == '__main__':
    try:
        think()
        
    except rospy.ROSInterruptException:
        pass