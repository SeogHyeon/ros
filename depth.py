#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image

image_depth=Image()

def state_depth(depth_topic):
    image_depth.data=depth_topic.data

def check_depth(y): # y=[x1,y1,x2,y2] bounding box, 9등분점 중 최솟값 return
    box_point = np.zeros((3,9))
    image_depth.data=np.frombuffer(image_depth.data,dtype=np.uint8)
    image_depth.data = np.reshape(image_depth.data, (480,640,2))
    for i in range(3):
        for j in range(3):
            x_point=(y[0]*(j+1)+y[2]*(3-j))//4
            y_point=(y[1]*(i+1)+y[3]*(3-i))//4
            box_point[0][3*i+j]=(image_depth.data[y_point][x_point][0]+image_depth.data[y_point][x_point][1]*256)/1000### unit:m
            if not box_point[0][3*i+j]:
                box_point[0][3*i+j]=10 ######가끔 0으로 튀는 값 (10m)로 처리
            box_point[1][3*i+j]=x_point## x   순서      |0 1 2|
            box_point[2][3*i+j]=y_point## y             |3 4 5|
                                        #               |6 7 8|
    return box_point[:,np.argmin(box_point[0])] #   


def think():
    rospy.init_node('think')
    rate=rospy.Rate(1)
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, state_depth, queue_size=1) ####size : 480*640*2
    while not rospy.is_shutdown():
        rate.sleep()
        y=[300,230,340,250]
        min_distance=check_depth(y)### min_distance=[value,x,y]
        print(min_distance)
        


if __name__ == '__main__':
        try:
            think()
        except rospy.ROSInterruptException:
            pass