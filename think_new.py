#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image as iii
import numpy as np
import cv2

import time
import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import torch

from control import *
from gps import *
import cv2
import time

from kobuki_project.msg import Status
from geometry_msgs.msg import Twist

publisher_status = rospy.Publisher('/kobuki_status', Status, queue_size=1)
publisher_action=rospy.Publisher('/kobuki_velocity',Twist,queue_size=1)

status=Status()
twist=Twist()
# """----------------------------------- model, weight 불러오기  --------------------------------------"""
parser = argparse.ArgumentParser()
###########
# parser.add_argument('--cfg', type=str, default='/home/cilab/yolov3/cfg/yolov3.cfg', help='*.cfg path')
# parser.add_argument('--names', type=str, default='/home/cilab/yolov3/data/coco.names', help='*.names path')
# parser.add_argument('--weights', type=str, default='/home/cilab/yolov3/weights/yolov3.pt', help='weights path')

# parser.add_argument('--cfg', type=str, default='/home/cilab/yolov3/cfg/yolov3-tiny-cilab.cfg', help='*.cfg path')
# parser.add_argument('--names', type=str, default='/home/cilab/yolov3/data/coco.names', help='*.names path')
# parser.add_argument('--weights', type=str, default='/home/cilab/weights/best_tiny_from_scratch.pt', help='weights path')

###########
parser.add_argument('--cfg', type=str, default='/home/cilab/yolov3/cfg/yolov3-cilab.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='/home/cilab/yolov3/data/coco_custom2.names', help='*.names path')
parser.add_argument('--weights', type=str, default='/home/cilab/weights/best.pt', help='weights path')
parser.add_argument('--img-size', type=int, default=256, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()
opt.cfg = check_file(opt.cfg)  # check file
opt.names = check_file(opt.names)  # check file
print(opt)

weights, half, view_img = opt.weights, opt.half, opt.view_img
imgsz = (256, 320)
# Initialize
device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
model = Darknet(opt.cfg, imgsz)

# Load weights
attempt_download(weights)
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)

# Second-stage classifier
classify = False
if classify:
    modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    modelc.to(device).eval()

# Eval mode
model.to(device).eval()

# Half precision
half = half and device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()

# Set Dataloader
vid_path, vid_writer = None, None
save_img = True
view_img = True

# Get names and colors
names = load_classes(opt.names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

"""----------------------------------- detect 부분 삽입 --------------------------------------"""
def detection(dataset):
    img = torch.zeros((1, 3, 480, 640), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0 = path, '', im0s

            # save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            boxcoord, boxclass, boxconf = [], [], []
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    boxcoord.append([*xywh])
                    boxclass.append(cls.tolist())
                    boxconf.append(conf.tolist())

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            if view_img:    ##  streaming!
                im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
                cv2.imshow('hello!', im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
                
    return im0, boxcoord, boxclass, boxconf

"""---------------------------- 실행부분 ----------------------------------"""
image_depth=iii()
image_rgb=iii()

##################################### GPS Setting
# mission_Range = 1
# # GPS 생성
# gps = gps(port='/dev/ttyUSB0',
#           brate=115200,
#           gps_filename='cilabReversePath.txt',
#           mode='trace',
#         #   mode='record',/';
#           degree_mode='gps',
#           start_range_num=mission_Range,
#           imu_port='COM3',
#           plot_flag=True)
############################################################

def state_depth(depth_topic):
    image_depth.data=depth_topic.data

def state_rgb(rgb_topic):
    image_rgb.data=rgb_topic.data

def check_depth(y_norm): # ynorm[x,y,w,h] -> y=[x1,y1,x2,y2] bounding box, 9등분점 중 최솟값 return
    image_depth.data=np.frombuffer(image_depth.data,dtype=np.uint8)
    image_depth.data = np.reshape(image_depth.data, (480,640,2))
    #########9개 픽셀 거리 중 최솟값, 중앙 픽셀 return
    box_point=[]
    for i in range(3):
        for j in range(3):
            x_point=y_norm[0]+y_norm[2]*(0.5)*(j-1)
            y_point=y_norm[1]+y_norm[3]*(0.5)*(i-1)
            x_point=int(x_point*640)
            y_point=int(y_point*480)
            box_point.append((image_depth.data[y_point][x_point][0]+image_depth.data[y_point][x_point][1]*256)/1000) ###unit: m
            if box_point[3*i+j]<0.3:
                box_point[3*i+j]=10 ######가끔 0.3 이하로 튀는 값 (10m)로 처리
    yyy=int(y_norm[1]*640)
    xxx=int(y_norm[0]*640)

    return min(box_point), xxx, yyy

def obstacle_decision(distance, obs_x): ######피해야하면 True return
    safety_distance=0.8  ##  중앙으로부터 거리
    a=abs(obs_x-320)/462.139
    a=distance*math.sin(math.atan(a))
    try:
        if a>safety_distance:
            return False
        return True
    except:
        return False

def think():
    rospy.init_node('think')
    rate = rospy.Rate(20)
    rospy.Subscriber('/camera/color/image_raw', iii, state_rgb, queue_size=1) ####size : 640*480*3
    rospy.Subscriber('/camera/depth/image_rect_raw', iii, state_depth, queue_size=1) ####size : 480*640*2
    while not rospy.is_shutdown():
        rate.sleep()
        image_rgb.data=np.frombuffer(image_rgb.data,dtype=np.uint8)
        realtime_img = np.reshape(image_rgb.data, (480,640,3)) ## Original img sz: (480,640,3)
        dataset = LoadImages_change(realtime_img, img_size=imgsz)
        with torch.no_grad():
            img, coord, boxclass, conf = detection(dataset)
        ##########장애물 목록 print
        obstacle=[]
        for i in range(len(coord)):
            try:
                obstacle_distance, obstacle_x, obstacle_y = check_depth(coord[i])
                obstacle_class = boxclass[i]
                obstacle.append([obstacle_decision(obstacle_distance,obstacle_x),obstacle_class,obstacle_distance,obstacle_x,obstacle_y])
            except:
                pass
        print('[obstacle?, class, distance, x, y] : ',obstacle)
        #########################

        ####움직이는부분~~##########
        P=0.3                   ### path planning 정도
        safety_distance=0.5     ### 안전거리
        twist.linear.x=0.1      ### 직진 속도(m/s)
        angular_path=0
        angular_obstacle=0  
        obstacle_coefficient=3    ### 장애물 피하는 정도

        ########angular_path value setting
        # Range, STEER = gps.gps_tracer()
        # if type(STEER) == int:
        #     angular_path=-STEER/1000

        #########angular_obstacle value setting
        for i in obstacle:      ### 0:obstacle?, 1:class, 2:distance, 3:x, 4:y
            if i[0]:
                if i[1]:
                    if i[1]<safety_distance:
                        twist.linear.x=0
                        angular_path=0
                        angular_obstacle=0
                        break
                if i[3]>320:
                    angular_obstacle+=obstacle_coefficient/(i[2]-safety_distance)
                else:
                    angular_obstacle-=obstacle_coefficient/(i[2]-safety_distance)
        if angular_obstacle>1:###너무 크면 안되니깐
            angular_obstacle=1
        if angular_obstacle<-1:
            angular_obstacle=-1

        twist.angular.z=P*angular_path+(1-P)*angular_obstacle/10
        publisher_action.publish(twist)
        ###########################
        # rate.sleep()

if __name__ == '__main__':
    try:
        think()
        
    except rospy.ROSInterruptException:
        pass