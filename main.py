# coding: utf-8
from tiki.mini import TikiMini
import cv2
import ipywidgets as widgets
from IPython.display import display
import time
from for_real_lane_follower_PID import lane_detect
#import subprocess
#subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "exposure_auto=1"])
#subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "exposure_time_absolute=50"])

CAR_R = 10.5 #수정금지
CAR_RAIL_LENGTH = 23 #수정금지



def frame_to_bytes(frame):
    #_, buf = cv2.imencode('.jpg', frame)
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])  #JPEG Quality 30 
    return buf.tobytes()
    
pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=10/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Camera Open Error")
    exit()


if __name__ == "__main__":

    lane_detect_object = lane_detect(False, True)
    now_time = time.time()
    tiki = TikiMini()
    tiki.set_motor_mode(tiki.MOTOR_MODE_PID)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, -1)

        
        now_time = time.time()

        #frame = frame_to_bytes(frame)
        linear_speed, ang_speed = lane_detect_object.camera_callback(frame)
        ##################### 절대 수정 금지 ##########################
        left_motor_rpm = linear_speed * 50 * 60/CAR_RAIL_LENGTH - ang_speed * CAR_R * 60/CAR_RAIL_LENGTH
        right_motor_rpm = linear_speed * 50 * 60/CAR_RAIL_LENGTH + ang_speed * CAR_R * 60/CAR_RAIL_LENGTH 
        ################################################################
        tiki.set_motor_power(tiki.MOTOR_LEFT, left_motor_rpm)
        tiki.set_motor_power(tiki.MOTOR_RIGHT, right_motor_rpm)

        # 주기 보정
        cycle = 0.1 - (time.time() - now_time)
        if cycle > 0:
            time.sleep(cycle)
        now_time = time.time()
        

        



