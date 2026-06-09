# -*- coding: utf-8 -*-


import cv2
import datetime

# 카메라 장치 열기
pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=20/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# 영상 캡처 루프
# while True:
#     # 프레임 읽기
#     ret, frame = cap.read()
#     if not ret:
#         print("카메라에서 프레임을 가져올 수 없습니다.")
#         break

#     # 프레임을 화면에 표시
#     cv2.imshow("Video", frame)

#     # 키 입력 대기
#     key = cv2.waitKey(1) & 0xFF

#     # 'a' 키를 누르면 프레임 캡처하여 저장
#     if key == ord('a'):
#         # 파일 이름에 현재 날짜와 시간을 추가하여 저장
#         filename = datetime.datetime.now().strftime("./checkerboards/capture_%Y%m%d_%H%M%S.png")
#         cv2.imwrite(filename, frame)
#         print("이미지가 저장되었습니다.")

#     # 'q' 키를 누르면 종료
#     elif key == ord('q'):
#         break

# # 자원 해제
# cap.release()
# cv2.destroyAllWindows()




# 사진 찍고 밑에 실행

import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_camera():
    # 체커보드의 차원 정의
    CHECKERBOARD = (5, 8)  # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
    objpoints = []
    # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
    imgpoints = [] 
    
    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
    images = glob.glob('./checkerboards/*.png')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray,
                                               CHECKERBOARD,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # 코너 그리기 및 표시
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # 카메라 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1], None, None)
    
    # 결과 출력
    print("Camera matrix : \n")
    print(mtx)
    print("\ndist : \n")
    print(dist)
    print("\nrvecs : \n")
    print(rvecs)
    print("\ntvecs : \n")
    print(tvecs)
    
    # 캘리브레이션 결과를 파일로 저장
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }
    
    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    
    return calibration_data

def live_video_correction(calibration_data):
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 크기 가져오기
        h, w = frame.shape[:2]
        
        # 최적의 카메라 행렬 구하기
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # 왜곡 보정
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # ROI로 이미지 자르기
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        # 원본과 보정된 이미지를 나란히 표시
        original = cv2.resize(frame, (1280, 720))
        corrected = cv2.resize(dst, (1280, 720))
        combined = np.hstack((original, corrected))
        
        # 결과 표시
        cv2.imshow('Original | Corrected', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 이미 캘리브레이션 파일이 있는지 확인
    if os.path.exists('camera_calibration.pkl'):
        print("Loading existing calibration data...")
        with open('camera_calibration.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        print("Performing new camera calibration...")
        calibration_data = calibrate_camera()
    
    # 실시간 비디오 보정 실행
    print("Starting live video correction...")
    live_video_correction(calibration_data)