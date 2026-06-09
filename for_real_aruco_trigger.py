# -*- coding: utf-8 -*-
#!/usr/bin/env python
import time, math, os
import cv2
import numpy as np

# OpenCV 버전에 따른 API 차이 호환
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
except AttributeError:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
except AttributeError:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# # ARUCO_PARAMS.adaptiveThreshWinSizeMin = 1       # 더 작은 window
# # ARUCO_PARAMS.adaptiveThreshWinSizeMax = 23      # 더 큰 window
# ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10
# ARUCO_PARAMS.adaptiveThreshConstant = 1         # 기본 7~10 → 1로 극단 민감도

# # --- 모션 블러 환경에서 필수 ---
# ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# ARUCO_PARAMS.minCornerDistanceRate = 0.01

# # --- 작은 마커 & 흐릿한 상황 ---
#ARUCO_PARAMS.minMarkerPerimeterRate = 0.001     # 기본 0.03 → 더 작은 마커 잡음

# # --- 유사 사각형도 인정 ---
# ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.02

# # --- 디텍터의 보수성을 낮춤 ---
# ARUCO_PARAMS.perspectiveRemoveIgnoredMarginPerCell = 0.1
# ARUCO_PARAMS.perspectiveRemovePixelPerCell = 4   # 기본 8 → 4로

class ArucoDetector(object):
    def __init__(self):
        pass


    def scale_aruco_corners(self, corners, old_size, new_size):
        #corners가 None, [], () 같은 경우 바로 반환
        if corners is None or len(corners) == 0 or not isinstance(corners, (list, np.ndarray)):
            return corners

        old_w, old_h = old_size
        new_w, new_h = new_size

        sx = new_w / old_w
        sy = new_h / old_h

        # ↳ numpy array로 강제 변환 (tuple → array 문제 해결)
        corners_np = np.array(corners, dtype=np.float32)

        # 기존 아루코 구조 reshape: (N,1,4,2) → (N,4,2)
        corners_np = corners_np.reshape(-1,4,2)

        # 스케일 적용
        corners_np[:,:,0] *= sx
        corners_np[:,:,1] *= sy

        # 다시 (N,1,4,2) 형태로 되돌리기
        return corners_np.reshape(-1,1,4,2)

    def detect_ids_and_corners(self, bgr_img):
        """ bgr_img에서 (id, center(x,y), 면적) 리스트 반환 """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        #corners = self.scale_aruco_corners(corners, (1280,720), (640,480))


        results = []
        if ids is not None:
            ids = ids.flatten()
            for c, i in zip(corners, ids):
                pts = c.reshape(-1, 2)          # 4x2
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                # 간단 면적 근사(사각형 bounding area)
                w = float(np.max(pts[:,0]) - np.min(pts[:,0]))
                h = float(np.max(pts[:,1]) - np.min(pts[:,1]))
                area = abs(w * h)
                results.append({"id": int(i), "center": (cx, cy), "area": area, "corners": c})
        return results, corners, ids

    

class ArucoTrigger(object):
    """
    - LANE_FOLLOW 상태에서만 마커를 감지해 트리거.
    - 새 ID 등장(혹은 동일 ID의 n번째 등장) + 쿨다운 충족 시 pending_actions(리스트) 세팅.
    - step()에서 리스트의 액션들을 순차 실행(제자리 회전) 후 다시 LANE_FOLLOW 복귀.
    """
    def __init__(self, M):
        # 규칙 테이블: { id: { nth: action or [actions...] } }
        # action: ("right"| "left" | "turn" | "turn1", degrees)
        #  - 요청사항: id=0 첫 등장 -> 오른쪽 90도 후 즉시 왼쪽 90도
        self.rules = {
            #should change here
            1: {1: ("right", 0)},      # 연속 액션rtrt
            2: {1: ("right", 0)},
            3: {1: ("left", 0)}, 
            4: {1:("right", 0)},
            5: {1: ("left", 1)},
            6: {1:("left", 0)},
            7: {1:("left", 0)},
            8: {1:("left", 0)},
            9: {1:("left", 0)},
            10: {1:("left",0)},
            11: {1: ("left", 0)},
            12: {1: ("left", 0)},
            13: {1: ("left", 0)}
            # 필요 시 계속 추가
        }

        self.detector = ArucoDetector()

        self.mode = "LANE_FOLLOW"
        self.pending_actions = []
        self.seen_counts = {}          # {id: nth}

        self.M = M
        self.x_bev = -1
        self.y_bev = -1
        self.y_origin = -1
        self.x_origin = -1
        self.dist = 480 #must change here in real competition
        self.ready_count=0
        self.aruco_center = (0,0)

        self.aruco_id = -1
        self.mode_nth = -1
        self.giving_mode = "pre"
        self.giving_mode = "pre"
        # 🔻 전역 쿨다운 제거하고 per-ID로 교체
        # self.last_trigger_time = 0.0
        # self.trigger_cooldown = 5.0

        # ✅ 기본(디폴트) 쿨다운 + 마커별 오버라이드
        self.cooldown_default = 5.0      # 기본값(초)
        self.cooldown_per_id = {
            #should change here
            1: 20,
            2: 20,   # id=2는 4초
            3: 20,   # id=3은 6초
            4: 20,  # id=4는 3초
            5: 20,
            6: 20,
            7: 20,
            8: 20,
            9: 20,
            10: 20,
            11: 20,
            12: 20,
            13: 20
            # 필요에 따라 추가/수정
        }
        self.last_trigger_times = {}      # {id: last_time}
        self.linear_speed = 0.12
    
        self.required_consecutive = 0 #origin 3 #should change here. 지금은 보자마자 모드 바꾸는 상태. 여기에 1더한 횟수만큼 봐야 모드 바뀜
        self._consec = {}
        #should change here
        self.min_area = 10
        self.min_y = 10.0 #origin 60
        self.max_y = 440.0
        self.for_calc_rate_of_dist_and_count = 0

    def convert_point_to_bev(self, point_xy):
        original_x, original_y = point_xy
        src_point = np.array([[[original_x, original_y]]], dtype=np.float32)
        bev_point = cv2.perspectiveTransform(src_point, self.M)
        return (bev_point[0][0][0], bev_point[0][0][1])
    
    def _gate(self, det):
        #should change here
        # if(det["id"] in {0, 2, 3, 4, 5}):
        #     area_ok = det["area"] >= self.min_area*0.3
        area_ok = det["area"] >= self.min_area
        self.aruco_center = det["center"]
        y = det["center"][1]
        y_ok = (y >= self.min_y) and (y <= self.max_y)


        return area_ok and y_ok

    def observe_and_maybe_trigger(self, bgr_img):
        if self.mode != "LANE_FOLLOW":
            return

        now = time.time()

        # 🔻 (삭제) 전역 쿨다운 체크는 제거
        # if (now - self.last_trigger_time) < self.trigger_cooldown:
        #     return

        dets, _, _ = self.detector.detect_ids_and_corners(bgr_img)
        if not dets:
            self._consec = {}
            return

        dets = [d for d in dets if self ._gate(d)]
        if not dets:
            self._consec = {}
            return

        det = max(dets, key=lambda x: x["area"])
        mid = det["id"]
        # 연속 프레임 카운트 갱신
        self._consec[mid] = self._consec.get(mid, 0) + 1
        for k in list(self._consec.keys()):
            if k != mid:
                self._consec[k] = 0
        # if mid in (0, 2, 3, 4, 5): #aruco 3 is strange. So, in real competition, dont use this code
        #     if self._consec[mid] < self.required_consecutive*0.3:
        #         return
        # else:
        if self._consec[mid] < self.required_consecutive:
                return

        # ✅ 여기서 '해당 마커'의 쿨다운만 확인
        last = self.last_trigger_times.get(mid, 0.0)
        cooldown = self.cooldown_per_id.get(mid, self.cooldown_default)
        if (now - last) < cooldown:
            #print(now-last, "cool: ", cooldown)
            return
        #print(mid)
        # 등장 횟수 → 규칙 매칭
        nth = self.seen_counts.get(mid, 0) + 1
        self.seen_counts[mid] = nth

        if (mid in self.rules) and (nth in self.rules[mid]):
            actions = self.rules[mid][nth]
            if isinstance(actions, tuple):
                actions = [actions]
            self.pending_actions = list(actions)

            self.mode = "EXECUTE_ACTION_READY"
            self.aruco_id = mid
            self.mode_nth = nth
            x = det["center"][0]
            y = det["center"][1]
            #print("y origin: ", y)
            self.x_bev, self.y_bev = self.convert_point_to_bev((x, y))
            self.y_origin = y
            self.x_origin = x
            # ✅ 트리거 타임스탬프는 해당 마커 id로 기록
            self.last_trigger_times[mid] = now

            self._consec = {}


    def change_linear_speed(self, speed):
        self.linear_speed = speed

    def step(self):
        """
        EXECUTE_ACTION 상태일 때 호출하여
        pending_actions에 쌓인 액션들을 차례대로 실행.
        모두 끝나면 LANE_FOLLOW로 복귀.
        for each id, they have different time that they have to wait before excute
        """
        
        if self.mode == "EXECUTE_ACTION_READY" and self.ready_count==0:
            #should change here 대회장 아루코 형태 보고 바꾸기
            if self.aruco_id == 1:
                self.ready_count = 1 #원래 썼으나, 그냥 lane detect의 turn track 지연 방식 쓰는 것이 합리적으로 보임.
                self.giving_mode = "a"

            elif self.aruco_id == 2:
                if(self.mode_nth == 1):
                    # self.ready_count = max(self.dist/(36*self.linear_speed/0.12), 20)-8-20*(11/36)
                    self.ready_count = 1
                    self.giving_mode = "b"

            elif self.aruco_id == 3:
                 #after 10prame, car go, 1prame = 0.1sec
                if(self.mode_nth == 1):
                    self.ready_count = 1 #dont set readycount 0
                    self.giving_mode = "c"
                    
            elif self.aruco_id == 4:
                if self.mode_nth == 1:
                    self.ready_count = 1
                    self.giving_mode = "d"

            elif self.aruco_id == 5:
                self.ready_count = 1
                self.giving_mode = "e"

            elif self.aruco_id == 6:
                self.ready_count = 1
                self.giving_mode = "f"

            elif self.aruco_id == 7:
                self.ready_count = 1
                self.giving_mode = "g"

            elif self.aruco_id == 8:
                self.ready_count = 1
                self.giving_mode = "h"
                
            elif self.aruco_id == 9:
                self.ready_count = 1
                self.giving_mode = "i"

            elif self.aruco_id == 10:
                self.ready_count = 1
                self.giving_mode = "j"

            elif self.aruco_id == 11:
                self.ready_count = 1
                self.giving_mode = "k"

            elif self.aruco_id == 12:
                self.ready_count = 1
                self.giving_mode = "L"

            elif self.aruco_id == 13:
                self.ready_count = 1
                self.giving_mode = "m"
                
            else:
                self.mode = "EXECUTE_ACTION"
                self.aruco_id = -1
                self.mode_nth = -1

        if(self.ready_count>0):
            self.ready_count-=1
            print("id: ", self.aruco_id, "nth: ", self.mode_nth, "count: ", self.ready_count)
            if(self.ready_count<=0):
                self.mode = "EXECUTE_ACTION"
                self.aruco_id = -1
                self.mode_nth = -1
                self.ready_count=0

        direction, deg = 0, 0

        if self.mode == "EXECUTE_ACTION" and self.pending_actions:
            # 안전 정지
            #rospy.sleep(0.00)
            # 맨 앞 액션 수행
            direction, deg = self.pending_actions.pop(0)
            #self._rotate_in_place(direction, deg, ang_speed=1.0)
            print("pop")
            # 남은 액션이 없으면 복귀
            if not self.pending_actions:
                self.mode = "LANE_FOLLOW"
            # if(self.giving_mode in {2, 3, 4, 5, 61, 71, 8, 9}):
            #     self.giving_mode = self.giving_mode

        # if(self.giving_mode == 2):
        #     #rospy.signal_shutdown("detected")
        #     self.for_calc_rate_of_dist_and_count += 1
        #     print("check count: ", self.for_calc_rate_of_dist_and_count)
        #if self.giving_mode==2:
            #rospy.signal_shutdown("detected")
        #key = cv2.waitKey(1) & 0xFF

        # 스페이스바(32) 누르면 저장
        # if key == 32:
        #     print("!!!!count!!!!!: ", self.for_calc_rate_of_dist_and_count)
        #여기 코드로 아루코 보고 얼마 뒤에 목표지점 가는지 계산하려했으나, 생각해보니 그냥 카메라 사진 전송 딜레이 예상치(THETA_FRAME_DELAY)에 이론상 걸리는 시간
        #빼면 될듯
        
        return self.giving_mode, direction, deg, len(self.pending_actions), self.x_origin, self.y_origin
    #after giving mode: give mode after ready count
    #giving mode: give mode before ready count
