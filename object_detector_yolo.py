import time
import os
import cv2
import shutil
import numpy as np
import json
import requests
from collections import Counter
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self,
                 input_dir="yolo_images",
                 output_img_dir="yolo_images_output",
                 output_coord_dir="yolo_images_coord",
                 model_path="yolov12s_best.pt",
                 img_size=640,
                 original_width=640,
                 original_height=480):

        self.input_dir = input_dir
        self.output_img_dir = output_img_dir
        self.output_coord_dir = output_coord_dir
        self.model_path = model_path
        self.img_size = img_size
        self.original_width = original_width
        self.original_height = original_height
        self.class_id_to_name = { 0: 'Box', 1: 'Car', 2: 'Enemy', 3: 'Hazmat', 4: 'Missile', 5: 'Mortar', 6: 'Tank' }

        # 전 이미지 처리 여부
        self.processed = set()

        # 🧠 추론 결과 저장 딕셔너리
        # key = 파일명(str), value = [[x1,y1,x2,y2,conf,cls], ... ]
        self.inference_output = {}

        # 폴더 초기화
        shutil.rmtree(self.input_dir, ignore_errors=True)
        shutil.rmtree(self.output_img_dir, ignore_errors=True)
        shutil.rmtree(self.output_coord_dir, ignore_errors=True)

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_img_dir, exist_ok=True)
        os.makedirs(self.output_coord_dir, exist_ok=True)

        # 모델 로드
        print("[INFO] YOLO 모델 로드 중...")
        self.model = YOLO(self.model_path)
        self.model.to("cuda")
        print("[INFO] 모델 로드 완료!")

    def filtering(self, detected, aruco_id):
        min_conf = 0.5
        min_size = 900
        filtered = []
        for x1, y1, x2, y2, conf, cls_id in detected:
            if conf >= min_conf and (x2 - x1) * (y2 - y1) >= min_size:
                filtered.append([x1, y1, x2, y2, conf, cls_id])
        return filtered

    def last_inference(self):

        alpha_aruco_id = ("-1-1", "2-1", "2-2", "3-1", "4-1", "4-2", "5-1", "5-2", "6-1", "6-2","6-3", "7-1")
        bravo_aruco_id = ("-1-2", "9-1", "9-2", "9-3")
        charlie_aruco_id = ("7-2","7-3","11-1","12-1", "10-1")

        max_objects_cnt = {i : 1 for l in [alpha_aruco_id, bravo_aruco_id, charlie_aruco_id] for i in l}
        max_objects_cnt["-1-2"] = 2
        max_objects_cnt["2-2"] = 2
        max_objects_cnt["12-1"] = 2


        alpha_objects = []
        bravo_objects = []
        charlie_objects = []

        for img_name in self.inference_output.keys():
            detected = self.inference_output[img_name]
            filtered = self.filtering(detected, img_name)

            filtered = sorted(filtered, key=lambda x: x[4], reverse=True)
            class_dict = {}
            class_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
            for x1, y1, x2, y2, conf, cls_id in filtered:
                if class_cnt[cls_id] % 4 == 0:
                    class_dict[cls_id + 7 * (class_cnt[cls_id] // 4)] = 0
                class_dict[cls_id + 7 * (class_cnt[cls_id] // 4)] += conf
                class_cnt[cls_id] += 1

            for _ in range(max_objects_cnt[img_name]):
                if(len(class_dict) == 0):
                    break
                max_key = max(class_dict, key=class_dict.get)
                if any(t in img_name for t in alpha_aruco_id):
                    alpha_objects.append(max_key % 4)
                elif any(t in img_name for t in bravo_aruco_id):
                    bravo_objects.append(max_key % 4)
                elif any(t in img_name for t in charlie_aruco_id):
                    charlie_objects.append(max_key % 4)
                class_dict.pop(max_key)

        self.json_path = "data.json"

        with open(self.json_path, "r", encoding="utf-8") as f:                         
            data = json.load(f)
        
        def make_entries(obj_list):
            counter = Counter(obj_list)
            result = []
            for cid, cnt in counter.items():
                obj_type = self.class_id_to_name.get(cid, f"class_{cid}")
                result.append({"type": obj_type, "count": cnt})
            return result
        
        data["detection"]["Alpha"] = make_entries(alpha_objects)
        data["detection"]["Bravo"] = make_entries(bravo_objects)
        data["detection"]["Charlie"] = make_entries(charlie_objects)

        # ---------------------------
        # 4) 다시 JSON 파일로 저장
        # ---------------------------
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        try:
            json_content = json.dumps(data, indent=2, ensure_ascii=False)

            files = {
                'file': ('1X4F.json', json_content, 'application/json')
            }

            print(f"서버로 전송 중: {'1X4F.json'} ...")

            response = requests.post(
                'http://58.229.150.23:5000/dashboard_json',
                files=files,
                timeout=10
            )
        except Exception as e:
            print("서버 전송 실패:", e)

    # =====================================================
    # 루프: 새 이미지 감지 → 추론 → 결과 저장
    # =====================================================
    def run(self):
        print(f"[INFO] '{self.input_dir}' 폴더 감시 시작...")

        while True:
            files = [f for f in os.listdir(self.input_dir)
                     if f.lower().endswith((".jpg", ".png"))]

            for img_name in files:
                if img_name in self.processed:
                    continue
                self.processed.add(img_name)

                img_path = os.path.join(self.input_dir, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"[WARN] 불러올 수 없는 이미지: {img_name}")
                    continue

                print(f"[INFO] 새 이미지 감지: {img_name}")

                # 리사이즈 후 YOLO 추론
                resized = cv2.resize(image, (self.img_size, self.img_size))

                start = time.time()
                results = self.model(resized, imgsz=self.img_size, verbose=False)
                infer_ms = (time.time() - start) * 1000
                print(f" └ 추론 시간: {infer_ms:.2f} ms")

                # 결과 이미지 저장
                annotated = results[0].plot()
                out_img_path = os.path.join(self.output_img_dir, img_name)
                cv2.imwrite(out_img_path, annotated)

                # -------------------------------
                # ⭐ inference_output 저장 부분
                # -------------------------------
                detected = []

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    detected.append([x1, y1, x2, y2, conf, cls_id])

                # 딕셔너리에 저장
                img_name = img_name.replace(".jpg", "").replace(".png", "")
                if self.inference_output.get(img_name) is None:
                    self.inference_output[img_name] = []
                self.inference_output[img_name].extend(detected)

                self.last_inference()

                # 좌표 txt 파일 저장
                txt_path = os.path.join(self.output_coord_dir,
                                        img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
                with open(txt_path, "w") as f:
                    for x1, y1, x2, y2, conf, cls_id in detected:
                        cx = int(((x1 + x2) / 2) * self.original_width / self.img_size)
                        cy = int(y2 * self.original_height / self.img_size)
                        f.write(f"{cls_id} {cx} {cy} {conf}\n")

                print(f" └ 추론 결과 저장 완료 → inference_output['{img_name}']")
                print(f" └ 좌표 txt 저장 완료")

                

            time.sleep(0.1)

if __name__ == "__main__":
    obj = ObjectDetector()
    obj.run()