# 🛡️ 2025 국방 AI 경진대회 (MAICON) - 8위 입상

본 리포지토리는 2025년 국방 AI 경진대회에서 수행한 **무한궤도 전차 자율주행 + 복합 임무 수행** 프로젝트의 구조와 흐름을 정리한 문서입니다.

---

## 🧠 Overview

| 항목 | 내용 |
|------|------|
| 주최 | 대한민국 국방부 |
| 과제 | 무한궤도 전차의 자율주행, 피아식별, 화재탐지 복합 임무 실시간 완수 |
| 개발 환경 | NVIDIA Jetson Nano / JupyterLab / CUDA |
| 팀명 | TEAM 공군돌이 |
| 팀 인원 | 5명 |
| 최종 성적 | 🏅 8위 입상 |

---

## 🧰 Tech Stack

- **Language & Framework**
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

- **Computer Vision & Data**
  ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
  ![Ultralytics](https://img.shields.io/badge/Ultralytics-00ADEF?style=for-the-badge&logo=ultralytics&logoColor=white)

- **Hardware**
  ![Jetson](https://img.shields.io/badge/Jetson_Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
  ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

---

## 📌 과제 개요

```
과제: 전차의 자율 주행 + 피아식별 + 화재탐지 복합 임무 실시간 완수

도전 환경
├─ Real-Time 임무환경: 최단 시간 내 임무 완수를 위한 실시간 탐지 및 제어
├─ 하드웨어 제약: Jetson Nano 환경에서 구동 가능한 경량화 모델 필수
└─ 차량 특성: 무한궤도 차량의 특성을 반영한 제어 및 회전 로직 필요

문제 해결 논리 구조
├─ 인지: 차선/경로 인식, 객체 검출 (YOLO, BEV, Color, Sliding Window)
├─ 계획/판단: ArUco 기반 미션 구간 판단 및 행동 결정
└─ 제어: PID/Feed Forward Control, 주행 모드 다변화 (직선, 코너)
```

---

## 👥 팀 구성 및 역할 분담

| 이름 | 역할 |
|------|------|
| 전다함 | 카메라 캘리브레이션, 차선인식 CV 파이프라인 |
| 전다함 + 박진영 | 객체탐지 YOLO (YOLOv12s) |
| 전다함 + 주예성 | 화재탐지 YOLO (YOLOv8n 앙상블) |
| 이태민 | ArUco 트리거/모드전환, PID 제어, Dead Reckoning |
| 백승민 | 시스템 통합 |

---

## ⚙️ 전체 시스템 구조

```
[하드웨어]
Jetson Nano + CSI 카메라 + TikiMini (무한궤도 전차)
        ↓
[진입점: main.py]
카메라 프레임 캡처 (1280x720, 10fps)
→ camera_callback() 호출
→ 모터 RPM 계산 후 TikiMini 전달
        ↓
[핵심 로직: for_real_lane_follower_PID.py]
  ├─ 캘리브레이션 보정 → BEV 변환 → 색상 필터 → Canny → HoughLinesP
  ├─ Sliding Window → 2차 다항식 피팅 → PID 제어
  ├─ ArUco 감지 → 모드 전환 → 이미지 캡처 트리거
  └─ ArUco 앵커 + Dead Reckoning → 차량 좌표 계산

[병렬 프로세스: object_detector.py]
  yolo_images 폴더 실시간 감시
  → YOLOv12s 추론 (7종 객체)
  → Alpha/Bravo/Charlie 구역별 집계
  → JSON 업데이트 + 대시보드 서버 전송

[화재탐지: fire_detect.ipynb]
  YOLOv8n → 9개 섹터 건물 감지
  → 다중 Confidence 앙상블
  → 화재 섹터 판별 → JSON 저장 → 이미지 전송

[대시보드 서버: 58.229.150.23:5000]
  /dashboard_json → 임무 결과 JSON
  /img/dashboard/fire_building → 화재 건물 이미지
```

---

## 🔍 내 기여 파트 상세

### 1. 카메라 캘리브레이션 (`calibration.py`)

체커보드를 다양한 각도에서 촬영하여 카메라 내부 파라미터(카메라 행렬, 왜곡 계수)를 추출하고 `.pkl`로 저장. 매 프레임 실시간 왜곡 보정에 활용.

- 체커보드 기반 캘리브레이션 (5×8 내부 코너)
- `cv2.calibrateCamera()`로 왜곡 계수 추출
- `initUndistortRectifyMap()`으로 보정 맵 사전 계산 → 실시간 `remap()` 적용으로 속도 최적화

---

### 2. 차선인식 컴퓨터비전 파이프라인 (`for_real_lane_follower_PID.py`)

Jetson Nano의 하드웨어 제약 속에서 실시간 차선 인식을 구현. 조명 변화에 강건한 전처리와 Sliding Window 기반 차선 추적을 설계.

**전처리 파이프라인**
```
원본 프레임
  → 캘리브레이션 보정 (remap)
  → BEV(Bird's Eye View) 변환
  → HLS 색공간 변환
  → 노란색 마스킹 제거 + 밝기 정규화 (strength=0.5)
  → 흰색 차선 추출
  → Grayscale → 이진화 → Canny Edge
  → HoughLinesP (각도 필터링)
```

**Sliding Window**
- 30개 윈도우 중 하위 8개 활용 (속도/정확도 균형)
- 2차 다항식 피팅으로 곡선 차선 처리
- 연결 컴포넌트 분석으로 노이즈 제거 (`keep_largest_vertical_component`)
- 가중 lookahead (원거리 0.6 + 근거리 0.4) 기반 횡방향 오차 계산

**오차 계산 및 제어**
```
combined_error = heading_error × heading_weight + lateral_error × lat_weight
→ PID 출력 → Feed Forward 보정 → 모터 angular speed
```

---

### 3. 객체탐지 YOLO (`object_detector.py`) — 박진영과 공동

7종 군사 객체(Box, Car, Enemy, Hazmat, Missile, Mortar, Tank) 실시간 탐지.

- YOLOv12s 모델 (CUDA 가속)
- `yolo_images` 폴더 실시간 감시 → 새 이미지 자동 추론
- Confidence 0.5 + BBox 크기 900px² 이상 필터링
- Alpha/Bravo/Charlie 구역별 객체 집계 후 JSON 저장
- 대시보드 서버로 결과 자동 전송

---

### 4. 화재탐지 (`fire_detect.ipynb`) — 주예성과 공동

9개 섹터 건물 중 화재 발생 건물 탐지.

- 데이터: 객체 4,279장, 화재 237장 (직접 포토샵으로 화재 이미지 증강)
- 어그멘테이션: Mosaic, Brightness, Rotation, Translation 등
- Hard Negatives (Tank, Car 등 취약 객체) 집중 학습
- CNN vs Transformer 비교 후 속도/정확도 균형으로 YOLOv8n 선택
- 다중 Confidence 앙상블 (0.25 / 0.20 / 0.15) + 섹터 사후 검증
- 최종 mAP50: **0.992**

---

## 🧭 회고

- Jetson Nano 엣지 환경에서 실시간 CV 파이프라인 최적화 경험
- 조명 변화에 강건한 전처리 설계의 중요성 체감
- 데이터 237장으로 mAP50 0.992 달성 — 데이터 품질과 어그멘테이션의 힘
- 5인 팀 협업에서 JSON 기반 프로세스 간 통신 설계의 중요성 체감
- 제한된 하드웨어에서 모델 경량화와 성능 균형 설계
