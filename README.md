
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
| 팀 인원 | 5명 (전다함, 이태민, 백승민, 주예성, 박진영) |
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
  ![Ultralytics](https://img.shields.io/badge/Ultralytics%20YOLO-00ADEF?style=for-the-badge&logoColor=white)

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
| 전다함 ⭐ | 카메라 캘리브레이션, 차선인식 CV 파이프라인, 객체탐지/화재탐지 공동 참여 |
| 이태민 | SLAM, ArUco 마커 기반 미션 판단 및 모드 전환 |
| 백승민 | PID 제어, 주행 제어 전반 |
| 주예성 | 화재탐지 YOLO (YOLOv8n 앙상블) |
| 박진영 | 객체탐지 YOLO (YOLOv12s) |

> ⭐ 표시는 본인 기여 파트

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
  ├─ ⭐ 전처리 & 차선인식 (전다함)
  │   캘리브레이션 보정 → BEV 변환 → 색상 필터
  │   → Canny Edge → HoughLinesP → Sliding Window → 2차 다항식 피팅
  │
  ├─ PID 제어 (백승민)
  │   횡방향 오차 + heading 오차 → PID → Feed Forward → linear/angular speed
  │
  ├─ ArUco 감지 (이태민)
  │   마커 ID 인식 → 모드 전환 → 구역별 이미지 캡처 트리거
  │
  └─ 위치 추정 (백승민, 이태민)
      ArUco 앵커 기준 Dead Reckoning → 차량 좌표 계산

[병렬 프로세스: object_detector.py] ⭐ 전다함 + 박진영
  yolo_images 폴더 실시간 감시
  → YOLOv12s 추론 (7종 객체)
  → Confidence/크기 필터링
  → Alpha/Bravo/Charlie 구역별 객체 집계
  → JSON 업데이트 + 대시보드 서버 전송

[화재탐지: fire_detect.ipynb] ⭐ 전다함 + 주예성
  YOLOv8n → 9개 섹터 건물 감지
  → 다중 Confidence 앙상블 (0.25 / 0.20 / 0.15)
  → 화재 섹터 판별 + 사후 검증
  → JSON 저장 + 화재 이미지 서버 전송

[대시보드 서버: 58.229.150.23:5000]
  /dashboard_json         → 임무 결과 JSON (객체 탐지 + 화재 섹터)
  /img/dashboard/fire_building → 화재 건물 이미지
```

---

## 🔍 수행 프로세스

### ⭐ 1. 카메라 캘리브레이션 (`calibration.py`)

체커보드를 다양한 각도에서 촬영하여 카메라 내부 파라미터를 추출하고 `.pkl`로 저장. 매 프레임 실시간 왜곡 보정에 활용.

- 체커보드 기반 캘리브레이션 (5×8 내부 코너)
- `cv2.calibrateCamera()`로 카메라 행렬 및 왜곡 계수 추출
- `initUndistortRectifyMap()`으로 보정 맵 사전 계산 → 실시간 `remap()` 적용으로 연산 최적화
- 캘리브레이션 결과를 `camera_calibration.pkl`로 저장하여 주행 시 자동 로드

---

### ⭐ 2. 차선인식 컴퓨터비전 파이프라인 (`for_real_lane_follower_PID.py`)

Jetson Nano의 하드웨어 제약 속에서 **실시간 10fps** 차선 인식을 구현. 대회장 조명 변화에 강건한 전처리와 Sliding Window 기반 차선 추적을 직접 설계.

#### 전처리 파이프라인

```
원본 프레임 (1280x720)
  → 캘리브레이션 보정 (remap, 왜곡 제거)
  → BEV(Bird's Eye View) 변환 (원근 → 정사영 시점)
      src: [[230,150], [0,480], [410,150], [640,480]]
      dst: [[0,0], [0,480], [350,0], [350,480]]
  → HLS 색공간 변환
  → 노란색 마스킹 제거 (H:0~150, S:0~255, L:0~255)
  → 밝기 정규화: L채널 평균 → 128로 보정 (strength=0.5)
  → 흰색 차선 추출 (L:210~255)
  → Grayscale → 이진화 (THRESH_BINARY)
  → Canny Edge Detection (threshold: 40, 150)
  → HoughLinesP (각도 필터: theta_min=15° 이상만 유효 차선으로 인정)
```

#### Sliding Window 차선 추적

```
hough_img (차선 픽셀)
  → histogram으로 차선 중심점 추정
  → 30개 윈도우 중 하위 8개 활용 (속도/정확도 균형)
      window_height = BEV_HEIGHT / nwindows
      margin = 80px (윈도우 너비)
  → 각 윈도우 내 흰 픽셀 중심 → 차선 중심점 갱신
  → x, y 좌표 누적 → np.polyfit(y, x, 2)로 2차 다항식 피팅
  → keep_largest_vertical_component()로 노이즈 컴포넌트 제거
      (2차 곡선과의 평균 거리가 가장 작은 컴포넌트 선택)
```

#### 오차 계산 및 제어 신호 생성

```python
# Lookahead 기반 횡방향 오차
lookahead_far  = 480 - (60 + 200 * linear_speed)   # 원거리 (weight 0.6)
lookahead_near = 480 - (200 * linear_speed)          # 근거리 (weight 0.4)
lat_norm = -(polyval(fit, lookahead) - center_line) / BEV_CENTERLINE

# Heading 오차
heading_rad = -atan(2*a*y + b)   # 2차 곡선 접선각

# 최종 제어 오차
combined_error = heading_rad * heading_weight + lat_norm * lat_weight
→ PID 출력 + Feed Forward (곡률 기반) → angular_speed
```

- 코너/직선 모드에서 PID 게인(kp, ki, kd) 및 lookahead 파라미터 자동 전환
- EMA(Exponential Moving Average) 필터로 오차 신호 스무딩 (alpha=0.8)
- 최대 angular speed 클리핑 (max=2.0 rad/s)

---

### ⭐ 3. 객체탐지 YOLO (`object_detector.py`) — 박진영과 공동

7종 군사 객체 실시간 탐지 후 구역별로 분류하여 임무 결과 JSON 생성.

| 클래스 | Box | Car | Enemy | Hazmat | Missile | Mortar | Tank |
|--------|-----|-----|-------|--------|---------|--------|------|
| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |

- YOLOv12s 모델 (CUDA 가속, imgsz=640)
- `yolo_images` 폴더 실시간 감시 → 새 이미지 감지 시 자동 추론
- 필터링: Confidence ≥ 0.5 + BBox 면적 ≥ 900px²
- Alpha / Bravo / Charlie 구역별 ArUco ID 매핑으로 객체 집계
- 동일 클래스 다중 탐지 시 confidence 누적 합산으로 최종 클래스 결정
- 임무 결과 JSON 업데이트 + 대시보드 서버 자동 전송

---

### ⭐ 4. 화재탐지 (`fire_detect.ipynb`) — 주예성과 공동

9개 섹터 건물 중 화재 발생 건물 탐지 후 JSON 저장.

**데이터 수집 및 전처리**
- 객체 4,279장 + 화재 237장 (직접 포토샵으로 화재 이미지 증강)
- 어그멘테이션: Mosaic, Brightness, Rotation, Translation 등
- Hard Negatives (Tank, Car 등 오탐지 취약 객체) 집중 학습
- 실제 평가장 데이터를 Test set으로 활용하여 도메인 갭 최소화

**모델 선택**

| 구분 | CNN | Transformer |
|------|-----|-------------|
| 강점 | 속도 | 정확성 |
| 정보처리 | Local Patterns | Global Relations |
| 학습 난이도 | 적은 데이터로 가능 | 초기 대용량 데이터 필요 |
| 선택 | ✅ **YOLOv8n** | — |

**추론 파이프라인**
```
테스트 이미지
  → YOLOv8n 추론 (다중 Confidence: 0.25 / 0.20 / 0.15)
  → 9개 건물 감지 확인 (미달 시 낮은 Conf로 재시도)
  → 섹터 할당 (좌표 기반 규칙 + 사후 검증)
  → 화재 후보 앙상블 (3개 Conf 결과 합산 → 상위 2개 선택)
  → 최종 화재 섹터 확정 → JSON 저장 + 이미지 서버 전송
```

- 최종 mAP50: **0.992**

---

## 🧩 주요 시행착오

- **조명별 레일 색 필터링**: 대회장 조명 환경에 따라 레일 색이 달라져 HLS 색상 필터 파라미터를 반복 조정. 단순 RGB 필터로는 대응이 어려워 밝기 정규화 로직을 직접 설계
- **엣지 환경 경량화**: Jetson Nano에서 실시간 10fps를 유지하면서 CV 파이프라인과 YOLO 추론을 동시에 돌리는 것이 난관. 모델 경량화(YOLOv8n)와 파이프라인 연산 최적화로 대응
- **주행 중 객체탐지**: 전차가 달리는 상태에서 촬영된 이미지로 객체를 탐지하다 보니 블러, 각도 변화 등으로 인식률 저하. 다양한 어그멘테이션과 Hard Negatives 학습으로 보완

---

## 🧭 회고

대회는 3일간 진행되었으나, 팀원들이 약 3주에 걸쳐 별도로 모여 사전 준비를 진행했다. 그만큼 준비에 많은 공을 들였지만, 연습 환경에 지나치게 최적화된 나머지 실전에서의 빠른 환경 변화에 유연하게 대응하지 못한 것이 아쉬움으로 남는다.

- 모든 파트를 완벽하게 구현하려다 전체 완성도가 낮아진 것이 아쉬움 — 선택과 집중의 중요성을 체감
- 연습 환경 최적화에 집중한 결과, 실전 환경 변화(조명, 레일 색 등)에 대한 빠른 적응이 부족했음
- Jetson Nano 엣지 환경에서 실시간 CV 파이프라인 최적화 경험
- 조명 변화에 강건한 전처리 설계의 중요성 체감
- 데이터 237장으로 mAP50 0.992 달성 — 데이터 품질과 어그멘테이션의 힘
- 5인 팀 협업에서 JSON 기반 프로세스 간 인터페이스 설계의 중요성 체감
- 제한된 하드웨어에서 모델 경량화와 실시간 성능 균형 설계
