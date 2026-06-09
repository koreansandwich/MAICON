#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# jetson to 대시보드 이미지 파일 전송 예제

import requests
from pathlib import Path
import json

def send_dashboard_image(image_path):
    image_file = Path(image_path)

    if not image_file.exists():
        print(f"이미지 파일 없음: {image_file}")
        return

    try:
        print(f"이미지 전송 중: {image_file.name}")

        with open(image_file, "rb") as f:
            files = {
                'file': (image_file.name, f, 'image/jpeg')
            }

            response = requests.post(
                'http://58.229.150.23:5000/img/dashboard/fire_building',
                files=files,
                timeout=10
            )

        print("이미지 서버 응답:", response.text)

    except Exception as e:
        print("이미지 전송 실패:", e)

# # 1. 상위 JSON 불러오기
# with open("../1X4F.json", "r") as f:
#     data = json.load(f)

# sector_str = data["fire_buildings"][0]          # 예: "sector2"
# sector_num = int(sector_str.replace("sector",""))

# # 2. 파일명 자동 생성
# filename = f"1X4F_sector{sector_num}.jpg"

# # 3. 전송 함수 호출
# send_dashboard_image(filename)

