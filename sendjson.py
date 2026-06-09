#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# jetson to 대시보드 임무코드.json 전송 예제

import json
import requests


# 🔥 외부 JSON 파일에서 data 읽기
with open("/home/jetson/Workspace/report/1X4F.json", "r") as f:
    data = json.load(f)


json_filename = "1X4F.json"   # 임무코드.json


def send_dashboard(json_obj, filename):
    try:
        json_content = json.dumps(json_obj, indent=2, ensure_ascii=False)

        files = {
            'file': (filename, json_content, 'application/json')
        }

        print(f"서버로 전송 중: {filename} ...")

        response = requests.post(
            'http://58.229.150.23:5000/dashboard_json',
            files=files,
            timeout=10
        )

        print("서버 응답:", response.text)
    
    except Exception as e:
        print("서버 전송 실패:", e)


