import requests
import json

header = {'Content-Type': 'application/json'}

param = {"processId": "62a40a1cce9eac281c3eebb9", "env": "work"}

for i in range(10):
    r = requests.post(
        'http://127.0.0.1:21300/predict?img_link=https://molar-app-daily.oss-cn-hangzhou.aliyuncs.com/NjMzMmI4NzM0YmI4NjU1NTRhNzgyMmMz/pc/2018-02-04_11-27-05_00100.pcd',
        data=json.dumps(param), headers=header)

    print(r.headers)
