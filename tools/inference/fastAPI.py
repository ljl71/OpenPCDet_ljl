import hashlib
import json
import time
import uvicorn
from fastapi import FastAPI, Request, APIRouter
from typing import Optional, List
from tools.inference.inference_nms import LoadModel
from tools.inference.inference_nms_pvrcnnpp_sq import LoadModel_pvrcnnpp_sq
from tools.inference.inference_nms_pvrcnnpp_ld import LoadModel_pvrcnnpp_ld
import requests
import sys, os
from urllib.parse import urlparse, unquote
sys.path.append(os.getcwd())
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(
    title="FastAPI Predict Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redocs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许访问的源
    allow_credentials=True,  # 支持 cookie
    allow_methods=["*"],  # 允许使用的请求方法
    allow_headers=["*"],  # 允许携带的 Headers
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    # token: str = Field(..., example='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c', title='JWT')
    pcURL: str = Field(..., example="http://molardata.com/1.pcd", title="预测图片URL")
    shape: Optional[int] = None
    model: Optional[str] = None
    thresh: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    # Initialize the pytorch model
    model = LoadModel()
    ld_model = LoadModel_pvrcnnpp_ld()
    sq_model = LoadModel_pvrcnnpp_sq()


    # add model and other preprocess tools too app state
    app.package = {"model": model, "ld_model": ld_model, "sq_model": sq_model}

def get_filename(url: str) -> str:
    # 解析 URL，拿到路径部分并做 URL 解码
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # 直接从路径获取文件名
    filename = os.path.basename(path)

    # 若路径以斜杠结尾导致没有文件名，可做兜底处理
    if not filename:
        parts = [p for p in path.split('/') if p]
        filename = parts[-1] if parts else ''
    return filename

def downloaderPointCloud(url, hashurl, time=0):
    try:
        # resp = requests.get(url, stream=True, timeout=5).raw
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            # urlContent = resp.read()
            path = "/workspace/OpenPCDet/tools/inference/pointcloud/" + hashurl
            fp = open(path, "wb")
            fp.write(resp.content)
            fp.close()
            # with open(path, 'wb') as f:
            #     f.write(urlContent)
            # image = np.asarray(bytearray(resp.read()), dtype="uint8")
            # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return path
    except Exception as e:
        print(e)
        if time < 3:
            time += 1
            return downloaderPointCloud(url, hashurl, time)
        else:
            return False

def getUUID(url):
    if isinstance(url, str):
        url = url.encode("utf-8")
    m = hashlib.md5()
    m.update(url)
    return m.hexdigest()

@app.post("/smart-tool/detection3d")
async def cat_kpt_inference(request: Request, body: InferenceInput):
    filename = get_filename(body.pcURL)
    hashPath = getUUID(body.pcURL)
    hashPath = hashPath + "." + filename.split(".")[-1]
    path = downloaderPointCloud(body.pcURL, hashPath)
    if body.shape:
        shape = body.shape
    else:
        shape = 4
    
    if body.model not in list(app.package.keys()):
        body.model = 'model'

    result = await app.package[body.model].inference(path, shape)
    return {"code": 200, "data": result}


if __name__ == "__main__":
    # try:
    #     port = sys.argv[1]
    # except:
    #     port = 5000
    uvicorn.run("fastAPI:app", host="0.0.0.0", port=8000)
