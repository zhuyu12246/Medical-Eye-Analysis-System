# encoding=utf-8
import datetime
import os
import tempfile
import traceback
import logging
import uuid

import cv2
import fastapi
import uvicorn
from asyncio import Lock
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from Common.AppConfig import AppConfig
from Service.MBEnhance.EnhanceWrapper import EnhanceWrapper

# fastapi
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = EnhanceWrapper()
async_lock = Lock()


class MBEnhanceController:
    @staticmethod
    @app.post("/run/mb/enhance")
    async def run(file: fastapi.UploadFile):
        # save to tmp, use PIL to decode
        temp_file = tempfile.NamedTemporaryFile(suffix=".png")
        temp_file.write(await file.read())

        try:
            out = model.infer(temp_file.name)
            async with async_lock:
                now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                now_uuid = str(uuid.uuid1())
                save_path = os.path.join(AppConfig.CACHE_PATH, "mb-enhance", now_time)
                os.makedirs(save_path, exist_ok=True)
                save_name = os.path.join(save_path, now_uuid + ".png")
                cv2.imwrite(save_name, out)

                return fastapi.responses.FileResponse(save_name)
        except Exception as e:
            logging.exception(traceback.format_exc())
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    'msg': 'error in process'
                }
            )


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=9001)
