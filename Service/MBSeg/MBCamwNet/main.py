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
from Common import Tools
from Service.MBSeg.MBCamwNet.MBCamwNetWrapper import MBCamwNetWrapper

# fastapi
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = MBCamwNetWrapper()
async_lock = Lock()


class MBCamwNetController:
    @staticmethod
    @app.post("/run/mb/seg/camw")
    async def run(file: fastapi.UploadFile):
        # save to tmp, use PIL to decode
        temp_file = tempfile.NamedTemporaryFile(suffix=".png")
        temp_file.write(await file.read())

        try:
            # generate cache path
            async with async_lock:
                now_time = Tools.generate_strftime()
                cache_path = os.path.join(AppConfig.CACHE_PATH, "mb-seg-camw", now_time)
                os.makedirs(cache_path, exist_ok=True)

            code, error_str = model.infer(cache_path, temp_file.name)
            if code:
                package_name = Tools.generate_package(cache_path, os.path.join(cache_path, model.result_name))
                return Tools.stream_response(package_name)
            else:
                logging.exception(error_str)
                return JSONResponse(
                    status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        'msg': error_str
                    }
                )
        except Exception as e:
            logging.exception(traceback.format_exc())
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    'msg': 'error in process'
                }
            )


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=9003)
