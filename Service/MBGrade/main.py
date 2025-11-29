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
from Service.MBGrade.MBGradeWrapper import MBGradeWrapper

# fastapi
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = MBGradeWrapper()
async_lock = Lock()


class MBGradeController:
    @staticmethod
    @app.post("/run/mb/grade")
    async def run(file: fastapi.UploadFile):
        # save to tmp, use PIL to decode
        temp_file = tempfile.NamedTemporaryFile(suffix=".png")
        temp_file.write(await file.read())

        try:
            # generate cache path
            async with async_lock:
                now_time = Tools.generate_strftime()
                cache_path = os.path.join(AppConfig.CACHE_PATH, "mb-grade", now_time)
                result_path = os.path.join(cache_path, model.result_name)
                os.makedirs(cache_path, exist_ok=True)
                os.makedirs(result_path, exist_ok=True)
            code, error_str = model.infer(result_path, temp_file.name)
            if code:
                package_name = Tools.generate_package(cache_path, result_path)
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
    uvicorn.run(app=app, host="0.0.0.0", port=9004)
