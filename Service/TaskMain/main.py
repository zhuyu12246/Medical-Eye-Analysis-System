# encoding=utf-8
import json
import logging
import os
import shutil
import tempfile
import threading
import traceback

import cv2
import numpy

from Common.Tools import convert_2_png
import fastapi
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from Common.AppConfig import AppConfig
from Common import Tools
from typing import List
from Service.TaskMain.Task.TaskQueue import TaskQueue
from Service.TaskMain.Task.TaskManager import TaskManager, TASK_STATUS
from Service.TaskMain.Task.TaskObj import run_service
from Service.TaskMain.Checker.ValidCheck import check_running_options
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from Service.TaskMain.Generate.ResultGenerate import generate_cache_json

logging.getLogger('apscheduler').setLevel(logging.WARNING)
scheduler = AsyncIOScheduler()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

cpf_task_queue = TaskQueue()


class CPFAnalysisGatewayController:
    @staticmethod
    @app.get("/gateway/task/generate")
    async def task_generate(user: str):
        return JSONResponse(
            status_code=fastapi.status.HTTP_200_OK,
            content={"res": Tools.generate_gateway_cache(user)}
        )

    @staticmethod
    @app.post("/gateway/task/post")
    async def task_post(files: List[fastapi.UploadFile], query_key: str = fastapi.Form()):
        """
            常规文件上传接口, 非chunk模式
            受制于文件传输性能，后续增加resize参数
        :param files: formData.append("files", each_file, each_file.name)
        :param query_key: task服务器id
        :return:
        """
        parent_path = os.path.join(AppConfig.CACHE_PATH, query_key)
        if not os.path.exists(parent_path):
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": "cache cleaned"}
            )

        try:
            for file in files:
                # image = cv2.imdecode(numpy.frombuffer(await file.read(), numpy.uint8), cv2.IMREAD_COLOR)
                # resize_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
                save_dst = os.path.join(parent_path, AppConfig.GATEWAY_USER_CACHE, file.filename)
                # cv2.imwrite(save_dst, resize_image)

                with open(save_dst, 'wb') as f:
                    f.write(await file.read())
        except Exception as e:
            logging.exception(traceback.format_exc())
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": "error in convert file"}
            )

        return JSONResponse(status_code=fastapi.status.HTTP_200_OK, content={"res": "ok"})

    @staticmethod
    @app.put("/gateway/task/in")
    async def task_in(query_key: str, running_options: str):
        # BUG, 重复使用一个key入队（对应UI重复点击执行），会导致TaskManager对象被覆盖
        # 30分钟删除时会导致对象丢失，现代码只能保证一轮标准操作的执行
        try:
            json_options = json.loads(running_options)
        except Exception as e:
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": "running options parser failed"}
            )

        # bug fix
        manager: TaskManager = cpf_task_queue.task_query(query_key)
        if manager is not None:
            if manager.task_status == TASK_STATUS.STATUS_RUNNING:
                # reject
                return JSONResponse(
                    status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"res": "Task: {} is running".format(manager.task_key)}
                )
            elif manager.task_status == TASK_STATUS.STATUS_CREATED:
                del cpf_task_queue.task_dict[query_key]
                cpf_task_queue.task_list.remove(query_key)
            else:
                # delete
                cpf_task_queue.task_clean(query_key)

        status, err_msg = check_running_options(json_options)
        if not status:
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": err_msg}
            )

        user_path = os.path.join(AppConfig.CACHE_PATH, query_key, AppConfig.GATEWAY_USER_CACHE)
        service_path = os.path.join(AppConfig.CACHE_PATH, query_key, AppConfig.GATEWAY_SERVICE_CACHE)

        t = threading.Thread(target=run_service, args=(user_path, service_path, json_options,
                                                       cpf_task_queue, query_key))
        task_manager = TaskManager(query_key, t)
        cpf_task_queue.task_in(query_key, task_manager)
        return JSONResponse(status_code=fastapi.status.HTTP_200_OK, content={"res": "ok"})
    
    @staticmethod
    @app.get("/gateway/task/query")
    async def task_query(query_key: str):
        task_manager = cpf_task_queue.task_query(query_key)
        if task_manager is None:
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": "task cleaned, try re-created"}
            )

        task_status = task_manager.get_task_status()
        has_error, error_msg = task_manager.get_error_msg()
        progress = task_manager.get_progress()
        if not has_error:
            error_msg = task_manager.get_next_running_msg()

        return JSONResponse(
            status_code=fastapi.status.HTTP_200_OK,
            content={
                "status": task_status.value,
                "msg": error_msg,
                "progress": progress
            }
        )

    @staticmethod
    @app.get("/gateway/task/download")
    async def task_download(query_key: str):
        parent_path = os.path.join(AppConfig.CACHE_PATH, query_key)
        service_done_path = os.path.join(parent_path, AppConfig.GATEWAY_SERVICE_CACHE)
        service_package_path = os.path.join(parent_path, AppConfig.GATEWAY_SERVICE_PACKAGE)

        if len(os.listdir(service_done_path)) == 0:
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": "no service cached, check running backend"}
            )

        save_zip = os.path.join(service_package_path, Tools.generate_randon_package_name())
        Tools.pack_zip(service_done_path, save_zip)

        return Tools.stream_response(save_zip)

    @staticmethod
    @app.get("/gateway/result/query")
    async def task_result(query_key: str, file_name: str, running_options: str):
        """
            根据前端需要整合需要返回的数据信息
        :param query_key:
        :param file_name:
        :return:
        """
        try:
            json_options = json.loads(running_options)
        except Exception as e:
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": "running options parser failed"}
            )

        status, json_str = await generate_cache_json(query_key, file_name, json_options)

        if not status:
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"res": "cache cleaned"}
            )

        return JSONResponse(
            status_code=fastapi.status.HTTP_200_OK,
            content={
                'res': json_str
            }
        )

    @staticmethod
    @app.get("/gateway/result/file")
    async def file_get(path_info: str):
        """
            返回远端文件，Get方式
            请求示例地址: http://0.0.0.0:9000/gateway/result/file?path_info=xxxx/IDRiD_20.jpg
        """
        if 'cam_plus' in path_info:
            path_info = path_info.replace('cam_plus', "cam++")
        actually_path = os.path.join(AppConfig.CACHE_PATH, path_info)
        if os.path.exists(actually_path):
            return fastapi.responses.FileResponse(actually_path)
        return fastapi.responses.JSONResponse(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                'res': "not found, {}".format(actually_path)
            }
        )

    @staticmethod
    @scheduler.scheduled_job('interval', seconds=3)
    def event_loop():
        task_key = cpf_task_queue.task_out()
        if task_key is not None:
            next_manager = cpf_task_queue.task_query(task_key)
            next_manager.task_start()

    @staticmethod
    @scheduler.scheduled_job('interval', seconds=60 * 30)  # 30 minute
    def task_clean():
        task_key = cpf_task_queue.task_done_query()
        if task_key is not None:
            cpf_task_queue.task_clean(task_key)

    @staticmethod
    @app.on_event("shutdown")
    async def shutdown_event():
        scheduler.shutdown()

    @staticmethod
    @app.on_event('startup')
    async def init_loop():
        scheduler.start()


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=9000)
