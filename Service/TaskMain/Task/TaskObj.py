# encoding=utf-8
import logging
import os
import tempfile
import time
import traceback
from enum import Enum
import cv2
import numpy
from Common import Tools
from Service.TaskMain.Checker.ValidCheck import USE_ENHANCE, SEG_NAME, GRADE_NAME, RUNNING_JSON_OPTIONS
from Service.TaskMain.Task.TaskManager import TaskManager, TASK_STATUS
from Service.TaskMain.Task.TaskQueue import TaskQueue
from Service.MBEnhance.client import MBEnhanceClient
from Service.MBSeg.MBUNet.client import MBUNetClient
from Service.MBSeg.MBCamwNet.client import MBCamwNetClient
from Service.MBGrade.client import MBGradeClient


class REMOTE_SERVICE_CACHE(Enum):
    ORI = "ori"
    ENHANCE = "enh"
    MB_ENHANCE = "mb-enhance-cache"
    MB_SEG_UNET = "mb-seg-unet-cache"
    MB_SEG_CAMW = "mb-seg-camw-cache"
    MB_GRADE = "mb-grade"


def _run_remote_enhance(user_file: str, dst_cache: str):
    file_name = os.path.basename(user_file)
    save_name = os.path.join(dst_cache, file_name)

    client = MBEnhanceClient()
    res = client.run_by_file(user_file)
    if res is None:
        raise RuntimeError("{}, Request failed in mb-enhance, check network".format(file_name))
    else:
        status_code, content = res
        if status_code != 200:
            raise RuntimeError("{}, Request run failed in mb-enhance, check code".format(file_name))
        else:
            image = cv2.imdecode(numpy.frombuffer(content, numpy.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(save_name, image)

    return save_name


def _run_remote_seg_unet(user_file: str, dst_cache: str):
    file_name = os.path.basename(user_file)
    _name, _ = os.path.splitext(file_name)
    client = MBUNetClient()
    res = client.run_by_file(user_file)
    if res is None:
        raise RuntimeError("{}, Request failed in mb-seg-unet, check network".format(file_name))
    else:
        status_code, content = res
        if status_code != 200:
            raise RuntimeError("{}, Request run failed in mb-seg-unet, check code".format(file_name))
        else:
            # zip download
            temp_file = tempfile.NamedTemporaryFile(suffix=".zip")
            temp_file.write(content)

            # unzip
            Tools.unzip_file(temp_file.name, dst_cache)

            # rename
            Tools.rename_files(dst_cache, _name, ['summary.csv'])


def _run_remote_seg_camw(user_file: str, dst_cache: str):
    file_name = os.path.basename(user_file)
    _name, _ = os.path.splitext(file_name)
    client = MBCamwNetClient()
    res = client.run_by_file(user_file)
    if res is None:
        raise RuntimeError("{}, Request failed in mb-seg-camw, check network".format(file_name))
    else:
        status_code, content = res
        if status_code != 200:
            raise RuntimeError("{}, Request run failed in mb-seg-camw, check code".format(file_name))
        else:
            # zip download
            temp_file = tempfile.NamedTemporaryFile(suffix=".zip")
            temp_file.write(content)

            # unzip
            Tools.unzip_file(temp_file.name, dst_cache)

            # rename
            Tools.rename_files(dst_cache, _name, ['summary.csv'])


def _run_remote_quality(user_file: str, dst_cache: str):
    file_name = os.path.basename(user_file)
    _name, _ = os.path.splitext(file_name)
    client = MBGradeClient()
    res = client.run_by_file(user_file)
    if res is None:
        raise RuntimeError("{}, Request failed in mb-grade, check network".format(file_name))
    else:
        status_code, content = res
        if status_code != 200:
            raise RuntimeError("{}, Request run failed in mb-grade, check code".format(file_name))
        else:
            # zip download
            temp_file = tempfile.NamedTemporaryFile(suffix=".zip")
            temp_file.write(content)

            # unzip
            Tools.unzip_file(temp_file.name, dst_cache)

            # rename
            Tools.rename_files(dst_cache, _name, [])


def _build_service_dir_by_options(service_cache: str, options: dict):
    cache_dir = {}
    if options[RUNNING_JSON_OPTIONS.STR_ENHANCE.value] == USE_ENHANCE.USE:
        _dir = os.path.join(service_cache, REMOTE_SERVICE_CACHE.MB_ENHANCE.value)
        cache_dir[RUNNING_JSON_OPTIONS.STR_ENHANCE.value] = _dir, _run_remote_enhance
        os.makedirs(_dir, exist_ok=True)
    else:
        cache_dir[RUNNING_JSON_OPTIONS.STR_ENHANCE.value] = None, None

    if options[RUNNING_JSON_OPTIONS.STR_SEG.value] == SEG_NAME.MB_UNET:
        _dir = os.path.join(service_cache, REMOTE_SERVICE_CACHE.MB_SEG_UNET.value)
        cache_dir[RUNNING_JSON_OPTIONS.STR_SEG.value] = _dir, _run_remote_seg_unet
        os.makedirs(_dir, exist_ok=True)

    if options[RUNNING_JSON_OPTIONS.STR_SEG.value] == SEG_NAME.MB_CAMW:
        _dir = os.path.join(service_cache, REMOTE_SERVICE_CACHE.MB_SEG_CAMW.value)
        cache_dir[RUNNING_JSON_OPTIONS.STR_SEG.value] = _dir, _run_remote_seg_camw
        os.makedirs(_dir, exist_ok=True)

    if options[RUNNING_JSON_OPTIONS.STR_GRADE.value] == GRADE_NAME.MB_GRADE:
        _dir = os.path.join(service_cache, REMOTE_SERVICE_CACHE.MB_GRADE.value)
        cache_dir[RUNNING_JSON_OPTIONS.STR_GRADE.value] = _dir, _run_remote_quality
        os.makedirs(_dir, exist_ok=True)

    return cache_dir


def _run_task(abs_file: str, cache_dict: dict):
    # file info
    file_name = os.path.basename(abs_file)
    _name, _ext = os.path.splitext(file_name)

    # get enhance file if need
    enhance_file = None
    _dir_enhance, _fun_enhance = cache_dict[RUNNING_JSON_OPTIONS.STR_ENHANCE.value]
    if _dir_enhance:
        enhance_file = _fun_enhance(abs_file, _dir_enhance)

    # run seg remote
    # cache_dir
    #   - file_name
    #       - ori
    #       - enh
    _dir_seg, _fun_seg = cache_dict[RUNNING_JSON_OPTIONS.STR_SEG.value]
    ori_seg_path = os.path.join(_dir_seg, _name, REMOTE_SERVICE_CACHE.ORI.value)
    enh_seg_path = os.path.join(_dir_seg, _name, REMOTE_SERVICE_CACHE.ENHANCE.value)
    os.makedirs(ori_seg_path, exist_ok=True)
    os.makedirs(enh_seg_path, exist_ok=True)

    _fun_seg(abs_file, ori_seg_path)
    if enhance_file:
        _fun_seg(enhance_file, enh_seg_path)

    # run grade remote
    _dir_grade, _fun_grade = cache_dict[RUNNING_JSON_OPTIONS.STR_GRADE.value]
    ori_grade_path = os.path.join(_dir_grade, _name, REMOTE_SERVICE_CACHE.ORI.value)
    enh_grade_path = os.path.join(_dir_grade, _name, REMOTE_SERVICE_CACHE.ENHANCE.value)
    os.makedirs(ori_grade_path, exist_ok=True)
    os.makedirs(enh_grade_path, exist_ok=True)

    _fun_grade(abs_file, ori_grade_path)
    if enhance_file:
        _fun_grade(enhance_file, enh_grade_path)


def _run_remote(user_cache: str, service_cache: str, options: dict, task_manager: TaskManager):
    running_cache = _build_service_dir_by_options(service_cache, options)
    file_names = os.listdir(user_cache)

    for i in range(len(file_names)):
        # file info
        abs_file = os.path.join(user_cache, file_names[i])

        # running
        task_manager.set_normal_msg("Now deal with {}".format(file_names[i]))
        start = time.perf_counter()

        _run_task(abs_file, running_cache)

        end = time.perf_counter()
        msg = "{}, done, cost: {}s".format(file_names[i], round((end - start), 2))
        logging.info(msg)
        task_manager.set_normal_msg(msg)
        # progress up
        task_manager.set_progress(round(((i + 1) / len(file_names) * 100), 1))


def run_service(user_cache: str, service_cache: str, options: dict,
                task_queue: TaskQueue, task_key: str):
    task_manager: TaskManager = task_queue.task_query(task_key)
    task_manager.set_running_state(TASK_STATUS.STATUS_RUNNING)
    try:
        _run_remote(user_cache, service_cache, options, task_manager)
        task_manager.set_running_state(TASK_STATUS.STATUS_OK)
        task_manager.set_normal_msg("Running done")
        task_queue.task_done(task_key)
    except Exception as e:
        logging.exception(traceback.format_exc())
        task_manager.set_running_state(TASK_STATUS.STATUS_ERROR)
        task_manager.set_error_msg(str(e))
        task_queue.task_done(task_key)
