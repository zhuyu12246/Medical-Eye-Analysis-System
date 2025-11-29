# encoding=utf-8
import hashlib
import json
import logging
import os
import traceback

from Common.AppConfig import AppConfig
from Service.TaskMain.Generate.CoroutinesLRUCache import LRUCache
from Service.TaskMain.Task.TaskObj import REMOTE_SERVICE_CACHE
from Service.TaskMain.Checker.ValidCheck import USE_ENHANCE, SEG_NAME, GRADE_NAME, RUNNING_JSON_OPTIONS

_lru_cache = LRUCache()


def _load_json(file_path: str, key: str):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                json_str = f.read()
                data = json.loads(json_str)
                return data[key]
        else:
            return ""
    except Exception as e:
        logging.exception(traceback.format_exc())
        return ""


def _replace_path_to_route(path: str):
    return path.replace(AppConfig.CACHE_PATH + "/", AppConfig.PROXY_URL + "/gateway/result/file?path_info=")


def _get_file_name_and_ext(filename: str):
    return os.path.splitext(filename)


def _build_user_cache(gateway_user: str, file_name: str):
    path = os.path.join(gateway_user, file_name)
    if os.path.exists(path):
        return True, _replace_path_to_route(path)
    return False, ""


def _build_enhance_cache(gateway_enh: str, file_name: str):
    path = os.path.join(gateway_enh, file_name)
    if os.path.exists(path):
        return True, _replace_path_to_route(path)
    return False, ""


def _build_seg_cache(gateway_service: str, file_name: str, options: dict, build_ori: USE_ENHANCE):
    _name, _ext = _get_file_name_and_ext(file_name)
    if options[RUNNING_JSON_OPTIONS.STR_SEG.value] == SEG_NAME.MB_CAMW:
        seg_path = os.path.join(gateway_service, REMOTE_SERVICE_CACHE.MB_SEG_CAMW.value)

    if options[RUNNING_JSON_OPTIONS.STR_SEG.value] == SEG_NAME.MB_UNET:
        seg_path = os.path.join(gateway_service, REMOTE_SERVICE_CACHE.MB_SEG_UNET.value)

    seg_path = os.path.join(seg_path, _name)
    if build_ori == USE_ENHANCE.USE:
        seg_path = os.path.join(seg_path, REMOTE_SERVICE_CACHE.ENHANCE.value)
    else:
        seg_path = os.path.join(seg_path, REMOTE_SERVICE_CACHE.ORI.value)

    path = os.path.join(seg_path, 'visualization', _name + ".png")
    if os.path.exists(path):
        return True, _replace_path_to_route(path)
    return False, ""


def _build_grade_cache(gateway_service: str, file_name: str, options: dict, build_ori: USE_ENHANCE):
    _name, _ext = _get_file_name_and_ext(file_name)
    if options[RUNNING_JSON_OPTIONS.STR_GRADE.value] == GRADE_NAME.MB_GRADE:
        grade_path = os.path.join(gateway_service, REMOTE_SERVICE_CACHE.MB_GRADE.value)

    grade_path = os.path.join(grade_path, _name)
    if build_ori == USE_ENHANCE.USE:
        grade_path = os.path.join(grade_path, REMOTE_SERVICE_CACHE.ENHANCE.value)
    else:
        grade_path = os.path.join(grade_path, REMOTE_SERVICE_CACHE.ORI.value)


    cam_path = os.path.join(grade_path, 'grade/cam', _name + ".png")
    cam_plus_path = os.path.join(grade_path, 'grade/cam++', _name + ".png")
    cam_pie = _load_json(os.path.join(grade_path, 'grade/pie', _name + ".json"), 'pie')

    if not os.path.exists(cam_path) or \
        not os.path.exists(cam_plus_path) or \
            cam_pie == "":
        return False, {
            'cam': "",
            'cam_plus': "",
            'pie': [0, 0, 0, 0, 0]
        }

    return True, {
        'cam': _replace_path_to_route(cam_path),
        'cam_plus': _replace_path_to_route(cam_plus_path.replace("cam++", "cam_plus")),
        'pie': cam_pie
    }


def _build_ori_cache_dict(gateway_user: str, gateway_service: str, file_name: str, options: dict):
    _s1, _c1 = _build_user_cache(gateway_user, file_name)
    _s2, _c2 = _build_seg_cache(gateway_service, file_name, options, USE_ENHANCE.NO_USE)
    _s3, _c3 = _build_grade_cache(gateway_service, file_name, options, USE_ENHANCE.NO_USE)
    status = (_s1 and _s2 and _s3)
    return status, {
        'file': _c1,
        'seg': _c2,
        'grade': _c3
    }


def _build_enh_cache_dict(gateway_service: str, file_name: str, options: dict):
    enhance_file_path = os.path.join(gateway_service, REMOTE_SERVICE_CACHE.MB_ENHANCE.value)
    _s1, _c1 = _build_enhance_cache(enhance_file_path, file_name)
    _s2, _c2 = _build_seg_cache(gateway_service, file_name, options, USE_ENHANCE.USE)
    _s3, _c3 = _build_grade_cache(gateway_service, file_name, options, USE_ENHANCE.USE)
    status = (_s1 and _s2 and _s3)
    return status, {
        'file': _c1,
        'seg': _c2,
        'grade': _c3
    }


def _build_service_dir_by_options(gateway_user: str, gateway_service: str, file_name: str, options: dict):
    if options[RUNNING_JSON_OPTIONS.STR_ENHANCE.value] == USE_ENHANCE.USE:
        _s1, _c1 = _build_ori_cache_dict(gateway_user, gateway_service, file_name, options)
        _s2, _c2 = _build_enh_cache_dict(gateway_service, file_name, options)
        return (_s1 and _s2), {
            'ori': _c1,
            'enh': _c2
        }

    _s1, _c1 = _build_ori_cache_dict(gateway_user, gateway_service, file_name, options)
    return _s1, {
        'ori': _c1
    }


async def generate_cache_json(query_key: str, file_name: str, running_options: dict):
    parent_path = os.path.join(AppConfig.CACHE_PATH, query_key)
    gateway_user_path = os.path.join(parent_path, AppConfig.GATEWAY_USER_CACHE)
    gateway_service_path = os.path.join(parent_path, AppConfig.GATEWAY_SERVICE_CACHE)

    if not os.path.exists(gateway_user_path) or not os.path.exists(gateway_service_path):
        return False, {}

    lru_key = hashlib.sha256((query_key + file_name + json.dumps(running_options)).encode()).hexdigest()
    _json = await _lru_cache.get(lru_key)
    if _json is not None:
        return True, _json

    status, json_dict = _build_service_dir_by_options(gateway_user_path, gateway_service_path, file_name, running_options)
    json_dict['done'] = status
    if status:
        await _lru_cache.put(lru_key, json.dumps(json_dict))
    return True, json.dumps(json_dict)