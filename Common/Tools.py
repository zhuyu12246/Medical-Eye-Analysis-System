# encoding=utf-8
import datetime
import os.path
import shutil
import tempfile
import time
from PIL import Image
import cv2 as cv
import hashlib
import uuid
import os
import zipfile
import fastapi
from Common.AppConfig import AppConfig


def time_record(func, *args, **kwargs):
    start = time.perf_counter()
    res = func(*args, *kwargs)
    end = time.perf_counter()

    return (end - start), res


def convert_2_png(file_path: str):
    if os.path.splitext(file_path)[-1] == ".png":
        return file_path

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        # pil method
        image = Image.open(file_path)
        image = image.convert('RGB')

        image.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        pass

    try:
        # opencv method
        image = cv.imread(file_path, cv.IMREAD_COLOR)
        cv.imwrite(temp_file.name, image)
        return temp_file.name
    except Exception as e:
        pass

    return None


def generate_time_mills():
    return int(round(time.time() * 1000))


def generate_strftime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_randon_package_name():
    return str(uuid.uuid4()) + ".zip"


# def generate_cache_dir(user_name: str = "running-cache"):
#     time_mills = str(generate_time_mills())
#     parent_path = os.path.join(AppConfig.CACHE_PATH, user_name, time_mills)
#
#     upload_csv = os.path.join(parent_path, AppConfig.CSV_CACHE)
#     plot_image = os.path.join(parent_path, AppConfig.PLOT_CACHE)
#     distance_result = os.path.join(parent_path, AppConfig.DISTANCE_CACHE)
#
#     os.makedirs(upload_csv, exist_ok=True)
#     os.makedirs(plot_image, exist_ok=True)
#     os.makedirs(distance_result, exist_ok=True)
#
#     return {
#         AppConfig.CSV_CACHE: upload_csv,
#         AppConfig.PLOT_CACHE: plot_image,
#         AppConfig.DISTANCE_CACHE: distance_result,
#         "parent_path": parent_path
#     }


def pack_zip(from_path, to_path):
    """打包指定路径到某目录下面
        :param from_path: 打包路径
        :param to_path: .zip结尾的文件存放位置
    """
    z = zipfile.ZipFile(to_path, 'w', zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(from_path):
        fpath = path.replace(from_path, '')
        for filename in filenames:
            encode_name = filename.encode('utf-8').decode('utf-8')
            z.write(os.path.join(path, filename),
                    os.path.join(fpath, encode_name))
    z.close()
    (file_path, file_name) = os.path.split(to_path)
    return file_name


def unzip_file(zip_file: str, save_path: str):
    if not (len(os.listdir(save_path)) == 0):
        shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(save_path)


def stream_response(zip_path):
    def send():
        with open(zip_path, 'rb') as file:
            while True:
                data = file.read(20 * 1024 * 1024)  # 20M
                if not data:
                    break
                yield data

    return fastapi.responses.StreamingResponse(send(), media_type='application/zip')


def generate_package(cache_path: str, from_path: str):
    parent_path = os.path.join(cache_path, "unet-package")
    os.makedirs(parent_path, exist_ok=True)

    file_name = generate_randon_package_name()
    save_path = os.path.join(parent_path, file_name)
    _ = pack_zip(from_path, save_path)
    return save_path


def generate_task_id(name: str):
    str_data = name + str(generate_time_mills())
    encoded = hashlib.sha256(str_data.encode()).hexdigest()
    return encoded


def generate_gateway_cache(name: str):
    task_id = generate_task_id(name)

    parent_path = os.path.join(AppConfig.CACHE_PATH, task_id)
    user_path = os.path.join(parent_path, AppConfig.GATEWAY_USER_CACHE)
    service_path = os.path.join(parent_path, AppConfig.GATEWAY_SERVICE_CACHE)
    packaged_path = os.path.join(parent_path, AppConfig.GATEWAY_SERVICE_PACKAGE)

    os.makedirs(user_path, exist_ok=True)
    os.makedirs(service_path, exist_ok=True)
    os.makedirs(packaged_path, exist_ok=True)
    return task_id


def rename_files(from_path: str, ori_name: str, exec_names: list = []):
    _ori_name = os.path.splitext(ori_name)[0]
    for root, dirs, files in os.walk(from_path):
        for filename in files:
            if filename in exec_names:
                continue
            name, ext = os.path.splitext(filename)
            new_name = "{}{}".format(_ori_name, ext)
            shutil.move(os.path.join(root, filename), os.path.join(root, new_name))
