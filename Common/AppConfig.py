# encoding=utf-8
import os
import logging
import sys
from logging.handlers import TimedRotatingFileHandler


class AppConfig:
    # 外网代理url
    PROXY_URL = "http://127.0.0.1:9000"

    # 常规配置
    CONFIG_PATH = os.path.join(os.path.dirname(__file__))
    PROJECT_PATH = os.path.abspath(os.path.join(CONFIG_PATH, ".."))
    RESOURCES_PATH = os.path.join(PROJECT_PATH, "resources")
    CACHE_PATH = os.path.join(PROJECT_PATH, "runtime-cache")
    LOGGER_PATH = os.path.join(PROJECT_PATH, "runtime-log")

    # gateway dir
    GATEWAY_USER_CACHE = "gateway-user-cache"
    GATEWAY_SERVICE_PACKAGE = "gateway-user-package"
    GATEWAY_SERVICE_CACHE = "gateway-service-cache"

    # lib
    THIRD_ENHANCE = os.path.join(PROJECT_PATH, "ThirdPart/Enhancement")
    THIRD_GRADING = os.path.join(PROJECT_PATH, "ThirdPart/Grading")
    THIRD_SEG = os.path.join(PROJECT_PATH, "ThirdPart/Segmentation")

    # res
    RES_GRAD = os.path.join(RESOURCES_PATH, "Grading/state.pth")
    RES_SEG_UNET = os.path.join(RESOURCES_PATH, "Segmentation/state-401-401-dict.pth")
    RES_SEG_CAMW = os.path.join(RESOURCES_PATH, "Segmentation/camw-dict.pth")
    RES_GAN = os.path.join(RESOURCES_PATH, "StillGAN/120_net_G_A.pth")


sys.path.append(AppConfig.THIRD_ENHANCE)
sys.path.append(AppConfig.THIRD_GRADING)
sys.path.append(AppConfig.THIRD_SEG)

os.makedirs(AppConfig.RESOURCES_PATH, exist_ok=True)
os.makedirs(AppConfig.CACHE_PATH, exist_ok=True)
os.makedirs(AppConfig.LOGGER_PATH, exist_ok=True)

URL_CONFIG = {
    # 2.0121531520017015s
    "enhance": 'http://192.168.3.99:9001/run/mb/enhance',
    # 4.389097958999628
    "seg-unet": 'http://192.168.3.99:9002/run/mb/seg/unet',
    # 4.531604387000698s
    "seg-camw": "http://192.168.3.99:9003/run/mb/seg/camw",
    # 2.199526246997266s
    "grade": 'http://192.168.3.99:9004/run/mb/grade'
}


# 日志配置
def set_logger():
    logger = logging.getLogger()  # root
    logger.setLevel(logging.INFO)

    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(AppConfig.LOGGER_PATH, "app-log"),
        when='D', interval=1
    )
    file_handler.suffix = "%Y-%m-%d.log"

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logging_format = logging.Formatter(
        "%(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s")

    file_handler.setFormatter(logging_format)
    console_handler.setFormatter(logging_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


set_logger()
