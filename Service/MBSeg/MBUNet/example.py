import os
import tempfile

import cv2
import numpy
from Common import Tools
from Common.AppConfig import AppConfig
from Service.MBSeg.MBUNet.client import MBUNetClient
from Common.RemoteClient import RemoteClient

client: RemoteClient = MBUNetClient()

res = client.run_by_file("../test_image.jpeg")
if res is None:
    print("远程访问链接失败或图像解析异常")
else:
    status_code, content = res
    if status_code != 200:
        print("远端执行异常")
    else:
        # zip download
        temp_file = tempfile.NamedTemporaryFile(suffix=".zip")
        temp_file.write(content)

        # unzip
        cache_path = os.path.join(AppConfig.CACHE_PATH, "test-cache/mb-unet-cache", Tools.generate_strftime())
        os.makedirs(cache_path, exist_ok=True)
        Tools.unzip_file(temp_file.name, cache_path)

        # rename
        Tools.rename_files(cache_path, "test_image", ['summary.csv'])