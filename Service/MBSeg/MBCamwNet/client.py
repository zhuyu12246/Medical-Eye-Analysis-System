# encoding=utf-8
import time
import logging
import traceback
import requests
from Common.AppConfig import AppConfig, URL_CONFIG
from Common.RemoteClient import RemoteClient
from Common.Tools import convert_2_png


class MBCamwNetClient(RemoteClient):
    def run_by_file(self, file_path: str):
        file_path = convert_2_png(file_path)
        if file_path is None:
            return None

        file = {"file": open(file_path, 'rb')}

        try:
            start = time.perf_counter()
            response = requests.post(URL_CONFIG['seg-camw'], files=file)
            end = time.perf_counter()

            res_content = response.content
            logging.info("time cost: {}s".format(end - start))
            return response.status_code, res_content
        except Exception as e:
            logging.exception(traceback.format_exc())
            return None
