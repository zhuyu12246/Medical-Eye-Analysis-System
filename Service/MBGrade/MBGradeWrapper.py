# encoding=utf-8
import logging
import os
import threading
import traceback
import cv2
import numpy
import torch
from torch import nn
from Common.AppConfig import AppConfig
from torchvision import models
from ThirdPart.Grading.utils import circle_crop
from ThirdPart.Grading.demo import grad_dr_wrapper, grad_dr_visual_wrapper


class MBGradeWrapper:
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.densenet121(pretrained=True)
        fc_feature = self.model.classifier.in_features
        self.model.classifier = nn.Linear(fc_feature, 5)
        self.model.load_state_dict(torch.load(AppConfig.RES_GRAD))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.result_name = "cam-result"

    def _preprocess(self, data_in: str):
        img_arr = cv2.resize(circle_crop(data_in), (224, 224), interpolation=cv2.INTER_LINEAR) / 255.0
        x = torch.from_numpy(img_arr[:, :, ::-1].astype(numpy.float32).transpose((2, 0, 1))).unsqueeze(0)
        return x, img_arr

    def _postprocess(self, cache_path: str, image_path: str, pred_arr):
        try:
            file_name = os.path.splitext(os.path.basename(image_path))[0]

            # csv
            csv_path = os.path.join(cache_path, "csv")
            os.makedirs(csv_path, exist_ok=True)
            csv_file = os.path.join(csv_path, file_name + ".csv")

            # grad cache
            grad_path = os.path.join(cache_path, "grade")
            os.makedirs(grad_path, exist_ok=True)

            grad_dr_visual_wrapper(pred_arr, image_path, grad_path, csv_file)
            return True, ""
        except Exception as e:
            except_str = traceback.format_exc()
            logging.exception(except_str)
            return False, except_str

    # @torch.no_grad()
    def infer(self, cache_path: str, image_path: str):
        trans_dim, img_arr = self._preprocess(image_path)
        trans_cuda = trans_dim.to(self.device)

        pred_arr = grad_dr_wrapper(self.model, trans_cuda, self.device, img_arr)
        return self._postprocess(cache_path, image_path, pred_arr)

    def __new__(cls, *args, **kwargs):
        MBGradeWrapper._lock.acquire()
        if cls._inst is None:
            cls._inst = super(MBGradeWrapper, cls).__new__(cls)
        MBGradeWrapper._lock.release()
        return cls._inst
