# encoding=utf-8
import logging
import os
import threading
import traceback

import cv2
import torch
from torchvision import transforms
from PIL import Image
from Common.AppConfig import AppConfig
from ThirdPart.Segmentation.model import Unet
from ThirdPart.Segmentation.DR_seg import segment_DR_unet_wrapper, show_seg, visual_wrapper
from Service.MBSeg.MBAnalysis.AnalysisWrapper import AnalysisWrapper


class MBUNetWrapper:
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Unet(3, 6)
        model_dict = torch.load(AppConfig.RES_SEG_UNET)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.result_name = "U-Net_results"

    def _preprocess(self, data_in: str):
        ori_Img = Image.open(data_in)
        ori_size = ori_Img.size
        trans_image = self.transform(ori_Img)
        trans_dim = trans_image.unsqueeze(0)
        return trans_dim, ori_size

    def _postprocess(self, cache_path: str, image_path: str, pred_arr):
        try:
            visual_wrapper(cache_path, image_path, pred_arr, self.result_name)
            file_name = os.path.basename(image_path)
            analysis = AnalysisWrapper(os.path.join(cache_path, self.result_name), file_name)
            analysis.generate_dr_statics()
            return True, ""
        except Exception as e:
            except_str = traceback.format_exc()
            logging.exception(except_str)
            return False, except_str

    @torch.no_grad()
    def infer(self, cache_path: str, image_path: str):
        trans_dim, ori_size = self._preprocess(image_path)
        trans_cuda = trans_dim.to(self.device)

        pred_arr = segment_DR_unet_wrapper(self.model, trans_cuda, ori_size[0], ori_size[1])
        return self._postprocess(cache_path, image_path, pred_arr)

    def __new__(cls, *args, **kwargs):
        MBUNetWrapper._lock.acquire()
        if cls._inst is None:
            cls._inst = super(MBUNetWrapper, cls).__new__(cls)
        MBUNetWrapper._lock.release()
        return cls._inst
