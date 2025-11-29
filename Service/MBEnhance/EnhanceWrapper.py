# encoding=utf-8
import threading

import torch
from torchvision import transforms
from PIL import Image
from Common.AppConfig import AppConfig
from ThirdPart.Enhancement.demo import Enhancement


class EnhanceWrapper:
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.enhance = Enhancement(AppConfig.RES_GAN)

    def _preprocess(self, data_in: str):
        ori_Img = Image.open(data_in).convert("RGB")
        ori_size = ori_Img.size
        trans_image = self.transform(ori_Img)
        trans_dim = trans_image.unsqueeze(0)
        return trans_dim, ori_size

    @torch.no_grad()
    def infer(self, image_path: str):
        trans_dim, ori_size = self._preprocess(image_path)
        trans_cuda = trans_dim.to(self.enhance.device)
        res = self.enhance.infer_wrapper(trans_cuda, ori_size)
        return res

    def __new__(cls, *args, **kwargs):
        EnhanceWrapper._lock.acquire()
        if cls._inst is None:
            cls._inst = super(EnhanceWrapper, cls).__new__(cls)
        EnhanceWrapper._lock.release()
        return cls._inst
