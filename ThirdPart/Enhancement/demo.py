# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

from .models.StillGAN.utils.util import tensor2im
from .models.StillGAN.models.networks import Identity, get_norm_layer, ResUNet


def array2bytes(array_img, suffix):
    # 对数组的图片格式进行编码
    success, encoded_array = cv2.imencode("." + suffix, array_img)
    # 将数组转为bytes
    bytes_img = encoded_array.tostring()
    
    return bytes_img


class Enhancement:
    """Define Image Enhancement.
    
    The well-trained StillGAN model is loaded to improve the visual quality of the given image.
    """
    def __init__(self, ckpt_path="models/StillGAN/checkpoints/120_net_G_A.pth", gpu_id=0, rescale_size=512):
        """Initialize the Enhancement class.
        
        Parameters:
            ckpt_path (str) -- the path of the model
            gpu_id (int) -- the id of the device (use negtive integer for CPU)
            rescale_size (int) -- the size after rescaling
        """
        super(Enhancement, self).__init__()
        if gpu_id >= 0:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.osize = [rescale_size, rescale_size]
        
        self.stillgan = ResUNet(3, 3, 64, norm_layer=get_norm_layer())
        self.model_parameter = torch.load(ckpt_path, map_location=self.device)
        self.stillgan.load_state_dict(self.model_parameter)
        self.stillgan = self.stillgan.to(self.device)
        self.stillgan.eval()
    
    def load_img(self, img_path):
        """Load the image
        
        Parameters:
            img_path (str) -- the path of the image
        """
        ori_Img = Image.open(img_path).convert("RGB")
        trans_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        transform_list = []
        transform_list.append(transforms.Resize(self.osize, Image.BICUBIC))
        # transform_list.append(transforms.RandomCrop(self.osize))
        transform_list += [transforms.ToTensor()]
        transform_list += [trans_norm]
        
        trans = transforms.Compose(transform_list)
        ori_Img = trans(ori_Img)
        ori_Img = Variable(torch.unsqueeze(ori_Img, dim=0).float(), requires_grad=False).to(self.device)
        
        return ori_Img
    
    def inference(self, img_path, suffix="png", is_keep_size=True):
        """Perform the image enhancement
        
        Parameters:
            img_path (str) -- the path of the image
            suffix (str) -- the format of the image
            is_keep_size (bool) -- keep size of the original image or not
        """
        pic = self.load_img(img_path)
        runned = self.stillgan(pic)
        image = tensor2im(runned)
        
        (r, g, b) = cv2.split(image)
        img_arr = cv2.merge([b, g, r])
        if is_keep_size:
            ori_arr = cv2.imread(img_path)
            H, W, C = ori_arr.shape
            img_arr = cv2.resize(img_arr, (W, H), interpolation=cv2.INTER_CUBIC)
        img_bytes = array2bytes(img_arr, suffix)
        
        return img_bytes

    def infer_wrapper(self, batch_data, ori_size):
        W, H = ori_size
        runned = self.stillgan(batch_data)
        image = tensor2im(runned)

        (r, g, b) = cv2.split(image)
        img_arr = cv2.merge([b, g, r])
        img_arr = cv2.resize(img_arr, (W, H), interpolation=cv2.INTER_CUBIC)
        img_bytes = array2bytes(img_arr, "png")

        image = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img_arr = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return img_arr


if __name__ == '__main__':
    StillGAN = Enhancement(ckpt_path="models/StillGAN/checkpoints/120_net_G_A.pth", gpu_id=0, rescale_size=512)
    start_t = time.time()
    img_bytes = StillGAN.inference("test_image.jpg", suffix="png", is_keep_size=True)
    end_t = time.time()
    print("Time cost: %f s. " % (end_t - start_t))
    image = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img_arr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite("test_result.png", img_arr)
