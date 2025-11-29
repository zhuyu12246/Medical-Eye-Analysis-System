# -*- coding: utf-8 -*-
import json
import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models

from grad_cam import GradCAM, GradCamPlusPlus
from guided_back_propagation import GuidedBackPropagation
from utils import circle_crop, get_last_conv_name, norm_image, gen_cam, gen_gb


# DenseNet from Kevin
def grad_DR(img_path, gpu_ids=""):
    # 是否使用cuda
    if gpu_ids != "" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 加载模型
    model = models.densenet121(pretrained=True)
    fc_feature = model.classifier.in_features
    model.classifier = nn.Linear(fc_feature, 5)
    softmax = nn.Softmax(dim=1)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load("state-9-0.7958954625621292.pth", map_location=device))
    model.eval()

    # 读入并预处理图像
    img_arr = cv2.resize(circle_crop(img_path), (224, 224), interpolation=cv2.INTER_LINEAR) / 255.0  # CUBIC##
    x = torch.from_numpy(img_arr[:, :, ::-1].astype(np.float32).transpose((2, 0, 1))).unsqueeze(0).to(device)

    # 模型预测
    with torch.no_grad():
        if device == torch.device("cuda"):
            y = softmax(model(x)).squeeze().cpu().detach().numpy().tolist()
        else:
            y = softmax(model.module(x)).squeeze().cpu().detach().numpy().tolist()

    x.requires_grad = True
    layer_name = get_last_conv_name(model)  # get last convolution layer
    grad_cam = GradCAM(model, layer_name, device)  # Grad-CAM
    grad_cam_plus_plus = GradCamPlusPlus(model, layer_name, device)  # Grad-CAM++

    # 输出图像
    image_dict = {}
    mask = grad_cam(x, None)  # cam mask
    image_dict["cam"], image_dict["heatmap"] = gen_cam(img_arr[:, :, ::-1].astype(np.float32), mask)
    grad_cam.remove_handlers()
    mask_plus_plus = grad_cam_plus_plus(x, None)  # cam++ mask
    image_dict["cam++"], image_dict["heatmap++"] = gen_cam(img_arr[:, :, ::-1].astype(np.float32), mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(model, device)
    x.grad.zero_()  # 梯度置零
    grad = gbp(x)

    gb = gen_gb(grad)
    image_dict["gb"] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict["cam_gb"] = norm_image(cam_gb)

    return y, y.index(max(y)), image_dict


def grad_dr_wrapper(model, x, device, img_arr):
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        y = softmax(model(x)).squeeze().cpu().detach().numpy().tolist()

    x.requires_grad = True
    layer_name = get_last_conv_name(model)  # get last convolution layer
    grad_cam = GradCAM(model, layer_name, device)  # Grad-CAM
    grad_cam_plus_plus = GradCamPlusPlus(model, layer_name, device)  # Grad-CAM++

    # 输出图像
    image_dict = {}
    mask = grad_cam(x, None)  # cam mask
    image_dict["cam"], image_dict["heatmap"] = gen_cam(img_arr[:, :, ::-1].astype(np.float32), mask)
    grad_cam.remove_handlers()
    mask_plus_plus = grad_cam_plus_plus(x, None)  # cam++ mask
    image_dict["cam++"], image_dict["heatmap++"] = gen_cam(img_arr[:, :, ::-1].astype(np.float32), mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(model, device)
    x.grad.zero_()  # 梯度置零
    grad = gbp(x)

    gb = gen_gb(grad)
    image_dict["gb"] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict["cam_gb"] = norm_image(cam_gb)

    return y, y.index(max(y)), image_dict


def grad_dr_visual_wrapper(grad_out, img_path, dst_dir, csv_path):
    p_lst, pred_cls, image_dict = grad_out
    file_name, ext = os.path.splitext(os.path.basename(img_path))

    with open(csv_path, "w") as f:
        f.write("%s,%f,%f,%f,%f,%f,%d\n"
                % (file_name + ext, p_lst[0], p_lst[1], p_lst[2], p_lst[3], p_lst[4], pred_cls))

    for key, image in image_dict.items():
        if not os.path.exists(os.path.join(dst_dir, key)):
            os.makedirs(os.path.join(dst_dir, key), exist_ok=True)
        cv2.imwrite(os.path.join(dst_dir, key, file_name + ".png"), image[:, :, ::-1])

    if not os.path.exists(os.path.join(dst_dir, "pie")):
        os.makedirs(os.path.join(dst_dir, "pie"), exist_ok=True)

    #plt.rcParams["font.sans-serif"] = "SimHei"  # 设置中文显示
    #plt.figure(figsize=(6, 6))  # 将画布设定为正方形，则绘制的饼图是正圆

    save_dict = {
        'pie': p_lst
    }
    with open(os.path.join(dst_dir, "pie", file_name + ".json"), 'w') as f:
        f.write(json.dumps(save_dict))

    # plt.pie(p_lst, explode=explode, labels=list(range(5)), autopct="%1.1f%%")  # 绘制饼图
    # plt.title("糖网分期预测概率类别分布饼状图")  # 绘制标题
    # plt.savefig(os.path.join(dst_dir, "pie", img_path.split("/")[-1].replace(".jpeg", ".png")))  # 保存图片


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR_grading")
    parser.add_argument("--gpu_ids", type=str, default="0", help="device")
    parser.add_argument("--img_path", type=str, required=True, help="path to image")
    parser.add_argument("--csv_path", type=str, required=True, help="path to csv file")
    parser.add_argument("--dst_dir", type=str, required=True, help="path to folder for saving visual results")
    args = parser.parse_args()

    start_t = time.time()
    p_lst, pred_cls, image_dict = grad_DR(args.img_path, gpu_ids=args.gpu_ids)
    end_t = time.time()
    print("Time cost: %f s. " % (end_t - start_t))

    f = open(args.csv_path, "a")
    f.write("%s,%f,%f,%f,%f,%f,%d\n"
            % (args.img_path, p_lst[0], p_lst[1], p_lst[2], p_lst[3], p_lst[4], pred_cls))
    f.close()

    for key, image in image_dict.items():
        if not os.path.exists(os.path.join(args.dst_dir, key)):
            os.makedirs(os.path.join(args.dst_dir, key))
        cv2.imwrite(os.path.join(args.dst_dir, key, args.img_path.split("/")[-1].replace(".jpeg", ".png")),
                    image[:, :, ::-1])

    if not os.path.exists(os.path.join(args.dst_dir, "pie")):
        os.makedirs(os.path.join(args.dst_dir, "pie"))
    plt.rcParams["font.sans-serif"] = "SimHei"  # 设置中文显示
    plt.figure(figsize=(6, 6))  # 将画布设定为正方形，则绘制的饼图是正圆
    rank = [index for index, value in sorted(list(enumerate(p_lst)), key=lambda x: x[1])]
    explode = [0.01] * 5
    for idx in range(len(rank)):
        explode[rank[idx]] *= (idx + 1)
    plt.pie(p_lst, explode=explode, labels=list(range(5)), autopct="%1.1f%%")  # 绘制饼图
    plt.title("糖网分期预测概率类别分布饼状图")  # 绘制标题
    plt.savefig(os.path.join(args.dst_dir, "pie", args.img_path.split("/")[-1].replace(".jpeg", ".png")))  # 保存图片
