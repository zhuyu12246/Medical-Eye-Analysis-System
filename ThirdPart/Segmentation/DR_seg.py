# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from models.net_builder import net_builder


# U-Net from Yanmiao Bai
def segment_DR_unet(img_path, gpu_ids=""):
    # 是否使用cuda
    if gpu_ids != "" and torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 加载模型
    model = torch.load("state-401-401.pth")
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    # 读入并预处理图像
    img = Image.open(img_path)
    w, h = img.size
    x = transforms.ToTensor()(img.resize((512, 512))).unsqueeze(0).to(device)

    # 模型预测: [6, 512, 512]: 0.0~255.0
    # {"background": 0, "microaneurysms": 1, "hemorrhages": 2, "hard_exudates": 3, "soft_exudates": 4, "disc": 5}
    with torch.no_grad():
        y = model(x).data.squeeze().cpu().numpy() * 255.0

    predict_list = []
    for index in range(y.shape[0]):
        predict = Image.fromarray(y[index].squeeze().astype(np.uint8), mode="L")  # 灰度模式存图，h * w无通道
        predict_resize = np.array(predict.resize((w, h)))
        predict_list.append(predict_resize)

    # {"background": 0, "hemorrhages": 1, "hard_exudates": 2, "microaneurysms": 3, "disc": 4, "soft_exudates": 5}
    predict_list = predict_list[0], predict_list[2], predict_list[3], predict_list[1], predict_list[5], predict_list[4]
    prob_arr = np.array(predict_list) / 255.0
    prob_arr /= np.sum(prob_arr, axis=0)
    pred_arr = np.zeros_like(prob_arr).astype(np.uint8)
    pred_label_arr = np.argmax(prob_arr, axis=0)

    for j in range(pred_arr.shape[0]):
        pred_arr[j, :, :][pred_label_arr == j] = 255

    return pred_arr


def segment_DR_unet_wrapper(model, batch, w, h):
    y = model(batch).data.squeeze().cpu().numpy() * 255.0

    predict_list = []
    for index in range(y.shape[0]):
        predict = Image.fromarray(y[index].squeeze().astype(np.uint8), mode="L")  # 灰度模式存图，h * w无通道
        predict_resize = np.array(predict.resize((w, h)))
        predict_list.append(predict_resize)

    # {"background": 0, "hemorrhages": 1, "hard_exudates": 2, "microaneurysms": 3, "disc": 4, "soft_exudates": 5}
    predict_list = predict_list[0], predict_list[2], predict_list[3], predict_list[1], predict_list[5], predict_list[4]
    prob_arr = np.array(predict_list) / 255.0
    prob_arr /= np.sum(prob_arr, axis=0)
    pred_arr = np.zeros_like(prob_arr).astype(np.uint8)
    pred_label_arr = np.argmax(prob_arr, axis=0)

    for j in range(pred_arr.shape[0]):
        pred_arr[j, :, :][pred_label_arr == j] = 255

    return pred_arr


# CAMWNet from Lianyu Wang
def segment_DR_camwnet(img_path, gpu_ids=""):
    # 是否使用cuda
    if gpu_ids != "" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 加载模型
    model = net_builder("camwnet", None, False)
    model = nn.DataParallel(model).to(device)
    checkpoint = torch.load("camw.pkl", map_location=device)
    model.load_state_dict(checkpoint["net"])
    model.eval()

    # 读入并预处理图像
    resize_img = transforms.Resize((512, 512), Image.BILINEAR)
    img = Image.open(img_path)
    w, h = img.size
    img = resize_img(img)
    img = np.array(img).astype(np.uint8)
    img = img.transpose(2, 0, 1) / 255.0
    x = torch.from_numpy(img.copy()).float().unsqueeze(0).to(device)

    # 模型预测: [6, 512, 512]: 0.0~255.0
    # {"background": 0, "hemorrhages": 1, "hard_exudates": 2, "microaneurysms": 3, "disc": 4, "soft_exudates": 5}
    with torch.no_grad():
        if device == torch.device("cuda"):
            y = torch.exp(model(x)[0]).data.squeeze().cpu().numpy() * 255.0
        else:
            y = torch.exp(model.module(x)[0]).data.squeeze().cpu().numpy() * 255.0

    predict_list = []
    for index in range(y.shape[0]):
        predict = Image.fromarray(y[index].squeeze().astype(np.uint8), mode="L")  # 灰度模式存图，h * w无通道
        predict_resize = np.array(predict.resize([w, h]))
        predict_list.append(predict_resize)

    prob_arr = np.array(predict_list) / 255.0
    prob_arr /= np.sum(prob_arr, axis=0)
    pred_arr = np.zeros_like(prob_arr).astype(np.uint8)
    pred_label_arr = np.argmax(prob_arr, axis=0)

    for j in range(pred_arr.shape[0]):
        pred_arr[j, :, :][pred_label_arr == j] = 255

    return pred_arr


def segment_DR_camwnet_wrapper(model, batch, w, h):
    y = torch.exp(model(batch)[0]).data.squeeze().cpu().numpy() * 255.0

    predict_list = []
    for index in range(y.shape[0]):
        predict = Image.fromarray(y[index].squeeze().astype(np.uint8), mode="L")  # 灰度模式存图，h * w无通道
        predict_resize = np.array(predict.resize([w, h]))
        predict_list.append(predict_resize)

    prob_arr = np.array(predict_list) / 255.0
    prob_arr /= np.sum(prob_arr, axis=0)
    pred_arr = np.zeros_like(prob_arr).astype(np.uint8)
    pred_label_arr = np.argmax(prob_arr, axis=0)

    for j in range(pred_arr.shape[0]):
        pred_arr[j, :, :][pred_label_arr == j] = 255

    return pred_arr


# Visualize the segmentation results
def show_seg(img_path, pred_arr):
    img_arr = cv2.imread(img_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for j in range(1, pred_arr.shape[0]):
        edge_arr = cv2.dilate(cv2.Canny(pred_arr[j, :, :], 50, 150), kernel, iterations=1)
        edge_arr = cv2.cvtColor(edge_arr, cv2.COLOR_GRAY2BGR)
        img_arr *= (1 - edge_arr // 255)

        if j // 4 == 0:
            edge_arr[:, :, 0] = 0
        if j % 4 < 2:
            edge_arr[:, :, 1] = 0
        if j % 2 == 0:
            edge_arr[:, :, -1] = 0
        img_arr += edge_arr

    return img_arr


def draw_text(img_arr):
    h, w, c = img_arr.shape

    h_5 = int(h * 0.02)
    w_5 = int(h * 0.02)
    rect_5 = min(h_5, w_5)

    # hemorrhages, 出血
    pre = 2
    col = 2
    cv2.rectangle(img_arr, (col * h_5, pre * w_5), (int((col + 0.5) * rect_5), int((pre + 0.5) * rect_5)), (0, 0, 255), -1)
    cv2.putText(img_arr, "hemorrhages", (int((col + 1.2) * h_5), int((pre + 0.5) * w_5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)

    # hard_exudates, 硬渗出
    pre += 1
    cv2.rectangle(img_arr, (col * h_5, pre * w_5), (int((col + 0.5) * rect_5), int((pre + 0.5) * rect_5)), (0, 255, 0), -1)
    cv2.putText(img_arr, "hard exudates", (int((col + 1.2) * h_5), int((pre + 0.5) * w_5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)

    # microaneurysms, 微血管瘤
    pre += 1
    cv2.rectangle(img_arr, (col * h_5, pre * w_5), (int((col + 0.5) * rect_5), int((pre + 0.5) * rect_5)), (0, 255, 255), -1)
    cv2.putText(img_arr, "microaneurysms", (int((col + 1.2) * h_5), int((pre + 0.5) * w_5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)

    # disc, 视盘
    pre += 1
    cv2.rectangle(img_arr, (col * h_5, pre * w_5), (int((col + 0.5) * rect_5), int((pre + 0.5) * rect_5)), (255, 0, 0), -1)
    cv2.putText(img_arr, "disc", (int((col + 1.2) * h_5), int((pre + 0.5) * w_5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)

    # soft_exudates, 软渗出
    pre += 1
    cv2.rectangle(img_arr, (col * h_5, pre * w_5), (int((col + 0.5) * rect_5), int((pre + 0.5) * rect_5)), (255, 0, 255), -1)
    cv2.putText(img_arr, "soft exudates", (int((col + 1.2) * h_5), int((pre + 0.5) * w_5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)
    return img_arr



def visual_wrapper(cache_path: str, image_path: str, pred_arr, result_name: str):
    obj_lst = ["background", "hemorrhages", "hard_exudates", "microaneurysms", "disc", "soft_exudates"]
    img_arr = show_seg(image_path, pred_arr)
    file_name = os.path.basename(image_path)

    # visualization
    visualization = os.path.join(cache_path, "{}/visualization".format(result_name))
    os.makedirs(visualization, exist_ok=True)

    # draw label
    img_arr = draw_text(img_arr)
    cv2.imwrite(os.path.join(visualization, file_name), img_arr)

    predict_array = np.zeros_like(pred_arr).astype(np.uint8)
    pred_flag_arr = np.argmax(pred_arr, axis=0)

    for j in range(predict_array.shape[0]):
        if not os.path.exists(os.path.join(cache_path, result_name, obj_lst[j])):
            os.makedirs(os.path.join(cache_path, result_name, obj_lst[j]), exist_ok=True)
        predict_array[j, :, :][pred_flag_arr == j] = 255
        cv2.imwrite(os.path.join(cache_path, result_name, obj_lst[j], file_name), predict_array[j, :, :])

    pred_color_arr = np.zeros((pred_flag_arr.shape[0], pred_flag_arr.shape[1], 3)).astype(np.uint8)
    pred_color_arr[:, :, 0][pred_flag_arr // 4 > 0] = 255
    pred_color_arr[:, :, 1][pred_flag_arr % 4 > 1] = 255
    pred_color_arr[:, :, -1][pred_flag_arr % 2 > 0] = 255

    if not os.path.exists(os.path.join(cache_path, "{}/all").format(result_name)):
        os.makedirs(os.path.join(cache_path, "{}/all".format(result_name)), exist_ok=True)
    cv2.imwrite(os.path.join(cache_path, "{}/all".format(result_name), file_name), pred_color_arr)



if __name__ == "__main__":
    # {"background": "black", "hemorrhages": "red", "hard_exudates": "green", "microaneurysms": "yellow", "disc": "blue", "soft_exudates": "pink"}
    obj_lst = ["background", "hemorrhages", "hard_exudates", "microaneurysms", "disc", "soft_exudates"]

    # U-Net from Yanmiao Bai
    start_t = time.time()
    pred_arr = segment_DR_unet("test_image.jpeg", gpu_ids="0")
    end_t = time.time()
    print("Time cost: %f s. " % (end_t - start_t))

    img_arr = show_seg("test_image.jpeg", pred_arr)
    if not os.path.exists("U-Net_results/visualization"):
        os.makedirs("U-Net_results/visualization")
    cv2.imwrite("U-Net_results/visualization/test_image.png", img_arr)

    predict_array = np.zeros_like(pred_arr).astype(np.uint8)
    pred_flag_arr = np.argmax(pred_arr, axis=0)

    for j in range(predict_array.shape[0]):
        if not os.path.exists(os.path.join("U-Net_results", obj_lst[j])):
            os.makedirs(os.path.join("U-Net_results", obj_lst[j]))
        predict_array[j, :, :][pred_flag_arr == j] = 255
        cv2.imwrite(os.path.join("U-Net_results", obj_lst[j], "test_image.png"), predict_array[j, :, :])

    pred_color_arr = np.zeros((pred_flag_arr.shape[0], pred_flag_arr.shape[1], 3)).astype(np.uint8)
    pred_color_arr[:, :, 0][pred_flag_arr // 4 > 0] = 255
    pred_color_arr[:, :, 1][pred_flag_arr % 4 > 1] = 255
    pred_color_arr[:, :, -1][pred_flag_arr % 2 > 0] = 255

    if not os.path.exists("U-Net_results/all"):
        os.makedirs("U-Net_results/all")
    cv2.imwrite("U-Net_results/all/test_image.png", pred_color_arr)

    # CAMWNet from Lianyu Wang
    start_t = time.time()
    pred_arr = segment_DR_camwnet("test_image.jpeg", gpu_ids="0")
    end_t = time.time()
    print("Time cost: %f s. " % (end_t - start_t))

    img_arr = show_seg("test_image.jpeg", pred_arr)
    if not os.path.exists("CAMWNet_results/visualization"):
        os.makedirs("CAMWNet_results/visualization")
    cv2.imwrite("CAMWNet_results/visualization/test_image.png", img_arr)

    predict_array = np.zeros_like(pred_arr).astype(np.uint8)
    pred_flag_arr = np.argmax(pred_arr, axis=0)

    for j in range(predict_array.shape[0]):
        if not os.path.exists(os.path.join("CAMWNet_results", obj_lst[j])):
            os.makedirs(os.path.join("CAMWNet_results", obj_lst[j]))
        predict_array[j, :, :][pred_flag_arr == j] = 255
        cv2.imwrite(os.path.join("CAMWNet_results", obj_lst[j], "test_image.png"), predict_array[j, :, :])

    pred_color_arr = np.zeros((pred_flag_arr.shape[0], pred_flag_arr.shape[1], 3)).astype(np.uint8)
    pred_color_arr[:, :, 0][pred_flag_arr // 4 > 0] = 255
    pred_color_arr[:, :, 1][pred_flag_arr % 4 > 1] = 255
    pred_color_arr[:, :, -1][pred_flag_arr % 2 > 0] = 255

    if not os.path.exists("CAMWNet_results/all"):
        os.makedirs("CAMWNet_results/all")
    cv2.imwrite("CAMWNet_results/all/test_image.png", pred_color_arr)
