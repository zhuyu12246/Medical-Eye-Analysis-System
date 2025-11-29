# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np


# 计算图像中每种病灶相关参数
def analyse_DR(pred_arr, lesion_mapping={"hemorrhages": 1, "hard_exudates": 2, "microaneurysms": 3, "soft_exudates": 5}):
    lesion_dct = {}
    for lesion, idx in lesion_mapping.items():
        lesion_dct[lesion] = {}
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(pred_arr[idx, :, :], connectivity=8)
        S = 0
        lesion_lst = []
        if num_labels > 1:
            S = stats[1:, -1].mean()
            for j in range(1, num_labels):
                lesion_lst.append((round(centroids[j, 0]), round(centroids[j, 1]), stats[j, -1]))
        lesion_dct[lesion]["objects"] = lesion_lst  # 计算该类病灶每个目标中心点横纵坐标及面积
        lesion_dct[lesion]["counts"] = num_labels - 1  # 统计该类病灶个数
        lesion_dct[lesion]["avgS"] = S  # 计算该类病灶的平均面积
    
    return lesion_dct


if __name__ == "__main__":
    obj_lst = ["background", "hemorrhages", "hard_exudates", "microaneurysms", "disc", "soft_exudates"]
    lesion_mapping = {"hemorrhages": 1, "hard_exudates": 2, "microaneurysms": 3, "soft_exudates": 5}
    img_name = "test_image"  # 图像名字,可修改
    
    # U-Net from Yanmiao Bai
    results_dir = "U-Net_results"
    img_lst = []
    for obj in obj_lst:
        img_lst.append(cv2.imread(os.path.join(results_dir, obj, img_name + ".png"), 0))
    img_arr = np.array(img_lst, np.uint8)
    
    start_t = time.time()
    lesion_dct = analyse_DR(img_arr, lesion_mapping)
    end_t = time.time()
    print("Time cost: %f s. " % (end_t - start_t))
    
    for lesion, att in lesion_dct.items():
        if not os.path.exists(os.path.join(results_dir, "statistics", lesion)):
            os.makedirs(os.path.join(results_dir, "statistics", lesion))
        f = open(os.path.join(results_dir, "statistics", lesion, img_name + ".csv"), "w")
        f.write("x,y,S\n")
        for obj in lesion_dct[lesion]["objects"]:
            f.write("%d,%d,%d\n" % (obj[0], obj[1], obj[2]))  # x, y, S
        f.close()
        f = open(os.path.join(results_dir, "statistics", lesion, "summary.csv"), "w")
        f.write("img_name,N,S\n")
        f.write("%s,%d,%f\n" % (img_name, lesion_dct[lesion]["counts"], lesion_dct[lesion]["avgS"]))
        f.close()
    
    # CAMWNet from Lianyu Wang
    results_dir = "CAMWNet_results"
    img_lst = []
    for obj in obj_lst:
        img_lst.append(cv2.imread(os.path.join(results_dir, obj, img_name + ".png"), 0))
    img_arr = np.array(img_lst, np.uint8)
    
    start_t = time.time()
    lesion_dct = analyse_DR(img_arr, lesion_mapping)
    end_t = time.time()
    print("Time cost: %f s. " % (end_t - start_t))
    
    for lesion, att in lesion_dct.items():
        if not os.path.exists(os.path.join(results_dir, "statistics", lesion)):
            os.makedirs(os.path.join(results_dir, "statistics", lesion))
        f = open(os.path.join(results_dir, "statistics", lesion, img_name + ".csv"), "w")
        f.write("x,y,S\n")
        for obj in lesion_dct[lesion]["objects"]:
            f.write("%d,%d,%d\n" % (obj[0], obj[1], obj[2]))  # x, y, S
        f.close()
        f = open(os.path.join(results_dir, "statistics", lesion, "summary.csv"), "w")
        f.write("img_name,N,S\n")
        f.write("%s,%d,%f\n" % (img_name, lesion_dct[lesion]["counts"], lesion_dct[lesion]["avgS"]))
        f.close()
