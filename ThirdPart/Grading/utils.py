# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch.nn as nn
from skimage import measure
from imutils import contours, grab_contours


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        # 先将图片转换成灰度
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 设置遮罩，255为纯白色， 0为纯黑色
        # 其实这个mask是过滤掉一些黑色像素
        mask = gray_img > tol

        # np.ix_([a1,a2,a3,...],[b1,b2,b3,...]): 讲一个数组 1、选取其中的a1,a2,a3列， 然后将每列元素以b1,b2,b3方式重新排列
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def circle_crop(img, sigmaX=30):
    """
    Create circular crop around image centre
    """

    img = cv2.imread(img)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    # bitwise_and 来裁剪原始图像，得到一个圆形图像
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)  # ##
    # print(img.shape)

    return img


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.cpu().numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def crop_disc(img_path, th=200):
    # load the image, convert it to grayscale, and blur it
    image = cv2.imread(img_path)
    # image = cv2.resize(image, (800, 615))##
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    median = cv2.medianBlur(blur, 5)
    
    
    # cv2.imshow("median", median)##
    
    # cv2.imshow()
    
    # threshold the image to reveal light regions in the blurred image
    thresh = cv2.threshold(median, th, 255, cv2.THRESH_BINARY)[1]  # 155, 255##
    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    
    # perform a connected component analysis on the thresholded image
    # then initialize a mask to store only the "large" components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        
        # otherwise, construct the label mask and count the number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        
        # if the number of pixels in the component is sufficiently large
        # then add it to our mask of "large blobs"
        if numPixels >300:
            mask = cv2.add(mask, labelMask)
        
    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    crop_image = image.copy()
    # loop over the contours
    for (i, c) in enumerate(cnts):
        ellipse = cv2.fitEllipse(c)
        (x, y, w, h) = cv2.boundingRect(c)
        # print(ellipse, x, y, w, h)##
        # cv2.putText(image, "#{}".format(i + 1), (x, y - 15),##
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)##
        cv2.ellipse(image, ellipse, (0, 255, 0), 5)
        break
    
    center_x = int(ellipse[0][0])
    center_y = int(ellipse[0][1])
    max_w = max(w, h)
    max_h = max(w, h)
    cv2.rectangle(image, (max(center_x-max_w//2, 0), max(center_y-max_h//2, 0)), (min(max(center_x-max_w//2, 0)+max_w, W), min(max(center_y-max_h//2, 0)+max_h, H)), (0, 255, 0), 5)
    
    return crop_image[max(center_y-max_h//2, 0):min(max(center_y-max_h//2, 0)+max_h, H), max(center_x-max_w//2, 0):min(max(center_x-max_w//2, 0)+max_w, W), :], image


if __name__ == "__main__":
    crop_arr, img_arr = crop_disc("6_right.png", th=200)
    cv2.imwrite("crop_6_right.png", crop_arr)
    cv2.imwrite("vis_6_right.png", img_arr)
