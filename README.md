[English](README.en.md) | [中文](README.md)
# 医疗眼部分析系统

## 项目简介

本项目是一个用于医疗眼部分析的系统，集成了图像增强、分割和分级等多种功能。系统基于深度学习技术，提供了多个服务模块，包括图像增强（MBEnhance）、图像分割（MBSeg）和图像分级（MBGrade）。每个模块都可以独立运行，并通过统一的任务管理接口进行协调。

## 主要功能

- **图像增强**：使用生成对抗网络（GAN）对输入的眼部图像进行增强，提高图像质量。
- **图像分割**：使用U-Net和CAMWNet等模型对眼部图像进行精确分割，识别不同的病变区域。
- **图像分级**：使用深度学习模型对眼部病变进行分级，帮助医生进行诊断。

## 系统架构

系统主要由以下几个部分组成：

- **Common**：包含通用工具类和配置信息。
- **Service**：包含各个服务模块，如图像增强、分割和分级。
- **ThirdPart**：包含第三方模型和工具，如StillGAN、Grading和Segmentation模型。
- **Docker**：包含各个服务模块的Docker配置文件，用于容器化部署。
- **UI**：前端用户界面，提供任务管理和结果展示功能。

## 安装指南

### 依赖项

- Python 3.8+
- PyTorch 1.11+
- FastAPI
- Docker

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/zhuyu12246/Medical-Eye-Analysis-System.git
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 构建Docker镜像：
   ```bash
   docker build -t mb-enhance -f Docker/Dockerfile-MBEnhance .
   docker build -t mb-grade -f Docker/Dockerfile-MBGrade .
   docker build -t mb-seg-unet -f Docker/Dockerfile-MBSegUNet .
   docker build -t mb-seg-camw -f Docker/Dockerfile-MBSegCamw .
   ```

4. 运行Docker容器：
   ```bash
   docker run -d -p 8001:8001 mb-enhance
   docker run -d -p 8002:8002 mb-grade
   docker run -d -p 8003:8003 mb-seg-unet
   docker run -d -p 8004:8004 mb-seg-camw
   ```

## 使用说明

### API 接口

- **图像增强**：`POST /run/mb/enhance`
- **图像分级**：`POST /run/mb/grade`
- **图像分割（U-Net）**：`POST /run/mb/seg/unet`
- **图像分割（CAMWNet）**：`POST /run/mb/seg/camw`

### 示例请求

```bash
curl -X POST "http://localhost:8001/run/mb/enhance" -H "Content-Type: multipart/form-data" -F "file=@test.jpg"
```


## 许可证

本项目使用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。
