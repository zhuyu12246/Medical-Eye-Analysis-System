[English](README.en.md) | [中文](README.md)
# Medical Eye Analysis System

## Project Overview

This project is a system designed for medical eye analysis, integrating multiple functions such as image enhancement, segmentation, and grading. Built on deep learning technologies, the system provides several service modules, including Image Enhancement (MBEnhance), Image Segmentation (MBSeg), and Image Grading (MBGrade). Each module can operate independently and is coordinated through a unified task management interface.

## Key Features

- **Image Enhancement**: Enhances input eye images using Generative Adversarial Networks (GANs) to improve image quality.
- **Image Segmentation**: Accurately segments eye images using models such as U-Net and CAMWNet to identify different pathological regions.
- **Image Grading**: Classifies eye pathologies using deep learning models to assist physicians in diagnosis.

## System Architecture

The system consists of the following main components:

- **Common**: Contains general utility classes and configuration information.
- **Service**: Includes individual service modules such as image enhancement, segmentation, and grading.
- **ThirdPart**: Contains third-party models and tools, such as StillGAN, Grading, and Segmentation models.
- **Docker**: Includes Docker configuration files for each service module, enabling containerized deployment.
- **UI**: Frontend user interface providing task management and result visualization functionality.

## Installation Guide

### Dependencies

- Python 3.8+
- PyTorch 1.11+
- FastAPI
- Docker

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/zhuyu12246/Medical-Eye-Analysis-System.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build Docker images:
   ```bash
   docker build -t mb-enhance -f Docker/Dockerfile-MBEnhance .
   docker build -t mb-grade -f Docker/Dockerfile-MBGrade .
   docker build -t mb-seg-unet -f Docker/Dockerfile-MBSegUNet .
   docker build -t mb-seg-camw -f Docker/Dockerfile-MBSegCamw .
   ```

4. Run Docker containers:
   ```bash
   docker run -d -p 8001:8001 mb-enhance
   docker run -d -p 8002:8002 mb-grade
   docker run -d -p 8003:8003 mb-seg-unet
   docker run -d -p 8004:8004 mb-seg-camw
   ```

## Usage Instructions

### API Endpoints

- **Image Enhancement**: `POST /run/mb/enhance`
- **Image Grading**: `POST /run/mb/grade`
- **Image Segmentation (U-Net)**: `POST /run/mb/seg/unet`
- **Image Segmentation (CAMWNet)**: `POST /run/mb/seg/camw`

### Example Request

```bash
curl -X POST "http://localhost:8001/run/mb/enhance" -H "Content-Type: multipart/form-data" -F "file=@test.jpg"
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
