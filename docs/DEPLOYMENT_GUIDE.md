# Deployment Guide

## Overview

This project runs as a Streamlit application using YOLO for number-plate detection and PaddleOCR for OCR.

The current submission does not require Docker.

## Required Files

The following files must be present in the repository:

```text
app.py
requirements.txt
models/best.pt
src/ocr_pipeline.py
.streamlit/config.toml
configs/data.yaml
```

## Recommended Python Version

Python 3.12

## Local Setup

### 1. Clone the repository

```powershell
git clone https://github.com/David-JR-DS/DS-Number-Plate-Detection-of-Vehicles.git
cd DS-Number-Plate-Detection-of-Vehicles
```

### 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The confirmed OCR environment uses:

```text
PaddlePaddle 3.2.0
PaddleOCR 3.2.0
```

## Windows Import-order Note

On the tested Windows setup, PyTorch and Ultralytics should be imported before PaddleOCR to avoid a DLL loading-order issue.

The final `app.py` already follows the working import order.

## Run the Application

```powershell
streamlit run app.py
```

## Application Features

The deployed application supports:

- JPG/JPEG/PNG upload
- camera frame capture
- YOLO plate detection
- plate crop display
- PaddleOCR extraction
- OCR confidence display
- manual OCR correction
- structured result table
- CSV export
- performance dashboard

## Camera Limitation

The current application uses Streamlit camera capture and processes a captured frame.

Continuous live-video inference is not implemented in the final submission.

A WebRTC-based streaming component can be added as a future enhancement.

## Model Location

The final selected detector must be located at:

```text
models/best.pt
```

## Streamlit Configuration

Expected `.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#2E7D32"

[server]
maxUploadSize = 50
```

## Dataset Configuration

Expected `configs/data.yaml`:

```yaml
path: ../data/processed

train: train/images
val: val/images
test: test/images

names:
  0: number_plate
```

## Docker

Docker is not required for the current project submission.

A Dockerfile can be added later if the application is deployed to a container platform such as:

- AWS ECS
- Azure Container Apps
- Google Cloud Run
- Kubernetes

## Fresh-clone Verification

Before final submission:

1. Clone the GitHub repository into a new folder.
2. Create a new virtual environment.
3. Install `requirements.txt`.
4. Confirm `models/best.pt` exists.
5. Run `streamlit run app.py`.
6. Test image upload.
7. Test PaddleOCR output.
8. Test camera capture.
9. Test CSV export.
