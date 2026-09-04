# Number Plate Detection and OCR

End-to-end Indian number-plate detection project using YOLO, PaddleOCR, OpenCV, and Streamlit.

The final production pipeline is:

Vehicle image  

→ YOLO number-plate detection  

→ original detected plate crop  

→ PaddleOCR  

→ generic Indian registration cleanup  

→ structured OCR result  

→ optional manual review  

→ CSV export

The application supports image upload and camera capture, displays detected plate regions and OCR text, provides a structured review table, and allows users to download reviewed detection logs.

---

## Key Features

- YOLO-based number-plate detection

- Streamlit image upload

- Camera capture using the device webcam

- Bounding-box visualization

- Enlarged detected plate crops

- PaddleOCR text extraction

- Generic Indian registration-format cleanup

- OCR confidence and review status

- Manual correction of OCR output

- CSV export of reviewed results

- Detector and OCR performance dashboard

- Detector inference-time reporting

- Validation-selected confidence threshold

---

## Final Production Configuration

### Detector

- Model: `models/best.pt`

- Input size: `832`

- Selected confidence threshold: `0.50`

- Threshold selected on: validation split

- Threshold applied once to: untouched test split

### OCR

- OCR engine: PaddleOCR

- PaddlePaddle: `3.2.0`

- PaddleOCR: `3.2.0`

- OCR input: original YOLO plate crop

- Post-processing: generic Indian registration cleanup

- No benchmark-specific registration corrections are used

---

## Dataset

The repository retains the original project-provided dataset for traceability and reproducibility.

### Original / Base Dataset

The original supplied project dataset is stored under:

```text
data/raw/NumberPlate Dataset with annotation/
```

This folder contains the base data provided with the project. It is preserved separately from the processed train/validation/test dataset so that the transformation from source data to model-ready data remains clear.

### Final Processed Dataset Audit

| Split | Images | Bounding Boxes |
|---|---:|---:|
| Training | 90 | 185 |
| Validation | 13 | 23 |
| Test | 12 | 25 |
| **Total** | **115** | **233** |

Additional audit results:

- Invalid annotations: `0`
- Low-sharpness images flagged for review: `8`
- Images removed after review: `0`

Difficult but valid images were retained so that the final evaluation remained realistic.

### Dataset Organization

```text
data/
├── README.md
├── raw/
│   └── NumberPlate Dataset with annotation/
│       ├── dataset_v1/
│       ├── images/
│       ├── yolo_labels/
│       └── best.pt
├── processed/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── ocr_benchmark/
    ├── crops/
    ├── ground_truth.csv
    └── summary.csv
```

The original `best.pt` inside the raw project-provided folder is retained only as part of the supplied base material. The final model selected for this project is stored separately at:

```text
models/best.pt
```

---

## Detector Evaluation

### Validation Results

| Metric | Result |

|---|---:|

| Precision | 99.11% |

| Recall | 86.96% |

| mAP@50 | 96.35% |

| mAP@50-95 | 52.16% |

### Untouched Test Results

| Metric | Result |

|---|---:|

| Precision | 93.05% |

| Recall | 60.00% |

| mAP@50 | 68.62% |

| mAP@50-95 | 38.32% |

The validation mAP@50 exceeded 95%, but this level was not maintained on the unseen test split. Test performance is therefore reported separately and transparently.

---

## Detection Operating Point

The confidence threshold was selected using the validation split only.

### Selected Validation Operating Point

| Metric | Result |

|---|---:|

| Confidence threshold | 0.50 |

| TP | 20 |

| FP | 1 |

| FN | 3 |

| Precision | 95.24% |

| Recall | 86.96% |

| F1 | 90.91% |

| Mean matched IoU | 0.775 |

The threshold was then frozen before test evaluation.

### Frozen Threshold Applied to Test

| Metric | Result |

|---|---:|

| Confidence threshold | 0.50 |

| TP | 14 |

| FP | 1 |

| FN | 11 |

| Precision | 93.33% |

| Recall | 56.00% |

| F1 | 70.00% |

| Mean matched IoU | 0.786 |

TP, FP, and FN are calculated using IoU ≥ 0.50 matching.

---

## Detection Failure Analysis

The main detector limitation is small-object detection.

Most false negatives occurred for:

- very small number plates

- distant vehicles

- wide street scenes

- partially visible plates

- difficult viewing angles

One test image contained several extremely small annotated plates and contributed disproportionately to the false-negative count.

Some FP/FN pairs can also represent localization errors where a predicted box is close to the correct plate but does not reach the IoU ≥ 0.50 matching requirement.

---

## Detector Inference Speed

Final GPU timing was performed on an NVIDIA Tesla T4.

| Metric | Result |

|---|---:|

| Mean YOLO inference | 19.73 ms/image |

| Median YOLO inference | 19.71 ms/image |

| Project target | <50 ms/image |

| Result | ACHIEVED |

YOLO inference timing is reported separately from OCR processing and Streamlit end-to-end application time.

---

## OCR Evaluation

A frozen benchmark of 19 verified readable number-plate crops was used.

### OCR Engines Evaluated

- PaddleOCR

- EasyOCR

- Tesseract

### Preprocessing Methods Evaluated

- Original

- Grayscale

- CLAHE

- Otsu

- Bilateral filtering

- Canny edge detection

### Selected OCR Pipeline

Original YOLO crop  

→ PaddleOCR  

→ generic Indian registration cleanup

### Final OCR Results

| Metric | Result |

|---|---:|

| Strict full-plate exact match | 84.21% |

| Mean character accuracy | 93.63% |

| Raw exact match before cleanup | 47.37% |

| Benchmark size | 19 plates |

Character-level OCR accuracy exceeded 90%.

Strict full-plate exact-match accuracy was 84.21%, so the project does not claim more than 90% strict exact-match accuracy.

Generic post-processing improved strict exact match from 47.37% to 84.21%.

Examples of generic cleanup include removal of:

- `IND`

- unrelated surrounding OCR text

- extra fragments outside a valid Indian registration pattern

No benchmark-specific registration-number corrections were used.

---

## Streamlit Application

The final Streamlit application provides the following features.

### Image Input

- Upload JPG, JPEG, or PNG images

- Camera capture using the connected webcam

### Detection Output

- Number-plate bounding boxes

- Detection confidence

- Enlarged detected plate crops

### OCR Output

- PaddleOCR extracted text

- OCR confidence

- Review status

- Editable reviewed plate value

### Export

- Download reviewed results as CSV

### Performance Dashboard

The application displays:

- test mAP@50

- test mAP@50-95

- test recall

- operating-point precision

- operating-point recall

- F1

- mean matched IoU

- OCR character accuracy

- OCR exact-match accuracy

- Tesla T4 inference time

---

## Deployment Files

Docker is not required for the current project submission.

The required setup/deployment files are:

- `requirements.txt`
- `.streamlit/config.toml`
- `configs/data.yaml`

A `Dockerfile`, `.dockerignore`, and `packages.txt` are intentionally omitted because the final application runs directly with Streamlit and uses PaddleOCR rather than Tesseract.

Docker can be added later if the application is deployed to a container-based platform such as AWS ECS, Azure Container Apps, Google Cloud Run, or Kubernetes.

---

## Webcam Note

The Streamlit prototype currently supports webcam/camera capture using `st.camera_input()`.

This captures a frame from the connected camera and runs the complete YOLO + PaddleOCR pipeline.

Continuous live-video inference would require an additional streaming component such as WebRTC and is retained as a future deployment enhancement.

---

## Run Locally

### Recommended Python Version

Python 3.12

### Create Virtual Environment

Windows PowerShell:

```powershell

py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

```

Upgrade pip:

```powershell

python -m pip install --upgrade pip

```

Install dependencies:

```powershell

python -m pip install -r requirements.txt

```

Start Streamlit:

```powershell

streamlit run app.py

```

Place the selected YOLO model at:

```text

models/best.pt

```

---

## Tested OCR Environment

The confirmed working local OCR environment uses:

```text

PaddlePaddle 3.2.0

PaddleOCR 3.2.0

```

On Windows, PyTorch and Ultralytics are imported before PaddleOCR to avoid DLL loading-order issues encountered during local setup.

---

## Repository Structure

```text
DS-Number-Plate-Detection-of-Vehicles/
│
├── .gitignore
├── README.md
├── PROJECT_CHECKLIST.md
├── requirements.txt
├── app.py
│
├── .streamlit/
│   └── config.toml
│
├── configs/
│   └── data.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── NumberPlate Dataset with annotation/
│   │       ├── dataset_v1/
│   │       ├── images/
│   │       ├── yolo_labels/
│   │       └── best.pt
│   ├── processed/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   └── ocr_benchmark/
│       ├── crops/
│       ├── ground_truth.csv
│       └── summary.csv
│
├── models/
│   └── best.pt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_on_colab.ipynb
│   ├── 03_ocr_evaluation.ipynb
│   └── 04_final_evaluation.ipynb
│
├── reports/
│   ├── eda_final/
│   ├── yolo_test_evaluation/
│   ├── detection_operating_point/
│   └── ocr_final/
│
├── src/
│   ├── __init__.py
│   └── ocr_pipeline.py
│
└── docs/
    ├── PERFORMANCE_REPORT.md
    └── DEPLOYMENT_GUIDE.md
```

Notes:

- `.venv/` is intentionally excluded from GitHub through `.gitignore`.
- `data/raw/` contains the original project-provided dataset and is kept for traceability.
- `data/processed/` contains the final train/validation/test dataset used for model development and evaluation.
- `data/ocr_benchmark/` contains the frozen OCR benchmark used for the reported OCR evaluation.
- The `best.pt` inside `data/raw/...` is part of the supplied base material; the final selected project model is `models/best.pt`.
- `tests/` is not included because this project does not currently use an automated test suite.
- `Dockerfile` and `.dockerignore` are not required for the current Streamlit-based submission.
- `configs/final_eval_data.yaml` is not required as a permanent repository file if the final evaluation notebook creates its evaluation configuration dynamically.
- Only final evaluation outputs should be kept under `reports/`; temporary experiments and duplicate outputs should not be committed.

---

## Notebook Workflow

### 01 — EDA

`notebooks/01_eda.ipynb`

Covers:

- dataset audit

- annotation validation

- split distribution

- image-quality analysis

### 02 — YOLO Training

`notebooks/02_train_on_colab.ipynb`

Covers:

- YOLO training

- model comparison

- model selection

- validation and test evaluation

### 03 — OCR Evaluation

`notebooks/03_ocr_evaluation.ipynb`

Covers:

- OCR engine comparison

- preprocessing comparison

- generic OCR cleanup

- frozen OCR benchmark

### 04 — Final Evaluation

`notebooks/04_final_evaluation.ipynb`

Covers:

- detector metrics

- validation-only threshold selection

- frozen test operating-point evaluation

- TP / FP / FN

- IoU

- failure analysis

- Tesla T4 inference timing

---

## Business Use Cases

The prototype can support human-reviewed workflows for:

- parking management

- toll collection

- fleet monitoring

- traffic analytics

- security checkpoints

- vehicle-entry systems

- traffic management

Production deployment would require:

- a larger representative dataset

- privacy and legal review

- access controls

- data-retention policies

- stronger generalization testing

- monitoring

- human review of uncertain OCR output

---

## Known Limitations

- The dataset is relatively small.

- Test recall is lower than validation recall.

- Tiny and distant number plates can be missed.

- OCR can fail with blur, glare, occlusion, extreme angles, or low resolution.

- OCR confidence is not an accuracy guarantee.

- Camera input currently captures individual frames rather than continuous live video.

- End-to-end application speed depends on system hardware and the number of detected plates.

---

## Project Deliverables

- [x] Original project-provided base dataset retained
- [x] Preprocessed train/validation/test dataset
- [x] Trained and selected YOLO model
- [x] Streamlit web application
- [x] PaddleOCR integration
- [x] Detector and OCR performance evaluation
- [x] Structured source-code repository
- [x] Local setup instructions
- [x] `data/README.md`
- [x] Frozen OCR benchmark copied into `data/ocr_benchmark/`
- [x] Final detector/OCR report outputs copied into `reports/`
- [x] `docs/PERFORMANCE_REPORT.md`
- [x] `docs/DEPLOYMENT_GUIDE.md`
- [ ] Final demo video link
- [ ] Final GitHub push and fresh-clone verification

---

## Demo Video

Demo video link:

To be added after final recording.

The final demo will show:

1. Project objective

2. Streamlit interface

3. Image upload

4. Number-plate detection

5. PaddleOCR extraction

6. Structured results

7. Camera capture

8. CSV export

9. Final performance metrics

10. Project limitations

---

## Final Summary

The final system integrates YOLO-based number-plate detection and PaddleOCR into an interactive Streamlit application.

Final detector performance:

- Test precision: 93.05%

- Test recall: 60.00%

- Test mAP@50: 68.62%

- Test mAP@50-95: 38.32%

- Mean Tesla T4 inference: 19.73 ms/image

At the validation-selected operating threshold of 0.50, the frozen test evaluation achieved:

- Precision: 93.33%

- Recall: 56.00%

- F1: 70.00%

- Mean matched IoU: 0.786

Final OCR performance:

- Mean character accuracy: 93.63%

- Strict full-plate exact match: 84.21%

The application demonstrates a complete workflow from image input to number-plate detection, OCR extraction, manual review, and structured CSV export.
