# Performance Report

## Project

Number Plate Detection and OCR

## Final Pipeline

Vehicle image → YOLO plate detection → original plate crop → PaddleOCR → generic Indian registration cleanup → structured result → manual review → CSV export

## Dataset Summary

| Split | Images | Bounding Boxes |
|---|---:|---:|
| Training | 90 | 185 |
| Validation | 13 | 23 |
| Test | 12 | 25 |
| **Total** | **115** | **233** |

Additional audit results:

- Invalid annotations: 0
- Low-sharpness images reviewed: 8
- Images removed after review: 0

## Detector Performance

### Validation

| Metric | Result |
|---|---:|
| Precision | 99.11% |
| Recall | 86.96% |
| mAP@50 | 96.35% |
| mAP@50-95 | 52.16% |

### Untouched Test

| Metric | Result |
|---|---:|
| Precision | 93.05% |
| Recall | 60.00% |
| mAP@50 | 68.62% |
| mAP@50-95 | 38.32% |

The validation mAP@50 exceeded 95%, but this level was not maintained on the unseen test split. Test performance is therefore reported separately.

## Detection Operating Point

The confidence threshold was selected using validation data only and then frozen before test evaluation.

### Validation-selected threshold

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

### Frozen threshold applied to test

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

TP, FP, and FN use IoU ≥ 0.50 matching.

## Detector Failure Analysis

The main detector limitation is small-object detection.

Most false negatives occurred for:

- very small number plates
- distant vehicles
- wide street scenes
- partially visible plates
- difficult viewing angles

Some FP/FN pairs can also represent localization errors where a predicted box is close to the ground-truth plate but does not reach the IoU ≥ 0.50 matching threshold.

## Inference Speed

Final detector timing was verified on an NVIDIA Tesla T4.

| Metric | Result |
|---|---:|
| Mean YOLO inference | 19.73 ms/image |
| Median YOLO inference | 19.71 ms/image |
| Project target | <50 ms/image |
| Result | Achieved |

YOLO inference time is reported separately from OCR and Streamlit application wall-clock time.

## OCR Evaluation

A frozen benchmark of 19 verified readable plate crops was used.

OCR engines evaluated:

- PaddleOCR
- EasyOCR
- Tesseract

Preprocessing methods evaluated:

- Original
- Grayscale
- CLAHE
- Otsu
- Bilateral filtering
- Canny edge detection

### Selected OCR configuration

Original YOLO crop → PaddleOCR → generic Indian registration cleanup

### Final OCR results

| Metric | Result |
|---|---:|
| Strict full-plate exact match | 84.21% |
| Mean character accuracy | 93.63% |
| Raw exact match before cleanup | 47.37% |
| Benchmark size | 19 plates |

Character-level OCR accuracy exceeded 90%.

Strict exact-match accuracy was 84.21%, so the project does not claim more than 90% strict full-plate exact-match accuracy.

Generic post-processing improved exact match from 47.37% to 84.21%.

No benchmark-specific registration-number corrections were used.

## Final Assessment

The final system demonstrates an end-to-end number-plate detection and OCR workflow.

Strengths:

- high detector precision
- validation mAP@50 above 95%
- detector inference below 50 ms on Tesla T4
- OCR character accuracy above 90%
- working Streamlit application with upload, camera capture, manual review, and CSV export

Main limitation:

- reduced recall on unseen small or distant plates
