# Dataset

This directory contains the original source data, the final processed YOLO dataset, and the frozen OCR benchmark used in the project.

## Original Project Dataset

The original dataset supplied with the project is retained under:

`data/raw/NumberPlate Dataset with annotation/`

It is preserved separately for traceability and is not modified in place.

## Processed Dataset

After dataset auditing and annotation validation, the final model-ready dataset was organized into train, validation, and test splits.

| Split | Images | Bounding Boxes |
|---|---:|---:|
| Training | 90 | 185 |
| Validation | 13 | 23 |
| Test | 12 | 25 |
| **Total** | **115** | **233** |

Dataset audit summary:

- Invalid annotations: 0
- Low-sharpness images reviewed: 8
- Images removed after review: 0

Difficult but valid images were retained to preserve realistic evaluation conditions.

The final processed dataset is stored under:

`data/processed/`

```text
processed/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## OCR Benchmark

The final OCR benchmark contains 19 manually verified readable number-plate crops.

It is stored under:

`data/ocr_benchmark/`

Expected contents:

```text
ocr_benchmark/
├── crops/
├── ground_truth.csv
└── summary.csv
```

`ground_truth.csv` contains the crop filename, verified registration number, OCR prediction, OCR confidence, exact-match result, and character-level accuracy.

`summary.csv` contains the aggregate OCR evaluation summary.

## Model Note

A `best.pt` file may exist inside the original supplied raw dataset. That file is retained only as part of the original project material.

The final selected detector for this project is stored separately at:

`models/best.pt`
