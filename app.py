"""Streamlit interface for number-plate detection and PaddleOCR."""

from __future__ import annotations

import hashlib
import os
import time
from io import BytesIO
from pathlib import Path


# ============================================================
# WINDOWS / PADDLE CPU COMPATIBILITY
# ============================================================
# Must be set BEFORE importing PaddleOCR/Paddle.
# Prevents the OneDNN fused_conv2d error seen on Windows CPU.

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"


# ============================================================
# IMPORTS
# ============================================================

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# IMPORTANT:
# Load PyTorch before PaddleOCR on Windows.
import torch
from ultralytics import YOLO

from paddleocr import PaddleOCR
from PIL import Image

from src.ocr_pipeline import (
    canonicalise_plate_candidate,
    classify_ocr_text,
    normalise_plate_text,
    read_plate,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_DETECTION_CONFIDENCE = 0.50


# ============================================================
# FINAL FROZEN EVALUATION RESULTS
# ============================================================

TEST_MAP50 = 68.62
TEST_MAP50_95 = 38.32
TEST_YOLO_RECALL = 60.00

OPERATING_PRECISION = 93.33
OPERATING_RECALL = 56.00
OPERATING_F1 = 70.00
MEAN_MATCHED_IOU = 0.786

OCR_CHARACTER_ACCURACY = 93.63
OCR_EXACT_MATCH = 84.21

T4_MEAN_INFERENCE_MS = 19.73


# ============================================================
# MODEL PATH
# ============================================================

def find_model_path() -> Path | None:

    configured = os.getenv("PLATE_MODEL_PATH")

    candidates = [
        Path(configured) if configured else None,
        PROJECT_DIR / "models" / "best.pt",
    ]

    return next(
        (
            path
            for path in candidates
            if path and path.is_file()
        ),
        None,
    )


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource(
    show_spinner="Loading YOLO detector..."
)
def load_detector(
    model_path: str,
) -> YOLO:

    return YOLO(model_path)


# ============================================================
# LOAD PADDLEOCR
# ============================================================

@st.cache_resource(
    show_spinner="Loading PaddleOCR (first run may download OCR model files)..."
)
def load_ocr_reader():
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # --------------------------------------------------------
    # PaddleOCR 2.x / Windows
    # --------------------------------------------------------

    try:

        return PaddleOCR(
            lang="en",
            use_angle_cls=False,
            use_gpu=False,
            enable_mkldnn=False,
            show_log=False,
        )

    except TypeError:

        # ----------------------------------------------------
        # PaddleOCR 3.x compatibility
        # ----------------------------------------------------

        return PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )


# ============================================================
# RESULT TABLE
# ============================================================

RESULT_COLUMNS = [
    "plate",
    "detection_confidence",
    "ocr_candidate",
    "ocr_confidence",
    "ocr_status",
    "reviewed_plate",
]


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyse_image(
    image_bgr: np.ndarray,
    detector: YOLO,
    confidence: float,
    use_ocr: bool,
    ocr_confidence_threshold: float,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    list[np.ndarray],
    float,
]:

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    detector_start = time.perf_counter()

    result = detector.predict(
        source=image_bgr,
        conf=confidence,
        imgsz=832,
        verbose=False,
    )[0]

    detector_wall_ms = (
        time.perf_counter()
        - detector_start
    ) * 1000.0


    annotated = image_bgr.copy()

    height, width = annotated.shape[:2]

    records = []

    plate_crops = []


    # --------------------------------------------------------
    # NO DETECTIONS
    # --------------------------------------------------------

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return (
            annotated,
            pd.DataFrame(
                columns=RESULT_COLUMNS
            ),
            plate_crops,
            detector_wall_ms,
        )


    # --------------------------------------------------------
    # LOAD OCR ONLY WHEN REQUIRED
    # --------------------------------------------------------

    reader = (
        load_ocr_reader()
        if use_ocr
        else None
    )


    boxes = (
        result.boxes.xyxy
        .cpu()
        .tolist()
    )

    scores = (
        result.boxes.conf
        .cpu()
        .tolist()
    )


    # --------------------------------------------------------
    # PROCESS EACH DETECTED PLATE
    # --------------------------------------------------------

    for index, (
        xyxy,
        score,
    ) in enumerate(
        zip(
            boxes,
            scores,
        ),
        start=1,
    ):

        x1 = max(
            0,
            int(xyxy[0]),
        )

        y1 = max(
            0,
            int(xyxy[1]),
        )

        x2 = min(
            width,
            int(xyxy[2]),
        )

        y2 = min(
            height,
            int(xyxy[3]),
        )


        if (
            x2 <= x1
            or y2 <= y1
        ):
            continue


        crop = image_bgr[
            y1:y2,
            x1:x2,
        ]


        raw_candidate = ""

        ocr_confidence = 0.0


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        if (
            reader is not None
            and crop.size
        ):

            try:

                (
                    raw_candidate,
                    ocr_confidence,
                    crop_rgb,
                ) = read_plate(
                    reader,
                    crop,
                )

            except Exception as exc:

                # Do not crash the entire Streamlit app
                # because one OCR crop fails.

                crop_rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB,
                )

                raw_candidate = ""

                ocr_confidence = 0.0

                st.warning(
                    f"OCR failed for plate "
                    f"{index}: {exc}"
                )

        else:

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB,
            )


        plate_crops.append(
            crop_rgb
        )


        # ----------------------------------------------------
        # CLEAN OCR TEXT
        # ----------------------------------------------------

        candidate = (
            canonicalise_plate_candidate(
                raw_candidate
            )
        )


        if use_ocr:

            ocr_status = (
                classify_ocr_text(
                    candidate,
                    ocr_confidence,
                    ocr_confidence_threshold,
                )
            )

        else:

            ocr_status = (
                "OCR disabled"
            )


        reviewed_plate = (
            candidate
            if ocr_status
            == "Accepted OCR"
            else ""
        )


        # ----------------------------------------------------
        # ANNOTATION
        # ----------------------------------------------------

        if candidate:

            label = candidate

        else:

            label = (
                f"Plate {index} "
                f"({float(score):.2f})"
            )


        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 200, 80),
            2,
        )


        cv2.putText(
            annotated,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 80),
            2,
            cv2.LINE_AA,
        )


        # ----------------------------------------------------
        # STRUCTURED RESULT
        # ----------------------------------------------------

        records.append(
            {
                "plate": index,

                "detection_confidence":
                    round(
                        float(score),
                        3,
                    ),

                "ocr_candidate":
                    candidate
                    or "Unreadable",

                "ocr_confidence":
                    round(
                        float(
                            ocr_confidence
                        ),
                        3,
                    )
                    if candidate
                    else None,

                "ocr_status":
                    ocr_status,

                "reviewed_plate":
                    reviewed_plate,
            }
        )


    return (
        annotated,
        pd.DataFrame(
            records
        ),
        plate_crops,
        detector_wall_ms,
    )


# ============================================================
# EXPORT TABLE
# ============================================================

def build_export_table(
    edited_table: pd.DataFrame,
) -> pd.DataFrame:

    export = (
        edited_table.copy()
    )


    export[
        "final_plate"
    ] = export[
        "reviewed_plate"
    ].map(
        normalise_plate_text
    )


    export[
        "review_status"
    ] = np.where(

        export[
            "final_plate"
        ].eq(""),

        "Needs review",

        np.where(

            export[
                "final_plate"
            ].eq(

                export[
                    "ocr_candidate"
                ].map(
                    normalise_plate_text
                )

            ),

            "Accepted OCR",

            "Manually corrected",
        ),
    )


    return export[
        [
            "plate",
            "detection_confidence",
            "ocr_candidate",
            "ocr_confidence",
            "ocr_status",
            "final_plate",
            "review_status",
        ]
    ]


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title=(
        "Number Plate Detection & OCR"
    ),
    page_icon="🚗",
    layout="wide",
)


st.session_state.setdefault(
    "analysis",
    None,
)

st.session_state.setdefault(
    "analysis_id",
    0,
)


st.title(
    "Number Plate Detection & OCR"
)


st.caption(
    "YOLO number-plate detection with "
    "PaddleOCR text extraction. "
    "OCR results should be visually "
    "verified before export."
)


# ============================================================
# PERFORMANCE DASHBOARD
# ============================================================

with st.expander(
    "Model Performance",
    expanded=True,
):

    st.caption(
        "Detector metrics use the frozen "
        "untouched test split. "
        "Confidence 0.50 was selected on "
        "validation and frozen before "
        "test evaluation."
    )


    row1 = st.columns(5)

    row1[0].metric(
        "Test mAP@50",
        f"{TEST_MAP50:.2f}%",
    )

    row1[1].metric(
        "Test mAP@50–95",
        f"{TEST_MAP50_95:.2f}%",
    )

    row1[2].metric(
        "Test recall",
        f"{TEST_YOLO_RECALL:.2f}%",
    )

    row1[3].metric(
        "Mean matched IoU",
        f"{MEAN_MATCHED_IOU:.3f}",
    )

    row1[4].metric(
        "T4 inference",
        f"{T4_MEAN_INFERENCE_MS:.2f} ms",
    )


    row2 = st.columns(5)

    row2[0].metric(
        "Precision @ conf 0.50",
        f"{OPERATING_PRECISION:.2f}%",
    )

    row2[1].metric(
        "Recall @ conf 0.50",
        f"{OPERATING_RECALL:.2f}%",
    )

    row2[2].metric(
        "F1 @ conf 0.50",
        f"{OPERATING_F1:.2f}%",
    )

    row2[3].metric(
        "OCR character accuracy",
        f"{OCR_CHARACTER_ACCURACY:.2f}%",
    )

    row2[4].metric(
        "OCR exact match",
        f"{OCR_EXACT_MATCH:.2f}%",
    )


    st.caption(
        "OCR benchmark: PaddleOCR on the "
        "original YOLO plate crop with "
        "generic Indian registration cleanup. "
        "YOLO timing excludes OCR."
    )


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header(
        "Detection settings"
    )


    detection_confidence = (
        st.slider(
            "Minimum detection confidence",
            min_value=0.10,
            max_value=0.90,
            value=(
                DEFAULT_DETECTION_CONFIDENCE
            ),
            step=0.05,
            help=(
                "0.50 is the validation-selected "
                "operating threshold."
            ),
        )
    )


    enable_ocr = st.checkbox(
        "Read detected plates with PaddleOCR",
        value=True,
    )


    ocr_confidence_threshold = (
        st.slider(
            "Minimum OCR confidence "
            "for auto-accept",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            disabled=(
                not enable_ocr
            ),
        )
    )


    st.caption(
        "Lower detection thresholds can "
        "recover faint or distant plates "
        "but may increase false positives."
    )


# ============================================================
# MODEL CHECK
# ============================================================

model_path = (
    find_model_path()
)


if model_path is None:

    st.error(
        "Model file not found. "
        "Place the selected model at "
        "`models/best.pt`."
    )

    st.stop()


# ============================================================
# IMAGE INPUT
# ============================================================

source_mode = st.radio(
    "Image source",
    [
        "Upload image",
        "Camera capture",
    ],
    horizontal=True,
)


if (
    source_mode
    == "Upload image"
):

    uploaded = st.file_uploader(
        "Upload a vehicle image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="image_upload",
    )

else:

    uploaded = st.camera_input(
        "Capture a vehicle image",
        key="camera_capture",
    )


if uploaded is None:

    st.info(
        "Upload or capture an image "
        "to begin."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

uploaded_bytes = (
    uploaded.getvalue()
)


source_id = hashlib.sha256(
    uploaded_bytes
).hexdigest()


image_rgb = np.array(

    Image.open(
        BytesIO(
            uploaded_bytes
        )
    ).convert(
        "RGB"
    )

)


image_bgr = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR,
)


st.image(
    image_rgb,
    caption="Input image",
    use_container_width=True,
)


# ============================================================
# RUN DETECTION
# ============================================================

if st.button(
    "Run detection",
    type="primary",
):

    detector = load_detector(
        str(
            model_path
        )
    )


    total_start = (
        time.perf_counter()
    )


    (
        annotated_bgr,
        result_table,
        plate_crops,
        detector_wall_ms,
    ) = analyse_image(

        image_bgr=image_bgr,

        detector=detector,

        confidence=(
            detection_confidence
        ),

        use_ocr=enable_ocr,

        ocr_confidence_threshold=(
            ocr_confidence_threshold
        ),
    )


    total_elapsed_ms = (
        time.perf_counter()
        - total_start
    ) * 1000.0


    annotated_rgb = (
        cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB,
        )
    )


    st.session_state[
        "analysis_id"
    ] += 1


    st.session_state[
        "analysis"
    ] = {

        "source_id":
            source_id,

        "annotated_rgb":
            annotated_rgb,

        "result_table":
            result_table,

        "plate_crops":
            plate_crops,

        "detector_wall_ms":
            detector_wall_ms,

        "total_elapsed_ms":
            total_elapsed_ms,

        "analysis_id":
            st.session_state[
                "analysis_id"
            ],
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

analysis = (
    st.session_state[
        "analysis"
    ]
)


if (
    analysis
    and analysis[
        "source_id"
    ] == source_id
):

    result_table = (
        analysis[
            "result_table"
        ]
    )


    st.subheader(
        "Detection result"
    )


    st.image(
        analysis[
            "annotated_rgb"
        ],
        caption=(
            "Detected plates "
            "and OCR text"
        ),
        use_container_width=True,
    )


    summary = st.columns(4)


    summary[0].metric(
        "Detected plates",
        len(
            result_table
        ),
    )


    summary[1].metric(
        "Detector wall time",
        (
            f"{analysis['detector_wall_ms']:.1f} ms"
        ),
    )


    summary[2].metric(
        "End-to-end processing",
        (
            f"{analysis['total_elapsed_ms']:.1f} ms"
        ),
    )


    accepted_count = int(

        result_table.get(
            "ocr_status",
            pd.Series(
                dtype=str
            ),
        )
        .eq(
            "Accepted OCR"
        )
        .sum()

    )


    summary[3].metric(
        "Accepted OCR",
        accepted_count,
    )


    st.caption(
        "19.73 ms is the frozen Tesla T4 "
        "YOLO inference benchmark. "
        "Current application wall time may "
        "differ depending on local hardware. "
        "End-to-end time includes PaddleOCR."
    )


    # ========================================================
    # NO DETECTIONS
    # ========================================================

    if result_table.empty:

        st.warning(
            "No number plate detected "
            "at the selected confidence."
        )


    # ========================================================
    # DETECTIONS FOUND
    # ========================================================

    else:

        st.subheader(
            "Detected plate crops"
        )


        plate_crops = (
            analysis.get(
                "plate_crops",
                [],
            )
        )


        for row_start in range(
            0,
            len(
                plate_crops
            ),
            3,
        ):

            row_crops = (
                plate_crops[
                    row_start:
                    row_start + 3
                ]
            )


            columns = (
                st.columns(
                    len(
                        row_crops
                    )
                )
            )


            for offset, (
                column,
                crop_rgb,
            ) in enumerate(
                zip(
                    columns,
                    row_crops,
                )
            ):

                plate_number = (
                    row_start
                    + offset
                    + 1
                )


                plate_row = (
                    result_table.iloc[
                        plate_number
                        - 1
                    ]
                )


                with column:

                    st.image(
                        crop_rgb,
                        caption=(
                            f"Plate "
                            f"{plate_number}"
                        ),
                        use_container_width=True,
                    )


                    confidence_text = (

                        f"{plate_row['ocr_confidence']:.3f}"

                        if pd.notna(
                            plate_row[
                                "ocr_confidence"
                            ]
                        )

                        else "n/a"
                    )


                    st.caption(
                        f"OCR: "
                        f"{plate_row['ocr_candidate']} | "
                        f"confidence: "
                        f"{confidence_text} | "
                        f"{plate_row['ocr_status']}"
                    )


        # ====================================================
        # STRUCTURED RESULTS
        # ====================================================

        st.subheader(
            "Structured OCR results"
        )


        st.caption(
            "Review the extracted plate "
            "text before export. "
            "Correct the Reviewed plate "
            "field when necessary."
        )


        edited_table = (
            st.data_editor(
                result_table,
                key=(
                    f"result_editor_"
                    f"{analysis['analysis_id']}"
                ),
                hide_index=True,
                disabled=[
                    "plate",
                    "detection_confidence",
                    "ocr_candidate",
                    "ocr_confidence",
                    "ocr_status",
                ],
                column_config={
                    "detection_confidence":
                        st.column_config.NumberColumn(
                            format="%.3f"
                        ),

                    "ocr_confidence":
                        st.column_config.NumberColumn(
                            format="%.3f"
                        ),

                    "reviewed_plate":
                        st.column_config.TextColumn(
                            "Reviewed plate",
                            help=(
                                "Enter only visible "
                                "registration characters."
                            ),
                        ),
                },
                use_container_width=True,
            )
        )


        export_table = (
            build_export_table(
                edited_table
            )
        )


        unresolved = int(
            export_table[
                "final_plate"
            ].eq(
                ""
            ).sum()
        )


        if unresolved:

            st.warning(
                f"{unresolved} plate "
                f"reading(s) still "
                f"require review."
            )

        else:

            st.success(
                "All detected plate "
                "readings have a "
                "final reviewed value."
            )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.download_button(
            "Download detection log (CSV)",
            data=(
                export_table
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            ),
            file_name=(
                "number_plate_detection_log.csv"
            ),
            mime="text/csv",
        )