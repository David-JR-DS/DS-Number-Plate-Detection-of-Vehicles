"""Production OCR helpers for the Streamlit number-plate application.

Production pipeline:
    YOLO plate crop
    -> PaddleOCR recognition
    -> generic Indian number-plate cleanup

No benchmark-specific registration corrections are used.
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np


# ============================================================
# INDIAN REGISTRATION RULES
# ============================================================

INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA",
    "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"
)

PLATE_EXTRACT_PATTERN = re.compile(
    r"([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})"
)


# ============================================================
# NORMALISE OCR TEXT
# ============================================================

def normalise_plate_text(text: object) -> str:
    """Uppercase OCR text and keep only A-Z / 0-9."""

    if not isinstance(text, str):
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper(),
    )


# ============================================================
# GENERIC INDIAN PLATE CLEANUP
# ============================================================

def canonicalise_plate_candidate(text: object) -> str:
    """Clean OCR output using generic Indian registration rules."""

    candidate = normalise_plate_text(text)

    if not candidate:
        return ""

    # Generic country marker commonly printed on Indian plates.
    candidate = candidate.replace(
        "IND",
        "",
    )

    matches = PLATE_EXTRACT_PATTERN.findall(
        candidate
    )

    valid_matches = [
        match
        for match in matches
        if match[:2] in INDIAN_STATE_CODES
    ]

    if valid_matches:
        return max(
            valid_matches,
            key=len,
        )

    return candidate


# ============================================================
# FORMAT VALIDATION
# ============================================================

def is_valid_plate_text(text: str) -> bool:
    """Check whether OCR output resembles an Indian registration."""

    candidate = normalise_plate_text(text)

    if not candidate:
        return False

    if not PLATE_PATTERN.fullmatch(candidate):
        return False

    return (
        candidate[:2]
        in INDIAN_STATE_CODES
    )


# ============================================================
# OCR REVIEW STATUS
# ============================================================

def classify_ocr_text(
    text: str,
    confidence: float,
    threshold: float,
) -> str:
    """Classify an OCR reading for automatic acceptance or review."""

    candidate = normalise_plate_text(text)

    if not candidate:
        return "Unreadable"

    if not is_valid_plate_text(candidate):
        return "Review required"

    if float(confidence) < float(threshold):
        return "Review required"

    return "Accepted OCR"


# ============================================================
# PADDLEOCR RESULT PARSER
# ============================================================

def _collect_ocr_pairs(
    obj: Any,
    texts: list[str],
    scores: list[float],
) -> None:
    """Recursively collect recognised text and confidence values."""

    if obj is None:
        return

    # --------------------------------------------------------
    # PaddleOCR 3.x result object
    # --------------------------------------------------------

    if hasattr(obj, "json"):

        try:

            value = obj.json

            if callable(value):
                value = value()

            _collect_ocr_pairs(
                value,
                texts,
                scores,
            )

            return

        except Exception:
            pass


    # --------------------------------------------------------
    # Dictionary-style output
    # --------------------------------------------------------

    if isinstance(obj, dict):

        if (
            "res" in obj
            and isinstance(
                obj["res"],
                dict,
            )
        ):

            _collect_ocr_pairs(
                obj["res"],
                texts,
                scores,
            )

            return


        rec_texts = obj.get(
            "rec_texts"
        )

        rec_scores = obj.get(
            "rec_scores"
        )


        if rec_texts:

            for text in rec_texts:

                texts.append(
                    str(text)
                )


            if rec_scores:

                for score in rec_scores:

                    try:

                        scores.append(
                            float(score)
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

            return


        for value in obj.values():

            _collect_ocr_pairs(
                value,
                texts,
                scores,
            )

        return


    # --------------------------------------------------------
    # List / tuple output
    # --------------------------------------------------------

    if isinstance(
        obj,
        (list, tuple),
    ):

        # Paddle recognition-only result:
        #
        # ("KL01AB1234", 0.98)

        if (
            len(obj) == 2
            and isinstance(
                obj[0],
                str,
            )
            and isinstance(
                obj[1],
                (
                    int,
                    float,
                    np.number,
                ),
            )
        ):

            texts.append(
                obj[0]
            )

            scores.append(
                float(obj[1])
            )

            return


        # Standard Paddle detection result:
        #
        # [box, ("KL01AB1234", 0.98)]

        if (
            len(obj) >= 2
            and isinstance(
                obj[1],
                (list, tuple),
            )
            and len(obj[1]) >= 2
            and isinstance(
                obj[1][0],
                str,
            )
        ):

            texts.append(
                str(obj[1][0])
            )

            try:

                scores.append(
                    float(
                        obj[1][1]
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

            return


        # Recursively inspect nested output.

        for item in obj:

            _collect_ocr_pairs(
                item,
                texts,
                scores,
            )


# ============================================================
# CONVERT PADDLE RESULT TO TEXT
# ============================================================

def _extract_paddle_raw(
    result: Any,
) -> tuple[str, float]:
    """Convert PaddleOCR output into one text string and confidence."""

    texts: list[str] = []
    scores: list[float] = []

    _collect_ocr_pairs(
        result,
        texts,
        scores,
    )

    raw_text = normalise_plate_text(
        "".join(texts)
    )

    confidence = (
        float(
            np.mean(scores)
        )
        if scores
        else 0.0
    )

    return (
        raw_text,
        confidence,
    )


# ============================================================
# RUN PADDLE OCR
# ============================================================

def _run_paddle_ocr(
    reader: Any,
    image: np.ndarray,
) -> Any:
    """Run PaddleOCR 3.2.0 on the YOLO plate crop."""

    return reader.predict(
        input=image
    )

    # --------------------------------------------------------
    # PaddleOCR 3.x
    # --------------------------------------------------------

    if hasattr(
        reader,
        "predict",
    ):

        return reader.predict(
            input=image
        )


    # --------------------------------------------------------
    # PaddleOCR 2.x / Windows
    # --------------------------------------------------------

    return reader.ocr(
        image,
        det=False,
        rec=True,
        cls=False,
    )


# ============================================================
# FINAL PRODUCTION OCR FUNCTION
# ============================================================

def read_plate(
    reader: Any,
    crop_bgr: np.ndarray,
) -> tuple[
    str,
    float,
    np.ndarray,
]:
    """Read one YOLO-detected number-plate crop.

    Returns
    -------
    cleaned_text:
        Generic cleaned registration candidate.

    confidence:
        Mean PaddleOCR recognition confidence.

    display_crop:
        Original plate crop converted from BGR to RGB.
    """

    # --------------------------------------------------------
    # EMPTY CROP CHECK
    # --------------------------------------------------------

    if (
        crop_bgr is None
        or crop_bgr.size == 0
    ):

        return (
            "",
            0.0,
            np.empty(
                (0, 0, 3),
                dtype=np.uint8,
            ),
        )


    # --------------------------------------------------------
    # ORIGINAL CROP
    # --------------------------------------------------------
    #
    # Notebook 03 showed that the original YOLO crop produced
    # the strongest OCR result.
    # --------------------------------------------------------

    if crop_bgr.ndim == 2:

        paddle_input = cv2.cvtColor(
            crop_bgr,
            cv2.COLOR_GRAY2BGR,
        )

    else:

        paddle_input = crop_bgr


    # Image used only for Streamlit display.

    display_crop = cv2.cvtColor(
        paddle_input,
        cv2.COLOR_BGR2RGB,
    )


    # --------------------------------------------------------
    # RUN OCR
    # --------------------------------------------------------

    result = _run_paddle_ocr(
        reader,
        paddle_input,
    )


    # --------------------------------------------------------
    # PARSE RAW OCR OUTPUT
    # --------------------------------------------------------

    raw_text, confidence = (
        _extract_paddle_raw(
            result
        )
    )


    # --------------------------------------------------------
    # GENERIC INDIAN PLATE CLEANUP
    # --------------------------------------------------------

    cleaned_text = (
        canonicalise_plate_candidate(
            raw_text
        )
    )


    return (
        cleaned_text,
        confidence,
        display_crop,
    )