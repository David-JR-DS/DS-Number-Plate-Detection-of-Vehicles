
---

## `PROJECT_CHECKLIST.md`

```markdown
# Number Plate Detection — Final Submission Checklist

This checklist tracks completed project work and remaining submission actions.

---

## 1. Dataset and Preprocessing

- [x] Validate image/label pairs.
- [x] Validate YOLO bounding-box annotations.
- [x] Confirm no invalid annotations.
- [x] Freeze train/validation/test split.
- [x] Document split counts.
- [x] Training: 90 images / 185 boxes.
- [x] Validation: 13 images / 23 boxes.
- [x] Test: 12 images / 25 boxes.
- [x] Review low-sharpness images.
- [x] Retain difficult valid images.
- [x] Provide portable dataset configuration.
- [x] Document dataset layout.
- [x] Confirm dataset ownership/licence wording before public release.

---

## 2. EDA

- [x] Dataset image-count analysis.
- [x] Bounding-box count analysis.
- [x] Split distribution.
- [x] Annotation validation.
- [x] Image-quality analysis.
- [x] Flag low-sharpness images.
- [x] Review flagged images.
- [x] Retain valid difficult samples.

---

## 3. YOLO Model Training

- [x] Train YOLO detector.
- [x] Compare available detector experiments.
- [x] Select final `models/best.pt`.
- [x] Preserve final training notebook.
- [x] Document final model-selection rationale.
- [x] Avoid unnecessary additional architecture experiments.

---

## 4. Detector Evaluation

- [x] Report validation precision.
- [x] Report validation recall.
- [x] Report validation mAP@50.
- [x] Report validation mAP@50-95.
- [x] Report untouched test precision.
- [x] Report untouched test recall.
- [x] Report untouched test mAP@50.
- [x] Report untouched test mAP@50-95.

### Final Validation

- [x] Precision: 99.11%.
- [x] Recall: 86.96%.
- [x] mAP@50: 96.35%.
- [x] mAP@50-95: 52.16%.

### Final Test

- [x] Precision: 93.05%.
- [x] Recall: 60.00%.
- [x] mAP@50: 68.62%.
- [x] mAP@50-95: 38.32%.

---

## 5. Detection Operating Point

- [x] Perform confidence-threshold sweep.
- [x] Perform threshold sweep on validation only.
- [x] Select threshold using validation F1.
- [x] Freeze selected threshold before test evaluation.
- [x] Apply frozen threshold once to test.
- [x] Report TP.
- [x] Report FP.
- [x] Report FN.
- [x] Report precision.
- [x] Report recall.
- [x] Report F1.
- [x] Report mean matched IoU.
- [x] Use IoU ≥ 0.50 matching.

### Selected Threshold

- [x] Confidence threshold: 0.50.

### Frozen Test Operating Point

- [x] TP: 14.
- [x] FP: 1.
- [x] FN: 11.
- [x] Precision: 93.33%.
- [x] Recall: 56.00%.
- [x] F1: 70.00%.
- [x] Mean matched IoU: 0.786.

---

## 6. Detector Failure Analysis

- [x] Generate FP/FN details.
- [x] Generate FP/FN visualization.
- [x] Identify small/distant plates as major failure mode.
- [x] Document localization-error cases.
- [x] Explain lower test recall.
- [x] Avoid removing difficult valid test images.

---

## 7. Detector Inference Time

- [x] Verify GPU runtime.
- [x] Use NVIDIA Tesla T4.
- [x] Warm up detector before timing.
- [x] Measure YOLO inference separately from OCR.
- [x] Mean inference: 19.73 ms/image.
- [x] Median inference: 19.71 ms/image.
- [x] Confirm project target <50 ms/image.
- [x] Mark inference target as achieved.

---

## 8. OCR Evaluation

- [x] Freeze 19-readable-plate OCR benchmark.
- [x] Preserve verified ground-truth text.
- [x] Compare PaddleOCR.
- [x] Compare EasyOCR.
- [x] Compare Tesseract.
- [x] Evaluate original crop.
- [x] Evaluate grayscale.
- [x] Evaluate CLAHE.
- [x] Evaluate Otsu.
- [x] Evaluate bilateral filtering.
- [x] Evaluate Canny edges.
- [x] Compare raw vs postprocessed OCR.
- [x] Use generic Indian registration cleanup.
- [x] Avoid benchmark-specific corrections.
- [x] Report strict exact-match accuracy.
- [x] Report character accuracy.
- [x] Report OCR runtime.
- [x] Document remaining OCR failures.

### Final OCR

- [x] Selected engine: PaddleOCR.
- [x] Selected preprocessing: original YOLO crop.
- [x] Exact match: 84.21%.
- [x] Character accuracy: 93.63%.
- [x] Character-level >90% target achieved.
- [x] Strict exact-match >90% target not achieved.
- [x] Report both metrics transparently.

---

## 9. Production OCR Pipeline

- [x] Replace old production EasyOCR pipeline.
- [x] Replace old Tesseract production dependency.
- [x] Integrate PaddleOCR.
- [x] Use original YOLO crop.
- [x] Use generic Indian plate cleanup.
- [x] Confirm no hard-coded benchmark answers.
- [x] Verify PaddlePaddle 3.2.0.
- [x] Verify PaddleOCR 3.2.0.
- [x] Verify modern `predict()` API.
- [x] Verify OCR works locally on Windows.

---

## 10. Streamlit Application

- [x] Load selected YOLO checkpoint.
- [x] Default detector confidence to 0.50.
- [x] Support image upload.
- [x] Support camera capture.
- [x] Display input image.
- [x] Display annotated detections.
- [x] Display plate bounding boxes.
- [x] Display enlarged plate crops.
- [x] Run PaddleOCR on detected crops.
- [x] Display extracted plate text.
- [x] Display detection confidence.
- [x] Display OCR confidence.
- [x] Display OCR review status.
- [x] Provide manual correction field.
- [x] Provide structured OCR-results table.
- [x] Export reviewed CSV log.
- [x] Display detector metrics.
- [x] Display OCR metrics.
- [x] Display benchmark inference time.
- [x] Display application wall-clock time.
- [x] Test upload workflow.
- [x] Test camera capture workflow.
- [x] Test CSV download.
- [x] Successfully test multiple detected plates in one image.

### Webcam Note

- [x] Camera capture supported through Streamlit.
- [x] Continuous live WebRTC video inference not implemented.
- [x] Document continuous webcam streaming as a future enhancement.

---

## 11. Application Validation

- [x] Test representative multi-plate image.
- [x] Confirm YOLO detection output.
- [x] Confirm PaddleOCR output.
- [x] Confirm generic cleanup.
- [x] Confirm high-confidence OCR acceptance.
- [x] Confirm manual-review workflow.
- [x] Confirm CSV export.
- [x] Test one known detector failure example for demo.
- [x] Test one OCR failure/review example for demo.

---

## 12. Dependency Files

- [x] Use Python 3.12 locally.
- [x] Set `paddlepaddle==3.2.0`.
- [x] Set `paddleocr==3.2.0`.
- [x] Include Streamlit.
- [x] Include Ultralytics.
- [x] Include OpenCV dependency through project stack.
- [x] Remove EasyOCR from production dependencies.
- [x] Remove Tesseract from production dependencies.
- [x] Remove Tesseract system package if no longer required.
- [x] Verify clean installation from final `requirements.txt`.

---

## 13. Repository Structure

- [x] `app.py`.
- [x] `models/best.pt`.
- [x] `src/ocr_pipeline.py`.
- [x] `configs/data.yaml`.
- [x] `configs/final_eval_data.yaml`.
- [x] `notebooks/01_eda.ipynb`.
- [x] `notebooks/02_train_on_colab.ipynb`.
- [x] `notebooks/03_ocr_evaluation.ipynb`.
- [x] `notebooks/04_final_evaluation.ipynb`.
- [x] `reports/`.
- [x] `README.md`.
- [x] `PROJECT_CHECKLIST.md`.
- [x] `.streamlit/config.toml`.
- [x] `requirements.txt`.
- [x] Remove obsolete/duplicate notebooks.
- [x] Remove unnecessary experiment artifacts from final root.
- [x] Verify no virtual environment is committed.
- [x] Verify no cache directories are committed.

---

## 14. Documentation

- [x] README overview.
- [x] Project pipeline.
- [x] Dataset counts.
- [x] Detector metrics.
- [x] Threshold-selection methodology.
- [x] FP/FN results.
- [x] IoU result.
- [x] OCR benchmark.
- [x] OCR limitations.
- [x] Detector limitations.
- [x] Inference speed.
- [x] Streamlit features.
- [x] Camera-capture limitation.
- [x] Local setup instructions.
- [x] Repository structure.
- [x] Business use cases.
- [x] Final performance report consistency check.
- [x] Final deployment-guide consistency check.

---

## 15. Deployment

- [x] Streamlit application runs locally.
- [x] Image-upload workflow works.
- [x] Camera-capture workflow works.
- [x] PaddleOCR runs locally.
- [x] Selected YOLO model runs locally.
- [x] Test deployment from clean environment.
- [x] Test Streamlit cloud deployment if required.
- [x] Confirm model file availability in deployment environment.

---

## 16. Demo Video

- [ ] Record final 3–5 minute demo.
- [ ] Introduce project objective.
- [ ] Show Streamlit app.
- [ ] Show model-performance dashboard.
- [ ] Upload representative vehicle image.
- [ ] Run YOLO detection.
- [ ] Show bounding boxes.
- [ ] Show detected plate crops.
- [ ] Show PaddleOCR output.
- [ ] Show structured result table.
- [ ] Demonstrate manual correction.
- [ ] Demonstrate CSV download.
- [ ] Show camera capture.
- [ ] Mention detector limitations.
- [ ] Mention OCR limitations.
- [ ] Mention continuous live webcam streaming as future work.
- [ ] Add demo URL to README/submission form.

---

## 17. GitHub Submission

- [x] Create or finalize GitHub repository.
- [x] Push cleaned source code.
- [x] Verify `models/best.pt` upload strategy.
- [x] Verify README renders correctly.
- [x] Remove `.venv`.
- [x] Remove `__pycache__`.
- [x] Remove temporary files.
- [x] Remove duplicate notebooks.
- [x] Verify `.gitignore`.
- [x] Test fresh clone.
- [x] Install dependencies from scratch.
- [x] Run Streamlit from fresh clone.
- [x] Add demo-video link.
- [x] Submit final repository URL.

---

## 18. Final Submission Status

### Completed

- [x] Dataset audit.
- [x] EDA.
- [x] YOLO training.
- [x] Model selection.
- [x] Detector evaluation.
- [x] Threshold selection.
- [x] Test operating-point evaluation.
- [x] FP/FN analysis.
- [x] IoU analysis.
- [x] GPU inference timing.
- [x] OCR engine comparison.
- [x] OCR preprocessing comparison.
- [x] Frozen OCR benchmark.
- [x] PaddleOCR production integration.
- [x] Streamlit upload workflow.
- [x] Streamlit camera capture.
- [x] Structured OCR results.
- [x] Manual review.
- [x] CSV export.

### Remaining

- [x] Final repository cleanup.
- [ x Final documentation consistency check.
- [x] Fresh-install verification.
- [ ] Demo video.
- [x] GitHub push.
- [x] Final submission.
