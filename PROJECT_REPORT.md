# RoadWatch: Pothole Detection from Road Images and Videos

Domain: Computer vision | Task: Pothole object detection | Model: YOLOv8n

## Abstract

RoadWatch is a computer vision prototype for detecting visible potholes in road photographs and recorded video. It fine-tunes YOLOv8n using Atikur Rahman Chitholian's Annotated Potholes Image Dataset from Kaggle. XML annotations are converted into YOLO labels and data is divided into training, validation, and held-out test sets. A Streamlit interface exports annotated media and CSV records. Actual supplied camera coordinates support optional mapping. Performance is measured using precision, recall, mAP@50, and mAP@50–95. Results must be added after execution; no accuracy claim is made yet.

## Objectives and approach

Prepare and validate annotated road images, fine-tune a compact detector, evaluate on held-out images, process image/video inputs, and export reviewable detection evidence.

Pipeline: Kaggle images/XML → coordinate conversion and exact duplicate removal → dataset splits → YOLOv8n fine-tuning → validation checkpoint selection → held-out test evaluation → Streamlit detection and optional GPS mapping.

Original test membership is preserved when splits.json is available; otherwise a deterministic 70/15/15 split is generated. Near-duplicate scenes require manual review. The emergency configuration requests up to 30 epochs, 512-pixel inputs, batch 8, and a 0.75-hour training budget. Record actual epochs, hardware, time, and package versions from the generated artifacts.

## Results — complete using generated artifacts

| Item | Actual value |
|---|---|
| Unique images and removed duplicates | [dataset_summary.json] |
| Training / validation / test counts | [dataset_summary.json] |
| Precision | [models/metrics.json] |
| Recall | [models/metrics.json] |
| mAP@50 | [models/metrics.json] |
| mAP@50–95 | [models/metrics.json] |
| Automated test results | [Run and record] |

Precision measures how many predicted positives are correct; recall measures how many labelled potholes are detected. mAP summarizes precision–recall performance at specified box-overlap thresholds. Add actual training curves, a successful detection, a missed pothole, and a false alarm if observed. Do not invent scores or tune using test results.

## Limitations and future work

Lighting, shadows, water, viewpoint, and local-road differences may cause errors. Image-dataset performance does not establish dashcam performance. Apparent box area does not measure depth or physical severity. Video observations can repeat the same pothole. GPS points indicate supplied camera location, not surveyed pothole position. A short emergency run may be insufficient.

Future work: representative local data split by road/recording, night/rain evaluation, object tracking, and calibrated geometry or depth sensing.

## Conclusion

[After evaluation: state the measured outcome, describe observed failures, and say whether results support a classroom demonstration. Do not claim production readiness.]

## Technical notes

- Detection provides locations of multiple potholes; image classification alone does not.
- Transfer learning adapts pretrained visual features to a smaller pothole dataset.
- Validation selects the model; held-out test data estimates performance independently.
- Confidence is a model score, not a guarantee.
- Camera distance changes box size, so box area cannot establish severity.
- The map uses actual supplied camera GPS; the model cannot infer coordinates from a road image.

## References

Dataset: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset

Ultralytics: https://docs.ultralytics.com/modes/train/ and https://docs.ultralytics.com/modes/predict/

Streamlit: https://docs.streamlit.io/ | Folium: https://python-visualization.github.io/folium/latest/
