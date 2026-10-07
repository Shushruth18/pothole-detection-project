# RoadWatch — Pothole Detection Project

A bachelor-level AI/ML project that detects potholes in road images and recorded video using a fine-tuned YOLOv8n model. A Streamlit interface provides annotated media, CSV detection logs, and optional maps using supplied camera GPS.

## Features

- Upload JPG/PNG photos or short road videos.
- Adjust detection confidence and video frame sampling.
- Download annotated images, AVI videos, and CSV observations.
- Plot camera locations when real coordinates or synchronized GPS data are supplied.
- Convert the selected Kaggle XML annotations into YOLO labels.
- Fine-tune in Google Colab and evaluate on held-out test data.

## Measured results

The included `models/best.pt` was trained in Google Colab. The recorded held-out test results in `models/metrics.json` are:

| Metric | Value |
|---|---:|
| Precision | 0.6285 |
| Recall | 0.6990 |
| mAP@50 | 0.7207 |
| mAP@50–95 | 0.4350 |

These results describe this dataset split; they do not establish accuracy on all local roads. Five automated pipeline tests passed during repository preparation. Image/video detection and GPS exports should also be inspected manually.

## Run on Windows

Install Python 3.11 or 3.12. From the project folder:

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 and keep the terminal open. Alternatively, double-click `setup_windows.bat` and then `run_app.bat`. Leave Streamlit's optional email prompt blank and press Enter.

The trained checkpoint is included at `models/best.pt`, so retraining is not required for a first demo. Only load model checkpoints from sources you trust.

## Run on macOS/Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Train with Google Colab

1. Open https://colab.research.google.com/ and upload `Train_in_Colab.ipynb`.
2. Select a GPU runtime and execute the cells in order.
3. Upload `prepare_data.py` and `train.py` when prompted.
4. Download the selected Kaggle dataset automatically or use the ZIP-upload fallback.
5. Complete training and held-out evaluation, then download the trained-artifacts ZIP.
6. Extract its contents into this folder to replace the model and evaluation artifacts.

The notebook requests up to 30 epochs, 512-pixel images, batch size 8, and a 0.75-hour training budget. GPU availability and elapsed time vary; setup and evaluation take additional time. Reduce batch size if GPU memory is insufficient.

## Dataset

[Annotated Potholes Image Dataset](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset), published by **Atikur Rahman Chitholian**. Dataset images are not redistributed in this repository. Consult the original dataset's database/content licences before redistribution or other use.

`prepare_data.py` validates boxes and image dimensions, converts VOC one-based inclusive coordinates, and removes exact decoded-pixel duplicates. It maps all objects to the single pothole class. When `splits.json` exists, its test membership is retained and 20% of the remaining pool becomes validation, using seed 42. Otherwise it creates a 70/15/15 split. Near-duplicate scenes still need manual review. For local videos, split by road or recording rather than adjacent frames.

Local training:

```cmd
.venv\Scripts\python.exe download_data.py
.venv\Scripts\python.exe prepare_data.py --source "PATH_PRINTED_BY_DOWNLOAD" --output dataset
.venv\Scripts\python.exe train.py --data dataset/data.yaml --epochs 30 --imgsz 512 --batch 8
```

Validation selects the checkpoint. Held-out test evaluation writes `models/metrics.json`; do not tune settings against test results.

## Command-line detection

```cmd
.venv\Scripts\python.exe predict.py --source "road.jpg" --output results/image
.venv\Scripts\python.exe predict.py --source "road.mp4" --stride 3 --output results/video
.venv\Scripts\python.exe predict.py --source "road.mp4" --gps "gps.csv" --stride 1 --output results/mapped
```

Video exports use AVI/MJPEG for a desktop video player. Skipped frames are explicitly marked. Use short clips in the web app because export bytes are held in memory.

## GPS mapping

Image mode accepts actual camera coordinates. Video mode accepts a synchronized CSV:

```csv
time_s,latitude,longitude
```

Add real observations to `gps_template.csv`. Time must be seconds relative to video start. The nearest GPS observation is used only within two seconds. No GPS means no map. Maps show supplied **camera locations**, not surveyed pothole positions, and map tiles require internet.

## Tests

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover coordinate conversion, invalid boxes, split membership, GPS tolerance, and CSV headers. They do not establish detector accuracy or verify the full browser interface.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `detection.py` | Inference exports, video processing, GPS maps |
| `predict.py` | Command-line inference |
| `prepare_data.py` | XML conversion and dataset preparation |
| `train.py` | Training and held-out evaluation |
| `Train_in_Colab.ipynb` | GPU training notebook |
| `models/best.pt` | Trained pothole checkpoint |
| `models/metrics.json` | Recorded test results |
| `tests/test_pipeline.py` | Pipeline checks |
| `PROJECT_REPORT.md` | Student report draft and viva notes |

## Limitations

Shadows, rain, repairs, camera viewpoint, and local-road differences may cause misses or false alarms. Repeated video detections are observations, not a count of unique potholes. Box image area does not measure physical size, depth, or validated severity. This is a student prototype, not a validated municipal deployment.

## References

- [Ultralytics training](https://docs.ultralytics.com/modes/train/)
- [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics licensing](https://www.ultralytics.com/license)
- [Streamlit](https://docs.streamlit.io/)
- [Folium](https://python-visualization.github.io/folium/latest/)

Dependency and dataset licences apply to their respective materials. No additional source-code licence is declared here.
