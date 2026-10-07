"""Detection exports. Coordinates describe camera location, never inferred pothole position."""
import csv
import io
import math
from pathlib import Path

import cv2
import folium

FIELDS = ['frame', 'time_s', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax',
          'image_area_fraction', 'camera_latitude', 'camera_longitude', 'location_source']


def load_model(path):
    from ultralytics import YOLO
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Train first, or put a trusted trained pothole checkpoint at models/best.pt.')
    model = YOLO(str(path))
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if len(names) != 1 or 'pothole' not in str(names[0]).lower():
        raise ValueError('Use the single-class pothole model trained by this project, not generic YOLO weights.')
    return model


def load_gps(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            item = tuple(float(row[k]) for k in ('time_s', 'latitude', 'longitude'))
            if not all(math.isfinite(v) for v in item) or item[0] < 0 or not -90 <= item[1] <= 90 or not -180 <= item[2] <= 180:
                raise ValueError('GPS rows require nonnegative time and valid latitude/longitude.')
            rows.append(item)
    return sorted(rows)


def gps_at(rows, time_s, tolerance=2.0):
    if not rows:
        return None
    row = min(rows, key=lambda r: abs(r[0]-time_s))
    return (row[1], row[2], 'timestamp_matched_camera_gps') if abs(row[0]-time_s) <= tolerance else None


def rows_for(result, frame=0, time_s=0., location=None):
    height, width = result.orig_shape
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        rows.append(dict(zip(FIELDS, [frame, round(time_s, 3), round(float(box.conf[0]), 5),
                       round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                       round((x2-x1)*(y2-y1)/(width*height), 6),
                       location[0] if location else '', location[1] if location else '',
                       location[2] if location else 'unavailable'])))
    return rows


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def map_html(rows):
    located = [r for r in rows if r['camera_latitude'] != '']
    if not located:
        return None
    m = folium.Map(location=[located[0]['camera_latitude'], located[0]['camera_longitude']], zoom_start=16)
    # Multiple frame observations are not unique potholes. Plot one marker per camera location.
    grouped = {}
    for r in located:
        key = (r['camera_latitude'], r['camera_longitude'])
        grouped.setdefault(key, []).append(r)
    for (lat, lon), observations in grouped.items():
        score = max(r['confidence'] for r in observations)
        folium.Marker([lat, lon], popup=f'Camera location; {len(observations)} detection observations; max confidence {score:.2f}').add_to(m)
    return m.get_root().render()


def process_video(model, source, output, confidence=.35, stride=3, gps=None, progress=None):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot decode this video. Try MP4/H.264.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no valid frame rate; timestamp alignment is unavailable.')
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # AVI/MJPEG is dependable for export; browsers may require downloading it.
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError('Cannot create annotated video')
    rows, frame = [], 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame % stride == 0:
                result = model.predict(image, conf=confidence, verbose=False)[0]
                rows.extend(rows_for(result, frame, frame/fps, gps_at(gps or [], frame/fps)))
                image = result.plot()
            else:
                cv2.putText(image, 'Frame not analysed', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            writer.write(image)
            frame += 1
            if progress and total > 0:
                progress(min(frame/total, 1.))
    finally:
        cap.release()
        writer.release()
    if frame == 0:
        raise ValueError('Video contains no decodable frames')
    return rows
