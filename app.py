import io
import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from detection import csv_bytes, load_gps, load_model, map_html, process_video, rows_for

st.set_page_config(page_title='RoadWatch | Pothole Detection', page_icon='🛣️', layout='wide')
st.title('🛣️ RoadWatch')
st.write('Detect potholes in road photos and recorded videos. Export annotated evidence and a detection log.')
with st.sidebar:
    st.header('Detection settings')
    checkpoint = st.text_input('Trained model path', 'models/best.pt')
    confidence = st.slider('Confidence threshold', .05, .95, .35, .05)
    stride = st.select_slider('Analyse every Nth video frame', options=[1, 2, 3, 5, 10], value=3)
    st.caption('Lower thresholds may detect more potholes but produce more false alarms.')
    st.caption('Only load model files from a source you trust.')
    metrics_path = Path('models/metrics.json')
    if metrics_path.exists():
        st.subheader('Held-out test results')
        metrics = json.loads(metrics_path.read_text())
        for label, key in [('Precision', 'test_precision'), ('Recall', 'test_recall'), ('mAP@50', 'test_mAP50')]:
            st.metric(label, f"{metrics[key]:.3f}")
    else:
        st.info('Measured results appear after training and test evaluation.')

if not Path(checkpoint).is_file():
    st.warning('The trained pothole model is missing. Run the Colab notebook or train.py, then place best.pt in models/.')
    st.code('python train.py --data dataset/data.yaml --epochs 30')
    st.stop()

@st.cache_resource
def cached_model(path, modified):
    return load_model(path)

try:
    model = cached_model(checkpoint, Path(checkpoint).stat().st_mtime_ns)
except Exception as exc:
    st.error(str(exc))
    st.stop()

mode = st.radio('Input type', ['Road image', 'Road video'], horizontal=True)
upload = st.file_uploader('Upload a road image or video', type=['jpg', 'jpeg', 'png'] if mode == 'Road image' else ['mp4', 'avi', 'mov'])
gps_upload = None
location = None
if mode == 'Road image':
    with st.expander('Optional camera location'):
        use_location = st.checkbox('I have the actual camera coordinates for this image')
        lat = st.number_input('Latitude', min_value=-90., max_value=90., value=0., format='%.6f')
        lon = st.number_input('Longitude', min_value=-180., max_value=180., value=0., format='%.6f')
        if use_location:
            location = (lat, lon, 'user_supplied_camera_location')
else:
    gps_upload = st.file_uploader('Optional synchronized GPS CSV: time_s,latitude,longitude', type=['csv'])
    st.caption('GPS timestamps must use seconds from the start of this video. Points farther than 2 seconds from a frame are omitted.')

if upload and st.button('Detect potholes', type='primary'):
    try:
        if mode == 'Road image':
            image = ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
            result = model.predict(image, conf=confidence, verbose=False)[0]
            rows = rows_for(result, location=location)
            annotated = Image.fromarray(result.plot()[..., ::-1])
            buffer = io.BytesIO()
            annotated.save(buffer, format='PNG')
            st.session_state['result'] = {'kind': 'image', 'media': buffer.getvalue(), 'rows': rows}
        else:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                source = path / ('input' + Path(upload.name).suffix.lower())
                source.write_bytes(upload.getvalue())
                gps = []
                if gps_upload:
                    gps_file = path / 'gps.csv'
                    gps_file.write_bytes(gps_upload.getvalue())
                    gps = load_gps(gps_file)
                progress = st.progress(0.)
                with st.spinner('Analysing video frames…'):
                    rows = process_video(model, source, path/'annotated.avi', confidence, stride, gps, progress.progress)
                st.session_state['result'] = {'kind': 'video', 'media': (path/'annotated.avi').read_bytes(), 'rows': rows}
    except Exception as exc:
        st.session_state.pop('result', None)
        st.error(f'Could not process the input: {exc}')

if 'result' in st.session_state:
    output = st.session_state['result']
    rows = output['rows']
    st.subheader('Last completed detection run')
    st.metric('Detection observations', len(rows))
    st.caption('Video observations may show the same pothole repeatedly; this is not a count of unique potholes.')
    if output['kind'] == 'image':
        st.image(output['media'], caption='Predicted potholes', use_container_width=True)
        st.download_button('Download annotated image', output['media'], 'annotated.png', 'image/png')
    else:
        st.download_button('Download annotated video', output['media'], 'annotated.avi', 'video/x-msvideo')
        st.caption('Open the AVI in a video player. Frames skipped by the analysis setting are explicitly marked.')
    if not rows:
        st.info('No potholes detected at this threshold. This does not establish that the road is pothole-free.')
    st.dataframe(rows, use_container_width=True)
    st.download_button('Download detection log', csv_bytes(rows), 'detections.csv', 'text/csv')
    html = map_html(rows)
    if html:
        st.subheader('Camera locations of detections')
        st.caption('Markers show supplied camera coordinates, not precisely surveyed pothole positions. Map tiles require internet.')
        components.html(html, height=450)
        st.download_button('Download map', html, 'detection-map.html', 'text/html')
    st.caption('Box area is reported as a fraction of image area. It is not a depth measurement or a validated severity score.')
