import argparse
from pathlib import Path
from PIL import Image, ImageOps
from detection import load_model, load_gps, rows_for, process_video, csv_bytes, map_html


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--model', default='models/best.pt')
    p.add_argument('--output', default='results')
    p.add_argument('--conf', type=float, default=.35)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--gps', help='Video GPS CSV: time_s,latitude,longitude')
    a = p.parse_args()
    if not 0 < a.conf < 1 or a.stride < 1:
        p.error('Confidence must be between 0 and 1, and stride must be positive')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.model)
    if Path(a.source).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
        image = ImageOps.exif_transpose(Image.open(a.source)).convert('RGB')
        result = model.predict(image, conf=a.conf, verbose=False)[0]
        Image.fromarray(result.plot()[..., ::-1]).save(out/'annotated.png')
        rows = rows_for(result)
    else:
        rows = process_video(model, a.source, out/'annotated.avi', a.conf, a.stride, load_gps(a.gps) if a.gps else [])
    (out/'detections.csv').write_bytes(csv_bytes(rows))
    html = map_html(rows)
    if html:
        (out/'map.html').write_text(html, encoding='utf-8')
    print(f'{len(rows)} detection observations exported to {out.resolve()}')


if __name__ == '__main__':
    main()
