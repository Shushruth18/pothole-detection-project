"""Convert the chosen Kaggle VOC dataset into a deduplicated YOLO dataset."""
import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageOps
import yaml

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def convert_boxes(root, width, height):
    lines = []
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        if box is None:
            raise ValueError('Object has no bounding box')
        x1, y1, x2, y2 = [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        # Pascal VOC uses one-based inclusive endpoints.
        x1, y1 = max(0, min(width, x1 - 1)), max(0, min(height, y1 - 1))
        x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid or empty bounding box')
        lines.append(f'0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
    return lines


def prepare(source, output, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'{output} already exists. Choose a fresh --output directory.')
    xmls = sorted(source.rglob('*.xml'))
    if not xmls:
        raise ValueError('No XML annotations found. Select the extracted Kaggle dataset folder.')
    images = {}
    for path in source.rglob('*'):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images.setdefault(path.name.lower(), []).append(path)
    original_test = set()
    split_paths = sorted(source.rglob('splits.json'))
    if split_paths:
        split = json.loads(split_paths[0].read_text(encoding='utf-8'))
        original_test = {str(x).replace('\\', '/').split('/')[-1] for x in split.get('test', [])}
    records, seen = [], {}
    for xml in xmls:
        root = ET.parse(xml).getroot()
        filename = Path((root.findtext('filename') or '').replace('\\', '/')).name
        candidates = images.get(filename.lower(), [])
        if not candidates:
            candidates = [p for paths in images.values() for p in paths if p.stem == xml.stem]
        local = [p for p in candidates if p.parent == xml.parent]
        candidates = local or candidates
        if len(candidates) != 1:
            raise ValueError(f'{xml.name}: cannot uniquely locate its image ({len(candidates)} matches)')
        img_path = candidates[0]
        with Image.open(img_path) as image:
            # Do not apply EXIF rotation: boxes refer to stored pixel orientation.
            rgb = image.convert('RGB')
            width, height = rgb.size
            declared = root.find('size')
            if declared is not None:
                dw, dh = int(declared.findtext('width')), int(declared.findtext('height'))
                if (dw, dh) != (width, height):
                    raise ValueError(f'{xml.name}: annotation and image dimensions disagree')
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        labels = convert_boxes(root, width, height)
        record = {'xml': xml, 'image': img_path, 'hash': digest, 'labels': labels,
                  'original_test': xml.name in original_test}
        if digest in seen:
            prior = seen[digest]
            if sorted(prior['labels']) != sorted(labels):
                raise ValueError(f'Duplicate image has conflicting labels: {xml.name}')
            prior['original_test'] |= record['original_test']
            continue
        seen[digest] = record
        records.append(record)
    if len(records) < 10:
        raise ValueError('At least 10 unique annotated images are required')
    rng = random.Random(seed)
    if original_test:
        test = [r for r in records if r['original_test']]
        pool = [r for r in records if not r['original_test']]
        rng.shuffle(pool)
        nval = max(1, round(len(pool) * .20))
        groups = {'train': pool[nval:], 'val': pool[:nval], 'test': test}
    else:
        rng.shuffle(records)
        ntest, nval = max(1, round(len(records)*.15)), max(1, round(len(records)*.15))
        groups = {'test': records[:ntest], 'val': records[ntest:ntest+nval], 'train': records[ntest+nval:]}
    if any(not group for group in groups.values()):
        raise ValueError('One of the dataset splits is empty')
    manifest = []
    for split, group in groups.items():
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
        for rec in group:
            stem = rec['hash'][:24]
            with Image.open(rec['image']) as image:
                image.convert('RGB').save(output / 'images' / split / f'{stem}.jpg', quality=95)
            (output / 'labels' / split / f'{stem}.txt').write_text('\n'.join(rec['labels']) + '\n', encoding='utf-8')
            manifest.append({'split': split, 'source_image': str(rec['image']), 'source_xml': str(rec['xml']),
                             'pixel_sha256': rec['hash'], 'boxes': len(rec['labels'])})
    config = {'path': output.as_posix(), 'train': 'images/train', 'val': 'images/val',
              'test': 'images/test', 'names': {0: 'pothole'}}
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    summary = {'seed': seed, 'source_annotations': len(xmls), 'unique_images': len(records),
               'duplicates_removed': len(xmls)-len(records), 'original_test_preserved': bool(original_test),
               'splits': {k: {'images': len(v), 'boxes': sum(len(r['labels']) for r in v)} for k,v in groups.items()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return output / 'data.yaml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', default='dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)
