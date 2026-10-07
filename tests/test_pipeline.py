import csv
import io
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from prepare_data import convert_boxes, prepare
from detection import csv_bytes, gps_at, load_gps


class PipelineTests(unittest.TestCase):
    def test_voc_full_image_box(self):
        xml = ET.fromstring('<annotation><object><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object></annotation>')
        numbers = list(map(float, convert_boxes(xml, 100, 50)[0].split()))
        self.assertEqual(numbers, [0, .5, .5, 1., 1.])

    def test_invalid_box_rejected(self):
        xml = ET.fromstring('<annotation><object><bndbox><xmin>90</xmin><ymin>1</ymin><xmax>10</xmax><ymax>50</ymax></bndbox></object></annotation>')
        with self.assertRaises(ValueError):
            convert_boxes(xml, 100, 50)

    def test_splits_have_no_duplicate_pixels_and_preserve_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            for i in range(12):
                Image.new('RGB', (100, 50), (i*15, 30, 50)).save(source/f'img-{i}.png')
                (source/f'img-{i}.xml').write_text(f'<annotation><filename>img-{i}.png</filename><size><width>100</width><height>50</height></size><object><name>pothole</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>50</xmax><ymax>25</ymax></bndbox></object></annotation>')
            (source/'splits.json').write_text(json.dumps({'test': ['img-0.xml', 'img-1.xml']}))
            prepare(source, root/'output')
            manifest = json.loads((root/'output/manifest.json').read_text())
            self.assertEqual(len({r['pixel_sha256'] for r in manifest}), len(manifest))
            self.assertEqual({Path(r['source_xml']).name for r in manifest if r['split']=='test'}, {'img-0.xml', 'img-1.xml'})
            self.assertEqual({r['split'] for r in manifest}, {'train', 'val', 'test'})

    def test_gps_matching_does_not_fabricate_distant_location(self):
        self.assertEqual(gps_at([(0., 12., 77.)], 1.), (12., 77., 'timestamp_matched_camera_gps'))
        self.assertIsNone(gps_at([(0., 12., 77.)], 10.))

    def test_empty_csv_contains_header(self):
        rows = list(csv.reader(io.StringIO(csv_bytes([]).decode())))
        self.assertEqual(len(rows), 1)
        self.assertIn('confidence', rows[0])


if __name__ == '__main__':
    unittest.main()
