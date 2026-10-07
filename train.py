"""Fine-tune YOLOv8n, then evaluate the best validation checkpoint on held-out test data."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/data.yaml')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default=None, help='0 for CUDA GPU, cpu for CPU')
    parser.add_argument('--hours', type=float, default=None, help='Optional training time budget, excluding setup/evaluation')
    args = parser.parse_args()
    from ultralytics import YOLO
    config = dict(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz,
                  batch=args.batch, workers=0, seed=42, deterministic=True,
                  project='runs', name='pothole', exist_ok=False, patience=10,
                  close_mosaic=0, plots=True)
    if args.device is not None:
        config['device'] = args.device
    if args.hours is not None:
        config['time'] = args.hours
    model = YOLO('yolov8n.pt')
    model.train(**config)
    run = Path(model.trainer.save_dir)
    best = run / 'weights' / 'best.pt'
    if not best.exists():
        raise RuntimeError('Training did not produce best.pt. Check the training log.')
    Path('models').mkdir(exist_ok=True)
    shutil.copy2(best, 'models/best.pt')
    trained = YOLO('models/best.pt')
    eval_args = dict(data=config['data'], split='test', imgsz=args.imgsz, batch=args.batch,
                     workers=0, project='runs', name='test', plots=True)
    if args.device is not None:
        eval_args['device'] = args.device
    metrics = trained.val(**eval_args)
    report = {'dataset': config['data'], 'checkpoint': 'models/best.pt', 'training_run': str(run),
              'test_precision': float(metrics.box.mp), 'test_recall': float(metrics.box.mr),
              'test_mAP50': float(metrics.box.map50), 'test_mAP50_95': float(metrics.box.map),
              'evaluation_split': 'test', 'seed': 42}
    Path('models/metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
