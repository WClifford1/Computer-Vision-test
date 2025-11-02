# Computer Vision with YOLO

Quick way to get a computer vision model running.

## Usage

```bash
source .venv/bin/activate
python yolo_demo.py --show --conf 0.4 --model yolov8n-seg.pt
```

## Options

- `--model` - Choose model (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8n-seg.pt` for segmentation)
- `--show` - Display webcam feed with detections
- `--conf` - Confidence threshold (e.g., `0.4`)

## Object Classes

If an object doesn't show up, it's probably not in the COCO training dataset. To see all detectable classes:

```bash
python show_classes.py
```
