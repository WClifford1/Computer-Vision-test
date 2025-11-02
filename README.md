# Computer Vision with YOLO

Quick way to get a computer vision model running on images, videos, or webcam.

## Usage

### Webcam (default)
```bash
source .venv/bin/activate
python yolo_demo.py --show --conf 0.4 --model yolov8n-seg.pt
```

### Images
```bash
python yolo_demo.py images/your_image.jpg --show --conf 0.4
```

### Videos
```bash
python yolo_demo.py path/to/video.mp4 --show
```

## Options

- `source` - Path to image/video file, or `0` for webcam (default: `0`)
- `--model` - Choose model (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8n-seg.pt` for segmentation)
- `--show` - Display results in a window
- `--conf` - Confidence threshold (default: `0.25`)
- `--no-save` - Don't save results (by default, results are saved to `outputs/detect/`)

## Object Classes

If an object doesn't show up, it's probably not in the COCO training dataset. To see all detectable classes:

```bash
python show_classes.py
```
