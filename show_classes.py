"""
Display all COCO classes that YOLO can detect
"""
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

print("YOLO can detect these 80 object classes:\n")
print("=" * 50)

for class_id, class_name in model.names.items():
    print(f"{class_id:2d}: {class_name}")

print("=" * 50)
print(f"\nTotal: {len(model.names)} classes")
