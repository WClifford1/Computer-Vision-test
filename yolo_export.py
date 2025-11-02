"""
YOLO Detection with Full Raw Data Export
Exports ALL data that YOLO provides - no filtering or cleaning
"""

from ultralytics import YOLO
import argparse
import json
from pathlib import Path
import numpy as np


def export_raw_detections(results, model, output_path):
    """
    Export ALL raw detection data from YOLO results

    Args:
        results: YOLO results object
        model: YOLO model (for class names)
        output_path: Path to save JSON file
    """
    all_data = []

    for i, result in enumerate(results):
        frame_data = {
            "frame_id": i,
            "image_path": str(result.path) if hasattr(result, 'path') else None,
            "image_shape": {
                "original": list(result.orig_shape) if hasattr(result, 'orig_shape') else None,
            },
            "speed": result.speed if hasattr(result, 'speed') else None,  # Inference speed metrics
            "detections": []
        }

        # Extract ALL box data
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes

            for j in range(len(boxes)):
                box = boxes[j]

                detection = {
                    "box_id": j,
                    # All box coordinate formats
                    "xyxy": box.xyxy[0].cpu().numpy().tolist() if hasattr(box, 'xyxy') else None,  # [x1, y1, x2, y2]
                    "xywh": box.xywh[0].cpu().numpy().tolist() if hasattr(box, 'xywh') else None,  # [x_center, y_center, width, height]
                    "xyxyn": box.xyxyn[0].cpu().numpy().tolist() if hasattr(box, 'xyxyn') else None,  # normalized [x1, y1, x2, y2]
                    "xywhn": box.xywhn[0].cpu().numpy().tolist() if hasattr(box, 'xywhn') else None,  # normalized [x_center, y_center, width, height]

                    # Confidence and class
                    "confidence": float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else None,
                    "class_id": int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else None,
                    "class_name": model.names[int(box.cls[0].cpu().numpy())] if hasattr(box, 'cls') else None,

                    # Track ID (if tracking is enabled)
                    "track_id": int(box.id[0].cpu().numpy()) if hasattr(box, 'id') and box.id is not None else None,

                    # Raw data
                    "raw_data": box.data[0].cpu().numpy().tolist() if hasattr(box, 'data') else None,
                }

                frame_data["detections"].append(detection)

        # Extract ALL segmentation mask data (if available)
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            frame_data["masks"] = {
                "count": len(masks.data) if hasattr(masks, 'data') else 0,
                "mask_data": []
            }

            try:
                # Get all polygon coordinates at once (safer)
                xy_coords = masks.xy if hasattr(masks, 'xy') else []
                xyn_coords = masks.xyn if hasattr(masks, 'xyn') else []

                for j in range(len(xy_coords)):
                    mask_info = {
                        "mask_id": j,
                        # Polygon coordinates (outline of mask)
                        "xy": xy_coords[j].tolist() if j < len(xy_coords) else None,
                        "xyn": xyn_coords[j].tolist() if j < len(xyn_coords) else None,  # normalized

                        # Full pixel mask (can be large!)
                        "data_shape": list(masks.data.shape) if hasattr(masks, 'data') and masks.data is not None else None,
                        # Note: Not saving full pixel array by default as it's huge - enable below if needed
                        # "pixel_mask": masks.data[j].cpu().numpy().tolist() if hasattr(masks, 'data') else None,
                    }

                    frame_data["masks"]["mask_data"].append(mask_info)
            except Exception as e:
                print(f"Warning: Could not extract all mask data: {e}")
                # Still save what we can
                frame_data["masks"]["error"] = str(e)

        # Extract keypoints data (if available - for pose detection)
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints
            frame_data["keypoints"] = {
                "count": len(keypoints),
                "keypoint_data": []
            }

            for j in range(len(keypoints)):
                kp = keypoints[j]
                kp_info = {
                    "keypoint_id": j,
                    "xy": kp.xy[0].cpu().numpy().tolist() if hasattr(kp, 'xy') else None,
                    "xyn": kp.xyn[0].cpu().numpy().tolist() if hasattr(kp, 'xyn') else None,
                    "conf": kp.conf[0].cpu().numpy().tolist() if hasattr(kp, 'conf') else None,
                }
                frame_data["keypoints"]["keypoint_data"].append(kp_info)

        # Extract oriented bounding boxes (if available - for OBB detection)
        if hasattr(result, 'obb') and result.obb is not None:
            obb = result.obb
            frame_data["obb"] = {
                "count": len(obb),
                "obb_data": []
            }

            for j in range(len(obb)):
                obb_box = obb[j]
                obb_info = {
                    "obb_id": j,
                    "xyxyxyxy": obb_box.xyxyxyxy[0].cpu().numpy().tolist() if hasattr(obb_box, 'xyxyxyxy') else None,
                    "xywhr": obb_box.xywhr[0].cpu().numpy().tolist() if hasattr(obb_box, 'xywhr') else None,
                    "conf": float(obb_box.conf[0].cpu().numpy()) if hasattr(obb_box, 'conf') else None,
                    "cls": int(obb_box.cls[0].cpu().numpy()) if hasattr(obb_box, 'cls') else None,
                }
                frame_data["obb"]["obb_data"].append(obb_info)

        # Add summary counts
        frame_data["summary"] = {
            "total_detections": len(frame_data["detections"]),
            "total_masks": len(frame_data["masks"]["mask_data"]) if "masks" in frame_data else 0,
            "total_keypoints": len(frame_data["keypoints"]["keypoint_data"]) if "keypoints" in frame_data else 0,
            "total_obb": len(frame_data["obb"]["obb_data"]) if "obb" in frame_data else 0,
        }

        all_data.append(frame_data)

    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\n✓ Exported ALL raw detection data to: {output_path}")

    # Print summary
    print(f"\n=== Data Export Summary ===")
    for frame in all_data:
        print(f"\nFrame {frame['frame_id']}:")
        print(f"  Total detections: {frame['summary']['total_detections']}")
        if frame['summary']['total_masks'] > 0:
            print(f"  Total masks: {frame['summary']['total_masks']}")
        if frame['summary']['total_keypoints'] > 0:
            print(f"  Total keypoints: {frame['summary']['total_keypoints']}")

        for det in frame['detections']:
            print(f"  - {det['class_name']}: conf={det['confidence']:.3f}")

    return all_data


def run_inference_with_export(source, model_name='yolov8n.pt', conf=0.25, show=False):
    """
    Run YOLO inference and export ALL raw data

    Args:
        source: Path to image/video
        model_name: YOLO model to use
        conf: Confidence threshold
        show: Whether to display results
    """
    # Load model
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Run inference
    print(f"Running inference on {source}...")
    results = model(
        source=source,
        conf=conf,
        save=True,
        project='outputs',
        name='detect',
        exist_ok=True
    )

    # Show results if requested
    if show:
        for result in results:
            result.show()

    # Prepare output path
    source_path = Path(source)
    output_dir = Path('outputs/detect')
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = source_path.stem if source_path.is_file() else 'detections'
    json_path = output_dir / f'{base_name}_raw_data.json'

    # Export ALL data
    export_raw_detections(results, model, json_path)

    print(f"\n✓ Annotated image saved to: outputs/detect/")


def main():
    parser = argparse.ArgumentParser(description='YOLO Detection with Full Raw Data Export')
    parser.add_argument('source',
                        help='Path to image or video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model to use (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--show', action='store_true',
                        help='Display results')

    args = parser.parse_args()

    run_inference_with_export(
        source=args.source,
        model_name=args.model,
        conf=args.conf,
        show=args.show
    )


if __name__ == '__main__':
    main()
