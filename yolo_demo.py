"""
YOLO Object Detection Demo
Simple script to get started with YOLOv8 using Ultralytics
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path


def run_inference(source, model_name='yolov8n.pt', save=True, show=False, conf=0.25):
    """
    Run YOLO inference on an image, video, or webcam

    Args:
        source: Path to image/video or 0 for webcam
        model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        save: Whether to save results
        show: Whether to display results
        conf: Confidence threshold for detections
    """
    # Load the model (downloads automatically if not present)
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Check if source is a webcam/stream
    is_stream = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

    if show and is_stream:
        # For real-time webcam display, use streaming mode with cv2.imshow
        print(f"Running real-time inference on webcam...")
        print("Press 'q' to quit")

        results_stream = model(source=source, conf=conf, stream=True)

        for result in results_stream:
            # Get the annotated frame
            annotated_frame = result.plot()

            # Display the frame
            cv2.imshow('YOLO Detection', annotated_frame)

            # Print detections to console
            boxes = result.boxes
            if len(boxes) > 0:
                classes = boxes.cls.cpu().numpy()
                class_names = [model.names[int(c)] for c in classes]
                unique_classes = set(class_names)
                detection_str = ", ".join([f"{class_names.count(cls)} {cls}" for cls in unique_classes])
                print(f"Detected: {detection_str}")

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return None
    else:
        # For images/videos or non-display mode
        print(f"Running inference on {source}...")
        results = model(
            source=source,
            conf=conf,
            save=save,
            project='outputs',
            name='detect',
            exist_ok=True
        )

        # Display results for images
        if show:
            for result in results:
                result.show()

        # Print detection summary
        for i, result in enumerate(results):
            boxes = result.boxes
            print(f"\nFrame {i}: Detected {len(boxes)} objects")

            # Print class counts
            if len(boxes) > 0:
                classes = boxes.cls.cpu().numpy()
                class_names = [model.names[int(c)] for c in classes]
                unique_classes = set(class_names)

                for cls in unique_classes:
                    count = class_names.count(cls)
                    print(f"  - {cls}: {count}")

        if save:
            print(f"\nResults saved to: outputs/detect/")
        return results


def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection Demo')
    parser.add_argument('source', nargs='?', default='0',
                        help='Image/video path or 0 for webcam (default: 0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model to use (default: yolov8n.pt). Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--show', action='store_true',
                        help='Display results in a window')

    args = parser.parse_args()

    # Convert source to int if it's a webcam index
    source = args.source
    if source.isdigit():
        source = int(source)

    run_inference(
        source=source,
        model_name=args.model,
        save=not args.no_save,
        show=args.show,
        conf=args.conf
    )


if __name__ == '__main__':
    main()
