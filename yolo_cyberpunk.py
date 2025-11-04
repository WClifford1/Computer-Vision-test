"""
YOLO Cyberpunk Visualization
Neon-style object detection with futuristic HUD aesthetics
"""

from ultralytics import YOLO
import cv2
import numpy as np
import argparse


# Cyberpunk color palette (BGR format for OpenCV)
NEON_COLORS = {
    'person': (255, 0, 255),      # Magenta
    'car': (255, 0, 128),          # Hot pink
    'truck': (255, 0, 128),        # Hot pink
    'bus': (255, 0, 128),          # Hot pink
    'bicycle': (0, 255, 255),      # Cyan
    'motorcycle': (0, 255, 255),   # Cyan
    'dog': (0, 128, 255),          # Orange
    'cat': (0, 128, 255),          # Orange
    'bird': (0, 255, 0),           # Green
    'default': (255, 255, 0)       # Yellow
}


def draw_corner_brackets(img, box, color, thickness=2, length_ratio=0.15):
    """Draw cyberpunk-style corner brackets instead of full rectangle"""
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1

    corner_length_x = int(width * length_ratio)
    corner_length_y = int(height * length_ratio)

    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + corner_length_x, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length_y), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - corner_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length_y), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + corner_length_x, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length_y), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - corner_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length_y), color, thickness)


def add_neon_glow(img, box, color, glow_intensity=3):
    """Add neon glow effect around the box"""
    x1, y1, x2, y2 = map(int, box)

    # Create multiple layers with decreasing opacity for glow effect
    for i in range(glow_intensity, 0, -1):
        thickness = i * 2
        alpha = 0.3 / i  # Decreasing opacity for outer layers

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud_label(img, text, position, color, confidence):
    """Draw futuristic HUD-style label with background"""
    x, y = position

    # Create label text
    label = f"[ {text} ]"
    conf_text = f"{confidence:.0%}"

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Get text size for background
    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale - 0.1, 1)

    # Draw semi-transparent background
    padding = 5
    bg_y1 = max(0, y - label_h - padding * 2)
    bg_y2 = y
    bg_x1 = x
    bg_x2 = x + max(label_w, conf_w) + padding * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Draw border
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

    # Draw text with glow
    for offset in [(2, 2), (1, 1)]:
        cv2.putText(img, label, (x + padding + offset[0], y - label_h - padding + offset[1]),
                    font, font_scale, (0, 0, 0), font_thickness + 1)
    cv2.putText(img, label, (x + padding, y - label_h - padding),
                font, font_scale, color, font_thickness)

    # Draw confidence
    cv2.putText(img, conf_text, (x + padding, y - padding),
                font, font_scale - 0.1, color, 1)


def add_scanlines(img, intensity=0.1, line_spacing=4):
    """Add CRT-style scanline effect"""
    overlay = img.copy()
    height, width = img.shape[:2]

    for y in range(0, height, line_spacing):
        cv2.line(overlay, (0, y), (width, y), (0, 0, 0), 1)

    cv2.addWeighted(overlay, intensity, img, 1 - intensity, 0, img)


def get_class_color(class_name):
    """Get neon color for a class"""
    return NEON_COLORS.get(class_name, NEON_COLORS['default'])


def cyberpunk_visualize(result, model):
    """Apply cyberpunk visualization to YOLO result"""
    img = result.orig_img.copy()

    # Add slight vignette effect
    height, width = img.shape[:2]
    vignette = np.ones_like(img, dtype=np.float32)
    center_x, center_y = width // 2, height // 2

    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            vignette[y, x] = 1 - (dist / max_dist) * 0.3

    img = (img * vignette).astype(np.uint8)

    # Process detections
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            class_name = model.names[int(cls)]
            color = get_class_color(class_name)

            # Add neon glow
            add_neon_glow(img, box, color, glow_intensity=3)

            # Draw corner brackets
            draw_corner_brackets(img, box, color, thickness=2, length_ratio=0.15)

            # Draw HUD label
            x1, y1, x2, y2 = map(int, box)
            draw_hud_label(img, class_name.upper(), (x1, y1), color, conf)

    # Add scanlines
    add_scanlines(img, intensity=0.08, line_spacing=4)

    # Add slight chromatic aberration on edges for extra cyberpunk feel
    shift = 2
    b, g, r = cv2.split(img)
    b_shifted = np.roll(b, shift, axis=1)
    r_shifted = np.roll(r, -shift, axis=1)
    img = cv2.merge([b_shifted, g, r_shifted])

    return img


def run_cyberpunk_inference(source, model_name='yolov8n.pt', save=True, show=False, conf=0.25):
    """Run YOLO inference with cyberpunk visualization"""
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Check if source is a webcam/stream
    is_stream = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

    if show and is_stream:
        # Real-time webcam with cyberpunk effect
        print(f"Running cyberpunk detection on webcam...")
        print("Press 'q' to quit")

        results_stream = model(source=source, conf=conf, stream=True)

        for result in results_stream:
            # Apply cyberpunk visualization
            cyberpunk_frame = cyberpunk_visualize(result, model)

            # Display the frame
            cv2.imshow('YOLO CYBERPUNK', cyberpunk_frame)

            # Print detections
            boxes = result.boxes
            if len(boxes) > 0:
                classes = boxes.cls.cpu().numpy()
                class_names = [model.names[int(c)] for c in classes]
                unique_classes = set(class_names)
                detection_str = ", ".join([f"{class_names.count(cls)} {cls}" for cls in unique_classes])
                print(f">> DETECTED: {detection_str}")

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return None
    else:
        # For images/videos
        print(f"Running cyberpunk inference on {source}...")
        results = model(source=source, conf=conf)

        # Process and save results
        for i, result in enumerate(results):
            cyberpunk_img = cyberpunk_visualize(result, model)

            if show:
                cv2.imshow(f'YOLO CYBERPUNK - Frame {i}', cyberpunk_img)
                cv2.waitKey(0)

            if save:
                from pathlib import Path
                output_dir = Path('outputs/cyberpunk')
                output_dir.mkdir(parents=True, exist_ok=True)

                if isinstance(source, (int, str)) and str(source).isdigit():
                    output_path = output_dir / f'webcam_{i}.jpg'
                else:
                    source_path = Path(source)
                    output_path = output_dir / f'cyberpunk_{source_path.stem}_{i}{source_path.suffix}'

                cv2.imwrite(str(output_path), cyberpunk_img)
                print(f"Saved: {output_path}")

            # Print detection summary
            boxes = result.boxes
            print(f"\n>> Frame {i}: Detected {len(boxes)} objects")
            if len(boxes) > 0:
                classes = boxes.cls.cpu().numpy()
                class_names = [model.names[int(c)] for c in classes]
                unique_classes = set(class_names)
                for cls in unique_classes:
                    count = class_names.count(cls)
                    print(f"   - {cls}: {count}")

        cv2.destroyAllWindows()
        return results


def main():
    parser = argparse.ArgumentParser(description='YOLO Cyberpunk Visualization')
    parser.add_argument('source', nargs='?', default='0',
                        help='Image/video path or 0 for webcam (default: 0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model to use (default: yolov8n.pt)')
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

    run_cyberpunk_inference(
        source=source,
        model_name=args.model,
        save=not args.no_save,
        show=args.show,
        conf=args.conf
    )


if __name__ == '__main__':
    main()
