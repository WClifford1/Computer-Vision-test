"""
MediaPipe Pose Detection Demo
Real-time skeleton tracking on webcam, images, or video
"""

import mediapipe as mp
import cv2
import argparse
from pathlib import Path


def run_pose_detection(source, save=True, show=False, conf=0.5):
    """
    Run MediaPipe Pose detection on webcam, image, or video

    Args:
        source: Path to image/video or 0 for webcam
        save: Whether to save results
        show: Whether to display results
        conf: Minimum detection confidence (0.0-1.0)
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy
        min_detection_confidence=conf,
        min_tracking_confidence=conf
    )

    # Check if source is webcam
    is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

    if is_webcam:
        cap = cv2.VideoCapture(source)
        print(f"Starting pose detection on webcam...")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = pose.process(image_rgb)

            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Print detection info
                print("Pose detected")

            if show:
                cv2.imshow('MediaPipe Pose', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Handle image or video file
        source_path = Path(source)

        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Single image
            print(f"Processing image: {source}")
            image = cv2.imread(str(source_path))

            if image is None:
                print(f"Error: Could not read image from {source}")
                return

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = pose.process(image_rgb)

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                print("Pose detected in image")
            else:
                print("No pose detected in image")

            if show:
                cv2.imshow('MediaPipe Pose', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save:
                output_dir = Path('outputs/pose')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'pose_{source_path.name}'
                cv2.imwrite(str(output_path), image)
                print(f"Saved: {output_path}")

        else:
            # Video file
            print(f"Processing video: {source}")
            cap = cv2.VideoCapture(str(source_path))

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup video writer if saving
            if save:
                output_dir = Path('outputs/pose')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'pose_{source_path.name}'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            frame_num = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame
                results = pose.process(image_rgb)

                # Draw pose landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                if save:
                    out.write(frame)

                if show:
                    cv2.imshow('MediaPipe Pose', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_num += 1
                if frame_num % 30 == 0:
                    print(f"Processed {frame_num} frames...")

            cap.release()
            if save:
                out.release()
                print(f"Saved: {output_path}")
            cv2.destroyAllWindows()

    pose.close()


def main():
    parser = argparse.ArgumentParser(description='MediaPipe Pose Detection Demo')
    parser.add_argument('source', nargs='?', default='0',
                        help='Image/video path or 0 for webcam (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--show', action='store_true',
                        help='Display results in a window')

    args = parser.parse_args()

    # Convert source to int if it's a webcam index
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run_pose_detection(
        source=source,
        save=not args.no_save,
        show=args.show,
        conf=args.conf
    )


if __name__ == '__main__':
    main()
