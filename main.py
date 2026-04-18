import cv2
import os
import argparse
from detector import ObjectDetector
from utils import load_image, save_image

def run_on_image(image_path, output_path, confidence=0.5):
    """Run detection on a single image."""
    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    print("Running detection...")
    detector = ObjectDetector(confidence=confidence)
    annotated_image, detections = detector.detect(image)

    os.makedirs("results", exist_ok=True)
    save_image(annotated_image, output_path)

    print(f"\nDetections found: {len(detections)}")
    for d in detections:
        print(f"  - {d['label']} | Confidence: {d['confidence']} | BBox: {d['bbox']}")

def run_on_webcam(confidence=0.5):
    """Run real-time detection using webcam."""
    print("Starting webcam detection... Press 'q' to quit.")
    detector = ObjectDetector(confidence=confidence)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, detections = detector.detect(frame)
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

def main():
    parser = argparse.ArgumentParser(description="Vision-Based Object Detection System")
    parser.add_argument("--mode", type=str, default="image",
                        choices=["image", "webcam"],
                        help="Run on image or webcam")
    parser.add_argument("--input", type=str, default="sample_images/test.jpg",
                        help="Path to input image (image mode only)")
    parser.add_argument("--output", type=str, default="results/output.jpg",
                        help="Path to save output image (image mode only)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold (0-1)")
    args = parser.parse_args()

    if args.mode == "image":
        run_on_image(args.input, args.output, args.confidence)
    elif args.mode == "webcam":
        run_on_webcam(args.confidence)

if __name__ == "__main__":
    main()