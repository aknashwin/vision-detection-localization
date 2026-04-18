import cv2

def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def save_image(image, output_path):
    """Save an image to a file path."""
    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")

def draw_detections(image, results):
    """Draw bounding boxes and labels on the image."""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = result.names[class_id]

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image