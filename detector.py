from ultralytics import YOLO
from utils import draw_detections

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        """
        Initialise the YOLO object detector.
        model_path: which YOLO model to use (yolov8n = nano, smallest & fastest)
        confidence: minimum confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image):
        """
        Run detection on an image and return annotated image + detection summary.
        """
        results = self.model(image, conf=self.confidence)
        annotated_image = draw_detections(image.copy(), results)
        detections = self._parse_detections(results)
        return annotated_image, detections

    def _parse_detections(self, results):
        """
        Extract detection details into a readable list.
        """
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "label": result.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 2),
                    "bbox": list(map(int, box.xyxy[0]))
                })
        return detections