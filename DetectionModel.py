from ultralytics import YOLO

class DetectionModel:
    def __init__(self, model_path):
        # Load the YOLOv11 model
        self.model = YOLO(model_path)

        # Map class names to class IDs
        object_names = ["person", "car", "motorcycle", "airplane", "bus", "boat", "stop sign", "snowboard",
                           "umbrella", "sports ball", "baseball bat", "bed", "tennis racket", "suitcase", "skis"]
        name_to_id = {v: k for k, v in self.model.names.items()}

        self.desired_classes_ids = list(map(lambda x: name_to_id[x], object_names))

    def predict(self, image_path):
        """
        Predict objects in an image.

        :param image_path: Path to the image file.

        :return: list of dictionaries.
                Each dictionary holds:
                "object": name of object detected,
                "position": coordinates of object's center,
                "confidence": confidence score.
        """

        # Perform prediction
        results = self.model.predict(source=image_path, classes=self.desired_classes_ids)
        print(len(results))

        print(self.model.names.items())

        detections = []

        # go through each detection and extract the class name, position and confidence
        for result in results:
            boxes = result.boxes
            print(len(boxes))

            for box in boxes:
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()

                # calculate the center of the bounding box
                position = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                detections.append({"class_name": class_name, "position": position, "confidence": confidence})

        # sort list by confidence, in descending order
        detections.sort(key = lambda x: x["confidence"], reverse = True)
        return detections




##TESTING PURPOSES ONLY
model = DetectionModel('./yolo11s.pt')
results = model.predict("./datasets/coco8/images/val/000000000036.jpg")
print(results)