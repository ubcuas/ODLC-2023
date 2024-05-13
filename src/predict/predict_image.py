from prediction_result.shape_result import ShapeResult
from prediction_result.prediction_result import PredictionResult
from parseq_ocr import ParseqOCR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import cv2


class PredictImage:
    # Maximum possible shape size in pixels
    MAX_SHAPE_SIZE = 70
    # Maximum possible emergent object size in pixels
    MAX_EMERGENT_SIZE = 1000

    def __init__(self, yolov8_model_path: str, emerging_object_model_path: str) -> None:
        # Initialize Shape Detection Model
        self.shape_detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=yolov8_model_path,
            confidence_threshold=0.65,
            device="cuda:0",
        )
        # Initialize Emerging Object Detection Model
        self.emerging_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=emerging_object_model_path,
            confidence_threshold=0.65,
            device="cuda:0",
        )
        # Initialize OCR Model
        self.ocr_model = ParseqOCR()

    def predict(
        self, image: np.array
    ) -> tuple[list[ShapeResult], list[PredictionResult]]:
        # Detect Shape
        shape_results = get_sliced_prediction(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            self.shape_detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # Filter out shapes that's too large (definitely not a valid shape)
        shape_results_filtered = []
        for r in shape_results.object_prediction_list:
            if (
                r.bbox.maxy - r.bbox.miny < self.MAX_SHAPE_SIZE
                and r.bbox.maxx - r.bbox.minx < self.MAX_SHAPE_SIZE
            ):
                shape_results_filtered.append(r)

        # Detect Emergent object
        emergent_results = get_sliced_prediction(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            self.emerging_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # Select only Person class
        emergent_results_filtered = []
        for r in emergent_results.object_prediction_list:
            if (
                r.bbox.maxy - r.bbox.miny < self.MAX_EMERGENT_SIZE
                and r.bbox.maxx - r.bbox.minx < self.MAX_EMERGENT_SIZE
                and r.category.name == "person"
            ):
                emergent_results_filtered.append(r)

        # Slice out detected shapes image for passing into OCR model
        shape_images = []
        for r in shape_results_filtered:
            x1 = round(r.bbox.minx)
            y1 = round(r.bbox.miny)
            x2 = round(r.bbox.maxx)
            y2 = round(r.bbox.maxy)
            shape_images.append(image[y1:y2, x1:x2])

        # OCR
        letter, conf = self.ocr_model.batch_predict(shape_images)

        # Format result into list of ShapeResult
        shape_results_formatted = []
        for i, r in enumerate(shape_results_filtered):
            x1 = round(r.bbox.minx)
            y1 = round(r.bbox.miny)
            x2 = round(r.bbox.maxx)
            y2 = round(r.bbox.maxy)
            shape_results_formatted.append(
                ShapeResult(x1, y1, x2, y2, r.category.name, letter[i], r.score.value)
            )

        emergent_results_formatted = []
        for r in emergent_results_filtered:
            x1 = round(r.bbox.minx)
            y1 = round(r.bbox.miny)
            x2 = round(r.bbox.maxx)
            y2 = round(r.bbox.maxy)
            emergent_results_formatted.append(
                PredictionResult(x1, y1, x2, y2, r.score.value)
            )
        return shape_results_formatted, emergent_results_formatted

    def visualize(
        self,
        image: np.array,
        shape_results: list[ShapeResult],
        emergent_results: list[PredictionResult],
    ) -> np.array:
        for pred in shape_results:
            # draw bbox
            cv2.rectangle(
                image,
                (pred.get_x1(), pred.get_y1()),
                (pred.get_x2(), pred.get_y2()),
                (56, 56, 255),
                2,
            )
            # draw text and textbox
            display_text = (
                f"{pred.get_shape()} {pred.get_conf():.2f} {pred.get_letter()}"
            )
            t_size = cv2.getTextSize(display_text, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(
                image,
                (pred.get_x1(), pred.get_y1() - t_size[1] - 3),
                (pred.get_x1() + t_size[0], pred.get_y1() + 3),
                (56, 56, 255),
                -1,
            )
            cv2.putText(
                image,
                display_text,
                (pred.get_x1(), pred.get_y1() - 2),
                0,
                0.6,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        for pred in emergent_results:
            # draw bbox
            cv2.rectangle(
                image,
                (pred.get_x1(), pred.get_y1()),
                (pred.get_x2(), pred.get_y2()),
                (0, 255, 0),
                2,
            )
        return image
