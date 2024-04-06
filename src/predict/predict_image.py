from prediction_result import PredictionResult
from parseq_ocr import ParseqOCR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import cv2


class PredictImage:
    def __init__(self, yolov8_model_path: str) -> None:
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=yolov8_model_path,
            confidence_threshold=0.6,
            device="cuda:0",  # or 'cuda:0'
        )
        self.ocr_model = ParseqOCR()

    def predict(self, image: np.array) -> list[PredictionResult]:
        results = []
        shape_results = get_sliced_prediction(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            self.detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        shape_images = []
        for r in shape_results.object_prediction_list:
            x1 = round(r.bbox.minx)
            y1 = round(r.bbox.miny)
            x2 = round(r.bbox.maxx)
            y2 = round(r.bbox.maxy)
            shape_images.append(image[y1:y2, x1:x2])

        letter, conf = self.ocr_model.batch_predict(shape_images)
        for i, r in enumerate(shape_results.object_prediction_list):
            x1 = round(r.bbox.minx)
            y1 = round(r.bbox.miny)
            x2 = round(r.bbox.maxx)
            y2 = round(r.bbox.maxy)
            results.append(
                PredictionResult(
                    x1, y1, x2, y2, r.category.name, letter[i], r.score.value
                )
            )
        return results

    def visualize(
        self, image: np.array, prediction_results: list[PredictionResult]
    ) -> np.array:
        for pred in prediction_results:
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
