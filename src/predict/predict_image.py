from prediction_result import PredictionResult
from parseq_ocr import ParseqOCR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import cv2

class PredictImage:
    def __init__(self, yolov8_model_path: str) -> None:
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
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
            overlap_width_ratio=0.2
        )
        for r in shape_results.object_prediction_list:
            x1 = round(r.bbox.minx)
            y1 = round(r.bbox.miny)
            x2 = round(r.bbox.maxx)
            y2 = round(r.bbox.maxy)
            print(x1, x2, y1, y2)
            print(image[y1:y2, x1:x2].shape)
            letter = self.ocr_model.predict(image[y1:y2, x1:x2])
            cv2.imwrite(f"tmp/{r.category.name}.jpg", image[y1:y2, x1:x2])
            results.append(PredictionResult(x1, y1, x2, y2, r.category.name, letter, r.score.value))
        return results

    # def visualize(self, image: np.array, prediction_result: list) -> np.array:
    #     boxes_list = []
    #     clss_list = []
    #     for ind, _ in enumerate(object_prediction_list):
    #         boxes = (
    #             object_prediction_list[ind].bbox.minx,
    #             object_prediction_list[ind].bbox.miny,
    #             object_prediction_list[ind].bbox.maxx,
    #             object_prediction_list[ind].bbox.maxy,
    #         )
    #         clss = object_prediction_list[ind].category.name
    #         boxes_list.append(boxes)
    #         clss_list.append(clss)

    #     for box, cls in zip(boxes_list, clss_list):
    #         x1, y1, x2, y2 = box
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
    #         label = str(cls)
    #         t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
    #         cv2.rectangle(
    #             frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
    #         )
    #         cv2.putText(
    #             frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
    #         )
