import cv2
import time
from predict_image import PredictImage
from prediction_filter import PredictionFilter

frame = cv2.imread("test_images/20240302_120243_448-image1.jpg")
SHAPE_MODEL_PATH = "model/shape-detection-v1/weights/best.pt"
EMERGENT_MODEL_PATH = "model/emergent-detection-v1/yolov8n.pt"


# TODO: Fetch target data from GCOM
# TODO: Filter out only valid letter from OCR Model
TARGETS = ["purple_triangle_A"]

model = PredictImage(SHAPE_MODEL_PATH, EMERGENT_MODEL_PATH)
result_filter = PredictionFilter(TARGETS)

# TODO: Wrap this in some loop to continously fetch image and GPS coordinate from GCOM
GPS_COORDINATE = (49.2602703, -123.2516302)

start_t = time.time()
shape_result, emergent_result = model.predict(frame)
print(emergent_result)
frame = model.visualize(frame, shape_result, emergent_result)
result_filter.add_prediction(shape_result, GPS_COORDINATE)
print(time.time() - start_t)

cv2.imwrite("output/out.jpg", frame)
# cv2.imshow("frame", cv2.resize(frame, (1336, 1002)))
# cv2.waitKey(0)
