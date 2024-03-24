import cv2
import time
from predict_image import PredictImage
from prediction_filter import PredictionFilter

frame = cv2.imread("test_images/20240302_120243_448-image1.jpg")
MODEL_PATH = "model-1/weights/best.pt"


# TODO: Fetch target data from GCOM
TARGETS = ["purple_triangle_A"]

model = PredictImage(MODEL_PATH)
result_filter = PredictionFilter(TARGETS)

# TODO: Wrap this in some loop to continously fetch image and GPS coordinate from GCOM
GPS_COORRDINATE = (49.2602703, -123.2516302)

start_t = time.time()
result = model.predict(frame)
result_filter.add_prediction(result, GPS_COORRDINATE)
print(time.time() - start_t)


# cv2.imwrite("out.jpg", frame)
cv2.imshow("frame", cv2.resize(frame, (1336, 1002)))
cv2.waitKey(0)
