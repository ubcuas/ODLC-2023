from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import time

frame = cv2.imread("images/trained_bg/images/0.jpg")
yolov8_model_path = "model/weights/best.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.6,
    device="cuda:0",  # or 'cuda:0'
)

start_t = time.time()
for i in range(20):
    results = get_sliced_prediction(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
print(time.time() - start_t)

# Save prediction result
results.export_visuals(export_dir=".", file_name="trained-bg0.jpg")

object_prediction_list = results.object_prediction_list

boxes_list = []
clss_list = []
for ind, _ in enumerate(object_prediction_list):
    boxes = (
        object_prediction_list[ind].bbox.minx,
        object_prediction_list[ind].bbox.miny,
        object_prediction_list[ind].bbox.maxx,
        object_prediction_list[ind].bbox.maxy,
    )
    clss = object_prediction_list[ind].category.name
    boxes_list.append(boxes)
    clss_list.append(clss)

for box, cls in zip(boxes_list, clss_list):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
    label = str(cls)
    t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
    cv2.rectangle(
        frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
    )
    cv2.putText(
        frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
    )

cv2.imwrite("out.jpg", frame)
cv2.imshow("frame", cv2.resize(frame, (1336, 1002)))
cv2.waitKey(0)
