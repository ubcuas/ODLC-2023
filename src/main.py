print("program started")

import ultralytics as ult
import cv2
import os


# mode is either "train" or "predict"
mode = "predict"

# num_epochs is the number of times the dataset is fed to the algorithm
num_epochs = 50

# n: nano, s: small, m: medium, l: large (GPU cannot handle), x: extra large (GPU cannot handle)
model_type = "m"

# file paths
path_dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./training/training_datasets/data.yaml")
path_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../runs/detect/train/weights/best.pt")
path_predict = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./predict/images1")

def main():
    if mode == "train":
        # train
        model = ult.YOLO("yolov8" + model_type + ".pt")
        model.train(data=path_dataset, epochs=num_epochs)

        # validate
        model.val()

    if mode == "predict":
        # predict
        model = ult.YOLO(path_weights)
        
        # extract results
        results = model(
            source=path_predict,
            imgsz=640, # image size
            half=False, # use FP16 format
            device=0, # device for computation
            max_det=100, # maximum no. of detections 
            conf=0.3, # minimum confidence
            iou=0.7, # removes bbox if IoU>=iou with another bbox
            classes=None, # filter by class (e.g. 52 is brown_circle)
            show=False,
            save=True,
            line_width=1,
            show_boxes=True, # show bounding boxes
            show_conf=False, # show confidence score
            show_labels=True, # show labels
            save_conf=True, # save results with confidence scores
            save_txt=True, # save results as .txt file
            save_frames=False, # for inference of videos
            augment=False # test-time augmentation
        )
        bboxes = results[0].boxes
        print(bboxes.data)


if __name__ == "__main__":
    main()

print("program ended")