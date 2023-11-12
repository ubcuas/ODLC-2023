# TODO: main running script
print("program started")

from ultralytics import YOLO
import os

def main():
    print(os.getcwd())

    # file paths
    file_path_train_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./training/training_datasets/data.yaml")
    file_path_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../runs/detect/train/weights/best.pt")
    file_path_test = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./predict/images")

    # train
    model = YOLO("yolov8s.pt")  # pass any model type
    model.train(
        data=file_path_train_data,
        epochs=3
    )  # epochs is how many times the dataset is passed through the algorithm

    # validate
    model.val()

    # # predict
    # model = YOLO(
    #     file_path_weights
    # )
    # model.predict(
    #     source=file_path_test,
    #     show=True,
    #     save=True,
    # )


if __name__ == "__main__":
    main()

print("program ended")