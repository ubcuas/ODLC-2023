import time
import ultralytics as ult
import os
import glob
import requests
import shutil

print("program started")

# mode is either "train" or "predict"
mode = "predict"

# num_epochs is the number of times the dataset is fed to the algorithm
num_epochs = 50

# n: nano, s: small, m: medium, l: large (GPU cannot handle), x: extra large (GPU cannot handle)
model_type = "m"

# file paths
path_dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./training/training_datasets/data.yaml")
path_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../runs/detect/train/weights/best.pt")
path_predict_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./predict/live_feed_new")
path_old_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./predict/live_feed_old")

labels = {
    0: 'white_triangle', 1: 'white_rectangle', 2: 'white_pentagon', 3: 'white_star', 4: 'white_circle', 
    5: 'white_semicircle', 6: 'white_quarter circle', 7: 'white_cross', 8: 'black_triangle', 
    9: 'black_rectangle', 10: 'black_pentagon', 11: 'black_star', 12: 'black_circle', 13: 'black_semicircle', 
    14: 'black_quarter circle', 15: 'black_cross', 16: 'red_triangle', 17: 'red_rectangle', 
    18: 'red_pentagon', 19: 'red_star', 20: 'red_circle', 21: 'red_semicircle', 22: 'red_quarter circle', 
    23: 'red_cross', 24: 'blue_triangle', 25: 'blue_rectangle', 26: 'blue_pentagon', 27: 'blue_star', 
    28: 'blue_circle', 29: 'blue_semicircle', 30: 'blue_quarter circle', 31: 'blue_cross', 32: 'green_triangle', 
    33: 'green_rectangle', 34: 'green_pentagon', 35: 'green_star', 36: 'green_circle', 37: 'green_semicircle', 
    38: 'green_quarter circle', 39: 'green_cross', 40: 'purple_triangle', 41: 'purple_rectangle', 
    42: 'purple_pentagon', 43: 'purple_star', 44: 'purple_circle', 45: 'purple_semicircle', 
    46: 'purple_quarter circle', 47: 'purple_cross', 48: 'brown_triangle', 49: 'brown_rectangle', 
    50: 'brown_pentagon', 51: 'brown_star', 52: 'brown_circle', 53: 'brown_semicircle', 
    54: 'brown_quarter circle', 55: 'brown_cross', 56: 'orange_triangle', 57: 'orange_rectangle', 
    58: 'orange_pentagon', 59: 'orange_star', 60: 'orange_circle', 61: 'orange_semicircle', 
    62: 'orange_quarter circle', 63: 'orange_cross'
}

# List of labels to trigger the HTTP request
trigger_labels = ['red_circle', 'blue_triangle', 'black_pentagon']  # Example list of labels

def get_latest_file(directory):
    """Get the latest file from the directory"""
    list_of_files = glob.glob(os.path.join(directory, '*')) # * means all files
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def send_http_request():
    """Send HTTP POST request to localhost:1323"""
    url = "http://localhost:1323/odlc-found"
    payload = {'found': 'true'}
    headers = {'Content-Type': 'application/json'}
    response = None
    try:
        response = requests.post(url, json=payload, headers=headers)
    except requests.exceptions.ConnectionError:
        print("Failed to send HTTP request! Connection refused.")
    if response != None and response.status_code == 200:
        print("HTTP request sent successfully!")
    else:
        print(f"Failed to send HTTP request! Response: {response}")

def main():
    if mode == "train":
        # train
        model = ult.YOLO("yolov8" + model_type + ".pt")
        model.train(data=path_dataset, epochs=num_epochs)

        # validate
        model.val()

    if mode == "predict":
        model = ult.YOLO(path_weights)

        while True:
            latest_file = get_latest_file(path_predict_dir)
            if latest_file:
                # predict
                results = model(
                    source=latest_file,
                    imgsz=640,  # image size
                    half=False,  # use FP16 format
                    device="cpu",  # device for computation
                    max_det=100,  # maximum no. of detections 
                    conf=0.3,  # minimum confidence
                    iou=0.7,  # removes bbox if IoU>=iou with another bbox
                    classes=None,  # filter by class (e.g. 52 is brown_circle)
                    show=False,
                    save=True,
                    line_width=1,
                    show_boxes=True,  # show bounding boxes
                    show_conf=False,  # show confidence score
                    show_labels=True,  # show labels
                    save_conf=True,  # save results with confidence scores
                    save_txt=True,  # save results as .txt file
                    save_frames=False,  # for inference of videos
                    augment=False  # test-time augmentation
                )
                bboxes = results[0].boxes
                print(bboxes.data)

                # Check if any trigger label is detected and send HTTP request
                for bbox in bboxes:
                    class_index = int(bbox.cls[0])
                    label = labels.get(class_index, "Unknown")
                    if label in trigger_labels:
                        send_http_request()


                # Move the processed file to the old directory
                if not os.path.exists(path_old_dir):
                    os.makedirs(path_old_dir)
                shutil.move(latest_file, path_old_dir)

            # Sleep for a specified duration (e.g., 60 seconds) before checking again
            time.sleep(10)

if __name__ == "__main__":
    main()

print("program ended")
