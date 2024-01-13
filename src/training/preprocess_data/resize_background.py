import cv2
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="backgrounds-resized")
parser.add_argument(
        "-i",
        "--input",
        default="backgrounds",
        help="Backgroud image folder to put object on",
    )
# parser.add_argument(
#         "-w",
#         "--width",
#         default=640,
#         help="Backgroud image folder to put object on",
#     )
args = parser.parse_args()
output_path = Path(args.output)
output_path.mkdir(exist_ok=True, parents=True)
input_path = Path(args.input)
imgs_path = list(input_path.glob("*.*"))


for img_path in imgs_path:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (853, 640))
    cv2.imwrite(str(output_path.joinpath(img_path.name)), img)
    