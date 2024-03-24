import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image


class ParseqOCR:
    MODEL_NAME = "parseq"

    def __init__(self, model_name=MODEL_NAME) -> None:
        self._preprocess = T.Compose(
            [
                T.Resize((32, 128), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )
        self.model = self._get_model(model_name)

    def _get_model(self, name):
        model = torch.hub.load(
            "baudm/parseq", name, pretrained=True, trust_repo=True
        ).eval()
        return model

    @torch.inference_mode()
    def predict(self, image: np.array, model_name=MODEL_NAME) -> str:
        """Return the letter with highest confidence in images"""
        image = Image.fromarray(image)
        image = self._preprocess(image).unsqueeze(0)
        # Greedy decoding
        pred = self.model(image).softmax(-1)
        label, _ = self.model.tokenizer.decode(pred)
        raw_label, raw_confidence = self.model.tokenizer.decode(pred, raw=True)
        # Format confidence values
        max_len = 25 if model_name == "crnn" else len(label[0]) + 1
        index = torch.argmax(raw_confidence[0][:max_len - 1])
        return label[0][index], raw_confidence[0][:max_len - 1].tolist()


if __name__ == "__main__":
    import cv2
    from PIL import Image

    ocr = ParseqOCR()
    img = cv2.imread("test_images/purple_triangle.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # img = torch.FloatTensor(img.transpose(2, 0, 1))
    # img = img / 255
    # print(img.shape)
    # print(img)
    print(ocr.predict(img))
