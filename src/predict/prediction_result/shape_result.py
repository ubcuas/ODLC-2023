from .prediction_result import PredictionResult


class ShapeResult(PredictionResult):
    def __init__(
        self, x1: int, y1: int, x2: int, y2: int, shape: str, letter: str, conf: float
    ) -> None:
        super().__init__(x1, y1, x2, y2, conf)
        if len(letter) > 1:
            raise Exception("Letter must contain 1 letter")
        self.shape = shape
        self.letter = letter

    def get_shape(self) -> str:
        return self.shape

    def get_letter(self) -> str:
        return self.letter
