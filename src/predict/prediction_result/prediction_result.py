class PredictionResult:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float) -> None:
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise Exception("Pixel coordinate must be positive")
        if conf < 0 or conf > 1:
            raise Exception("Confidence Score must be between 0 and 1")
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf

    def get_x1(self) -> int:
        return self.x1

    def get_y1(self) -> int:
        return self.y1

    def get_x2(self) -> int:
        return self.x2

    def get_y2(self) -> int:
        return self.y2

    def get_center_x(self) -> int:
        return round((self.get_x1() - self.get_x2) / 2)

    def get_center_y(self) -> int:
        return round((self.get_y1() - self.get_y2) / 2)

    def get_conf(self) -> float:
        return self.conf

    def __str__(self) -> str:
        return f"pt1 = ({self.get_x1()}, {self.get_y1()})  pt2 = ({self.get_x2()}, {self.get_y2()})"
