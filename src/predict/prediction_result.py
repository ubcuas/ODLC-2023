class PredictionResult:
    def __init__(self, x1:int, y1:int, x2:int, y2:int, shape:str, letter:str, conf:float) -> None:
        if len(letter) > 1:
            raise Exception("Letter must contain 1 letter")
        self.x1 = x1
        self.y1 = y1
        self.y2 = y2
        self.y2 = y2
        self.shape = shape
        self.letter = letter
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

    def get_shape(self) -> str:
        return self.shape
    
    def get_letter(self) -> str:
        return self.letter
    
    def get_conf(self) -> float:
        return self.conf