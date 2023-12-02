# colors
white = ["white", [255, 255, 255]]
black = ["black", [0, 0, 0]]
red = ["red", [0, 0, 255]]
blue = ["blue", [255, 0, 0]]
green = ["green", [0, 128, 0]]
purple = ["purple", [128, 0, 128]]
brown = ["brown", [0, 75, 150]]
orange = ["orange", [0, 165, 255]]
colors = [white, black, red, blue, green, purple, brown, orange]

# shapes
shapes = [
    "triangle",
    "rectangle",
    "pentagon",
    "star",
    "circle",
    "semicircle",
    "quarter circle",
    "cross",
]
# shapes = ["triangle", "quarter circle"]

lookup_table = [
    "white_triangle",
    "white_rectangle",
    "white_pentagon",
    "white_star",
    "white_circle",
    "white_semicircle",
    "white_quarter circle",
    "white_cross",
    "black_triangle",
    "black_rectangle",
    "black_pentagon",
    "black_star",
    "black_circle",
    "black_semicircle",
    "black_quarter circle",
    "black_cross",
    "red_triangle",
    "red_rectangle",
    "red_pentagon",
    "red_star",
    "red_circle",
    "red_semicircle",
    "red_quarter circle",
    "red_cross",
    "blue_triangle",
    "blue_rectangle",
    "blue_pentagon",
    "blue_star",
    "blue_circle",
    "blue_semicircle",
    "blue_quarter circle",
    "blue_cross",
    "green_triangle",
    "green_rectangle",
    "green_pentagon",
    "green_star",
    "green_circle",
    "green_semicircle",
    "green_quarter circle",
    "green_cross",
    "purple_triangle",
    "purple_rectangle",
    "purple_pentagon",
    "purple_star",
    "purple_circle",
    "purple_semicircle",
    "purple_quarter circle",
    "purple_cross",
    "brown_triangle",
    "brown_rectangle",
    "brown_pentagon",
    "brown_star",
    "brown_circle",
    "brown_semicircle",
    "brown_quarter circle",
    "brown_cross",
    "orange_triangle",
    "orange_rectangle",
    "orange_pentagon",
    "orange_star",
    "orange_circle",
    "orange_semicircle",
    "orange_quarter circle",
    "orange_cross",
]


class ObjectLabel:
    def __init__(
        self,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        imgw: int,
        imgh: int,
        shape: str,
        shape_color: tuple[str, tuple[int, int, int]],
        text_color: tuple[str, tuple[int, int, int]],
    ) -> None:
        self.pt1 = pt1
        self.pt2 = pt2
        self.imgw = imgw
        self.imgh = imgh
        self.shape = shape
        self.shape_color = shape_color
        self.text_color = text_color
        self.name = f"{shape}_{shape_color[0]}"

    def get_normalized_centerx(self) -> float:
        return ((self.pt1[0] + self.pt2[0]) / 2) / self.imgw

    def get_normalized_centery(self) -> float:
        return ((self.pt1[1] + self.pt2[1]) / 2) / self.imgh

    def get_normalized_width(self) -> float:
        return (self.pt2[0] - self.pt1[0]) / self.imgw

    def get_normalized_height(self) -> float:
        return (self.pt2[1] - self.pt1[1]) / self.imgh


if __name__ == '__main__':
    # To generate data.yaml and lookup_table
    a = 0
    for color in colors:
        for shape in shapes:
            print(str(a) + ": " + color[0] + "_" + shape)
            name_color_shape = color[0] + "_" + shape
            lookup_table.append(name_color_shape)
            a += 1
    print(lookup_table)
