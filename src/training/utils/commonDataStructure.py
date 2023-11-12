# colors
yellow = ["yellow", [0, 255, 255]]
white = ["white", [255,255,255]]
black = ["black", [0,0,0]]
red = ["red", [0,0, 255]]
blue = ["blue", [255,0,0]]
green = ["green", [0,128,0]]
purple = ["purple", [128,0,128]]
brown = ["brown", [0, 75, 150]]
orange = ["orange", [0,165,255]]
colors = [yellow, white, black, red, blue, green, purple, brown, orange]
# colors = [yellow, white]

# shapes
shapes = ["triangle", "rectangle","pentagon", "star", "circle", "semicircle", "quarter circle", "cross"]
# shapes = ["triangle", "quarter circle"]

lookup_table = ['yellow_triangle', 'yellow_rectangle', 'yellow_pentagon', 'yellow_star', 'yellow_circle', 'yellow_semicircle', 'yellow_quarter circle', 'yellow_cross', 'white_triangle', 'white_rectangle', 'white_pentagon', 'white_star', 'white_circle', 'white_semicircle', 'white_quarter circle', 'white_cross', 'black_triangle', 'black_rectangle', 'black_pentagon', 'black_star', 'black_circle', 'black_semicircle', 'black_quarter circle', 'black_cross', 'red_triangle', 'red_rectangle', 'red_pentagon', 'red_star', 'red_circle', 'red_semicircle', 'red_quarter circle', 'red_cross', 'blue_triangle', 'blue_rectangle', 'blue_pentagon', 'blue_star', 'blue_circle', 'blue_semicircle', 'blue_quarter circle', 'blue_cross', 'green_triangle', 'green_rectangle', 'green_pentagon', 'green_star', 'green_circle', 'green_semicircle', 'green_quarter circle', 'green_cross', 'purple_triangle', 'purple_rectangle', 'purple_pentagon', 'purple_star', 'purple_circle', 'purple_semicircle', 'purple_quarter circle', 'purple_cross', 'brown_triangle', 'brown_rectangle', 'brown_pentagon', 'brown_star', 'brown_circle', 'brown_semicircle', 'brown_quarter circle', 'brown_cross', 'orange_triangle', 'orange_rectangle', 'orange_pentagon', 'orange_star', 'orange_circle', 'orange_semicircle', 'orange_quarter circle', 'orange_cross']

# # To generate data.yaml and lookup_table
# a = 0
# for color in colors:
#     for shape in shapes:
#         # print(str(a) + ": " + color[0] + "_" + shape)
#         name_color_shape = color[0] + "_" + shape
#         lookup_table.append(name_color_shape)
#         a += 1
# print(lookup_table)
