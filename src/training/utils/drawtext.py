import cv2

ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName="./src/training/resources/fonts/arial.ttf", idx=0)

def draw_text(img, text, text_bbox, text_color = (255, 255, 255), font_size: float = 1.0):
    """
    Args:
        font_size: percentage of font size
    """
    font_height = int((text_bbox[1][1] - text_bbox[0][1]) * font_size)
    width, height = ft.getTextSize(text, font_height, -1)[0]
    img = ft.putText(
        img=img,
        text=text,
        org=(
            text_bbox[0][0] + int(((text_bbox[1][0] - text_bbox[0][0]) - width) / 2),
            text_bbox[1][1] - int(((text_bbox[1][1] - text_bbox[0][1]) - height) / 2),
        ),
        fontHeight=font_height,
        color=text_color,
        thickness=-1,
        line_type=cv2.LINE_AA,
        bottomLeftOrigin=True,
    )
    return img
