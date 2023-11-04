import cv2

ft = cv2.freetype.createFreeType2()


def putText(img, text, text_bbox, font_size: float = 1.0):
    """
    Args:
        font_size: percentage of font size
    """
    ft.loadFontData(fontFileName="./src/training/resources/fonts/arial.ttf", idx=0)
    font_height = int((text_bbox[1][1] - text_bbox[0][1]) * font_size)
    width, height = ft.getTextSize(text, font_height, -1)[0]
    print(height)
    ft.putText(
        img=img,
        text=text,
        org=(
            text_bbox[0][0] + int(((text_bbox[1][0] - text_bbox[0][0]) - width) / 2),
            text_bbox[1][1] - int(((text_bbox[1][1] - text_bbox[0][1]) - height) / 2),
        ),
        fontHeight=font_height,
        color=(255, 255, 255),
        thickness=-1,
        line_type=cv2.LINE_AA,
        bottomLeftOrigin=True,
    )

    return img
