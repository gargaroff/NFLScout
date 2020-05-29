#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

# Coordinates where certain areas are located in the picture
POSITION_X_START = 530
POSITION_Y_START = 110
POSITION_X_SIZE = 110
POSITION_Y_SIZE = 35

NAME_X_START = 535
NAME_Y_START = 145
NAME_X_SIZE = 620
NAME_Y_SIZE = 137

HEIGHT_X_START = 900
HEIGHT_Y_START = 380
HEIGHT_X_SIZE = 50
HEIGHT_Y_SIZE = 30

WEIGHT_X_START = 895
WEIGHT_Y_START = 425
WEIGHT_X_SIZE = 50
WEIGHT_Y_SIZE = 30

# Tesseract configs
POSITION_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # PSM 8: Single word, only letters
NAME_CONFIG = r'--oem 3 --psm 4'  # PSM 4: Single column of text of variable sizes
HEIGHT_CONFIG = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789\'\""  # PSM 7: Single line, only numbers and '"
WEIGHT_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # PSM 7: Single line, only numbers


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    # Threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def show_images(images, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()

        plt.imshow(image)
        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def extract_player_name(image):
    # Get area of picture which contains the player name
    name_crop = image[NAME_Y_START:NAME_Y_START + NAME_Y_SIZE, NAME_X_START:NAME_X_START + NAME_X_SIZE]

    # OCR the player name. Will contain \n so replace it with space
    player_name = pytesseract.image_to_string(name_crop, config=NAME_CONFIG)
    player_name = player_name.replace('\n', ' ')
    return player_name.strip()


def extract_player_position(image):
    # Get area of picture which contains the position
    pos_crop = image[POSITION_Y_START:POSITION_Y_START + POSITION_Y_SIZE,
                     POSITION_X_START:POSITION_X_START + POSITION_X_SIZE]

    # OCR the position
    position = pytesseract.image_to_string(pos_crop, config=POSITION_CONFIG)
    return position.strip()


def extract_player_height(image):
    # Get area of picture which contains the height
    height_crop = image[HEIGHT_Y_START:HEIGHT_Y_START + HEIGHT_Y_SIZE, HEIGHT_X_START:HEIGHT_X_START + HEIGHT_X_SIZE]

    # OCR the height
    height = pytesseract.image_to_string(height_crop, config=HEIGHT_CONFIG)
    return height.strip()


def extract_player_weight(image):
    # Get area of picture which contains the weight
    weight_crop = image[WEIGHT_Y_START:WEIGHT_Y_START + WEIGHT_Y_SIZE, WEIGHT_X_START:WEIGHT_X_START + WEIGHT_X_SIZE]

    # OCR the weight
    weight = pytesseract.image_to_string(weight_crop, config=WEIGHT_CONFIG)
    return weight.strip()


def main():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image = cv2.imread('test.png')

    # Preprocess the image
    dskw = deskew(image)
    gray = get_grayscale(dskw)
    thresh = thresholding(gray)

    extract_player_position(thresh)
    extract_player_name(thresh)
    extract_player_height(thresh)
    extract_player_weight(thresh)


if __name__ == '__main__':
    main()
