#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


NAME_X_START = 539
NAME_Y_START = 150
NAME_X_SIZE = 610
NAME_Y_SIZE = 127


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
    name_crop = image[NAME_Y_START:NAME_Y_START+NAME_Y_SIZE, NAME_X_START:NAME_X_START+NAME_X_SIZE]
    cv2.imshow("name", name_crop)
    cv2.waitKey(0)


def main():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    image = cv2.imread('test.png')
    dskw = deskew(image)
    gray = get_grayscale(dskw)
    thresh = thresholding(gray)
    rnoise = remove_noise(gray)
    dlt = dilate(gray)
    erd = erode(gray)
    opn = opening(gray)
    cny = canny(gray)

    custom_config = r'--oem 3 --psm 6'
    dskw_str = pytesseract.image_to_string(dskw, config=custom_config)
    gray_str = pytesseract.image_to_string(gray, config=custom_config)
    thresh_str = pytesseract.image_to_string(thresh, config=custom_config)
    rnoise_str = pytesseract.image_to_string(rnoise, config=custom_config)
    dlt_str = pytesseract.image_to_string(dlt, config=custom_config)
    erd_str = pytesseract.image_to_string(erd, config=custom_config)
    opn_str = pytesseract.image_to_string(opn, config=custom_config)
    cny_str = pytesseract.image_to_string(cny, config=custom_config)

    # For player name: deskew, thresh, image
    show_images([gray, rnoise, dlt, erd, thresh, dskw, opn, cny], 3,
                ["gray", "rnoise", "dilate", "erode", "thresh", "deskew", "opening", "canny"])


if __name__ == '__main__':
    main()
