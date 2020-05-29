#!/usr/bin/env python3
import re

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

AGE_X_START = 895
AGE_Y_START = 465
AGE_X_SIZE = 35
AGE_Y_SIZE = 35

PROJ_X_START = 1410
PROJ_Y_START = 170
PROJ_X_SIZE = 200
PROJ_Y_SIZE = 80

ARCH_X_START = 895
ARCH_Y_START = 545
ARCH_X_SIZE = 200
ARCH_Y_SIZE = 40

SKILL1_X_START = 745
SKILL2_X_START = 960
SKILL3_X_START = 1175
SKILL_Y_START = 855
SKILL_X_SIZE = 210
SKILL_Y_SIZE = 70

GRADE1_X_START = 810
GRADE2_X_START = 1025
GRADE3_X_START = 1235
GRADE_Y_START = 765
GRADE_X_SIZE = 85
GRADE_Y_SIZE = 65

COMB_X_START = 1400
COMB_DASH_START = 385
COMB_VERT_START = 425
COMB_CONE_START = 510
COMB_SHUTTLE_START = 550
COMB_BENCH_START = 590
COMB_DASH_SIZE = 55
COMB_VERT_SIZE = 70
COMB_CONE_SIZE = 55
COMB_SHUTTLE_SIZE = 55
COMB_BENCH_SIZE = 35
COMB_Y_SIZE = 30

# Tesseract configs
POSITION_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # PSM 8: Single word, only letters
NAME_CONFIG = r'--oem 3 --psm 4'  # PSM 4: Single column of text of variable sizes
HEIGHT_CONFIG = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789\'\""  # PSM 7: Single line, only numbers and '"
WEIGHT_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # PSM 7: Single line, only numbers
AGE_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # PSM 7: Single line, only numbers
PROJ_CONFIG = r'--oem 3 --psm 4 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "'  # PSM 5: Single uniform block of vertically aligned text, only letters, numbers and space
ARCH_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "'  # PSM 7: Single line, only numbers, only letters and space
SKILL_CONFIG = r'--oem 3 --psm 4 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ. "'  # PSM 4: Single column of text of variable sizes, only letters and space
GRADE_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEF+-'  # PSM 7: Single line, only letters A-F, +, -
COMB_TIME_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'  # PSM 7: Single line, only numbers and .
COMB_VERT_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.\"'  # PSM 7: Single line, only numbers, . and "
COMB_BENCH_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # PSM 7: Single line, only numbers

# Lookup dictionaries
PROJ_LOOKUP = {
    "EARLY1STROUNDER": "1E",
    "MID1STROUNDER":   "1M",
    "LATE1STROUNDER":  "1L",
    "EARLY2NDROUNDER": "2E",
    "MID2NDROUNDER":   "2M",
    "LATE2NDROUNDER":  "2L",
    "EARLY3RDROUNDER": "3E",
    "MID3RDROUNDER":   "3M",
    "LATE3RDROUNDER":  "3L",
    "EARLY4THROUNDER": "4E",
    "MID4THROUNDER":   "4M",
    "LATE4THROUNDER":  "4L",
    "EARLY5THROUNDER": "5E",
    "MID5THROUNDER":   "5M",
    "LATE5THROUNDER":  "5L",
    "EARLY6THROUNDER": "6E",
    "MID6THROUNDER":   "6M",
    "LATE6THROUNDER":  "6L",
    "EARLY7THROUNDER": "7E",
    "MID7THROUNDER":   "7M",
    "LATE7THROUNDER":  "7L",
    "UNDRAFTED":       "UD",
}

ARCH_LOOKUP = {
    "Man To Man":   "Man-to-Man",
    "Physical":     "Red Zone Threat",
    "Route Runner": "Possession",
}

SKILL_LOOKUP = {
    # Physical
    "ACCELERATION":       "ACC",
    "AGILITY":            "AGI",
    "JUMP":               "JMP",
    "SPEED":              "SPD",
    "STRENGTH":           "STR",

    # Mental
    "AWARENESS":          "AWR",
    "PLAYRECOGNITION":    "PRC",

    # Kicking
    "KICKACCURARY":       "KAC",
    "KICKPOWER":          "KPW",

    # Carrying
    "BCVISION":           "BCV",
    "BREAKTACKLE":        "BTK",
    "CARRYING":           "CAR",
    "ELUSIVENESS":        "ELU",
    "JUKEMOVE":           "JKM",
    "STIFFARM":           "SFA",
    "SPINMOVE":           "SPM",
    "TRUCKING":           "TRK",

    # Receiving
    "CTHINTRAFFIC":       "CIT",
    "CATCHING":           "CTH",
    "DEEPROUTE":          "DRT",
    "MEDROUTE":           "MRT",
    "SHORTROUTE":         "SRT",
    "RELEASE":            "RLS",
    "SPECCATCH":          "SPC",

    # Throwing
    "BREAKSACK":          "BSK",
    "THROWACCSHORT":      "SAC",
    "THROWACCMID":        "MAC",
    "THROWACCDEEP":       "DAC",
    "PLAYACTION":         "PAC",
    "THROWPOWER":         "THP",
    "THROWONTHERUN":      "TOR",
    "THROWUNDERPRESSURE": "TUP",

    # Blocking
    "IMPACTBLOCK":        "IBK",
    "LEADBLOCK":          "LBK",
    "PASSBLOCK":          "PBK",
    "PASSBLOCKFINESSE":   "PBF",
    "PASSBLOCKPOWER":     "PBP",
    "RUNBLOCK":           "RBK",
    "RUNBLOCKFINESSE":    "RBF",
    "RUNBLOCKPOWER":      "RBP",

    # Defensive
    "BLOCKSHEDDING":      "BSH",
    "FMOVES":             "FMV",
    "M.COVERAGE":         "MCV",
    "PWRMOVES":           "PMV",
    "HITPOWER":           "POW",
    "PRESS":              "PRS",
    "PURSUIT":            "PUR",
    "TACKLE":             "TAK",
    "Z.COVERAGE":         "ZCV",

    # Misc
    "INJURY":             "INJ",
    "KICKRETURN":         "KRT",
    "STAMINA":            "STA",
    "TOUGHNESS":          "TGH",
}

# RegEx patterns for matching OCR'd text
ARCH_PATTERN = r'^(Physical|Man To Man|Field General|Scrambler|Strong Arm|West Coast|Agile|Pass Protector|Power|Run Stopper|Speed Rusher|Pass Coverage|Accurate|Elusive|Receiving|Slot|Zone|Blocking|Utility|Hybrid|Run Support|Deep Threat|Route Runner)'


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    # Threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def invert(image):
    # Invert the colors of the image, making black white and vice versa
    return cv2.bitwise_not(image)


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


def extract_player_age(image):
    # Get area of picture which contains the age
    age_crop = image[AGE_Y_START:AGE_Y_START + AGE_Y_SIZE, AGE_X_START:AGE_X_START + AGE_X_SIZE]

    # OCR the age
    age = pytesseract.image_to_string(age_crop, config=AGE_CONFIG)
    return age.strip()


def extract_player_proj_round(image):
    # Get area of picture which contains the age
    proj_crop = image[PROJ_Y_START:PROJ_Y_START + PROJ_Y_SIZE, PROJ_X_START:PROJ_X_START + PROJ_X_SIZE]

    # OCR the age
    proj = pytesseract.image_to_string(proj_crop, config=PROJ_CONFIG)

    # Remove all whitespaces and newlines, then get value from lookup table
    proj = proj.replace(' ', '').replace('\n', '')
    return PROJ_LOOKUP[proj]


def extract_player_archetype(image):
    # Get area of picture which contains the archetype
    arch_crop = image[ARCH_Y_START:ARCH_Y_START + ARCH_Y_SIZE, ARCH_X_START:ARCH_X_START + ARCH_X_SIZE]

    # OCR the archetype
    arch = pytesseract.image_to_string(arch_crop, config=ARCH_CONFIG)

    # Remove all whitespaces and newlines, then get value from lookup table
    arch = arch.strip()
    result = re.match(ARCH_PATTERN, arch)
    arch = result.group(1)
    return ARCH_LOOKUP.get(arch, arch)


def extract_player_skills(image):
    # Get area of picture which contains the skills and the grades
    skill1_crop = image[SKILL_Y_START:SKILL_Y_START + SKILL_Y_SIZE, SKILL1_X_START:SKILL1_X_START + SKILL_X_SIZE]
    skill2_crop = image[SKILL_Y_START:SKILL_Y_START + SKILL_Y_SIZE, SKILL2_X_START:SKILL2_X_START + SKILL_X_SIZE]
    skill3_crop = image[SKILL_Y_START:SKILL_Y_START + SKILL_Y_SIZE, SKILL3_X_START:SKILL3_X_START + SKILL_X_SIZE]

    # Colors on the grade image are inverted. Invert it
    inverted = invert(image)
    grade1_crop = inverted[GRADE_Y_START:GRADE_Y_START + GRADE_Y_SIZE, GRADE1_X_START:GRADE1_X_START + GRADE_X_SIZE]
    grade2_crop = inverted[GRADE_Y_START:GRADE_Y_START + GRADE_Y_SIZE, GRADE2_X_START:GRADE2_X_START + GRADE_X_SIZE]
    grade3_crop = inverted[GRADE_Y_START:GRADE_Y_START + GRADE_Y_SIZE, GRADE3_X_START:GRADE3_X_START + GRADE_X_SIZE]

    # OCR the skill names and grades
    skill1 = pytesseract.image_to_string(skill1_crop, config=SKILL_CONFIG)
    skill2 = pytesseract.image_to_string(skill2_crop, config=SKILL_CONFIG)
    skill3 = pytesseract.image_to_string(skill3_crop, config=SKILL_CONFIG)

    grade1 = pytesseract.image_to_string(grade1_crop, config=GRADE_CONFIG)
    grade2 = pytesseract.image_to_string(grade2_crop, config=GRADE_CONFIG)
    grade3 = pytesseract.image_to_string(grade3_crop, config=GRADE_CONFIG)

    # Remove all whitespaces and newlines
    skill1 = skill1.replace(' ', '').replace('\n', '')
    skill2 = skill2.replace(' ', '').replace('\n', '')
    skill3 = skill3.replace(' ', '').replace('\n', '')

    grade1 = grade1.strip()
    grade2 = grade2.strip()
    grade3 = grade3.strip()

    # Lookup skills in dictionary
    skill1 = SKILL_LOOKUP[skill1.strip()]
    skill2 = SKILL_LOOKUP[skill2.strip()]
    skill3 = SKILL_LOOKUP[skill3.strip()]

    return {
        skill1: grade1,
        skill2: grade2,
        skill3: grade3,
    }


def extract_player_combine_stats(image):
    # Get areas of the image which contain the combine stats
    dash_crop = image[COMB_DASH_START:COMB_DASH_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_DASH_SIZE]
    vertical_crop = image[COMB_VERT_START:COMB_VERT_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_VERT_SIZE]
    cone_crop = image[COMB_CONE_START:COMB_CONE_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_CONE_SIZE]
    shuttle_crop = image[COMB_SHUTTLE_START:COMB_SHUTTLE_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_SHUTTLE_SIZE]
    bench_crop = image[COMB_BENCH_START:COMB_BENCH_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_BENCH_SIZE]

    # OCR the stats
    dash = pytesseract.image_to_string(dash_crop, config=COMB_TIME_CONFIG)
    vertical = pytesseract.image_to_string(vertical_crop, config=COMB_VERT_CONFIG)
    cone = pytesseract.image_to_string(cone_crop, config=COMB_TIME_CONFIG)
    shuttle = pytesseract.image_to_string(shuttle_crop, config=COMB_TIME_CONFIG)
    bench = pytesseract.image_to_string(bench_crop, config=COMB_BENCH_CONFIG)

    # Clean up results a bit
    dash = dash.strip()
    vertical = vertical.replace('"', '').strip()
    cone = cone.strip()
    shuttle = shuttle.strip()
    bench = bench.strip()

    # Dash, cone and shuttle require a . as the 2nd character, vertical as the 3rd. Make sure it's there
    dash = dash[0] + '.' + dash[1:] if dash[1] != '.' else dash
    vertical = vertical[0:2] + '.' + vertical[2:] if vertical[2] != '.' else vertical
    cone = cone[0] + '.' + cone[1:] if cone[1] != '.' else cone
    shuttle = shuttle[0] + '.' + shuttle[1:] if shuttle[1] != '.' else shuttle

    return dash, vertical, cone, shuttle, bench


def main():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image = cv2.imread('test3.png')

    # Preprocess the image
    dskw = deskew(image)
    gray = get_grayscale(dskw)
    thresh = thresholding(gray)

    extract_player_position(thresh)
    extract_player_name(thresh)
    extract_player_height(thresh)
    extract_player_weight(thresh)
    extract_player_age(thresh)
    extract_player_proj_round(thresh)
    extract_player_archetype(thresh)
    extract_player_skills(thresh)
    extract_player_combine_stats(thresh)


if __name__ == '__main__':
    main()
