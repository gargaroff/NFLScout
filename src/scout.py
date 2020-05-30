#!/usr/bin/env python3
import csv
import ctypes
import re
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import win32gui
from PIL import ImageGrab

# Coordinates where certain areas are located in the picture
POSITION_X_START = 530
POSITION_Y_START = 110
POSITION_X_SIZE = 80
POSITION_Y_SIZE = 35

NAME_X_START = 535
NAME_Y_START = 145
NAME_X_SIZE = 430
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

PROJ_X_START = 1400
PROJ_Y_START = 170
PROJ_X_SIZE = 210
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

COMB_X_START = 1405
COMB_DASH_START = 380
COMB_VERT_START = 425
COMB_CONE_START = 510
COMB_SHUTTLE_START = 550
COMB_BENCH_START = 595
COMB_DASH_SIZE = 50
COMB_VERT_SIZE = 70
COMB_CONE_SIZE = 50
COMB_SHUTTLE_SIZE = 50
COMB_BENCH_SIZE = 30
COMB_Y_SIZE = 30

TALENT_X_START = 1435
TALENT_Y_START = 860
TALENT_X_SIZE = 150
TALENT_Y_SIZE = 60

# Tesseract configs
POSITION_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=FCBDQHPWMRKLGESTO'  # PSM 8: Single word, only letters
NAME_CONFIG = r'--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ\''  # PSM 4: Single column of text of variable sizes
HEIGHT_CONFIG = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789\'\""  # PSM 7: Single line, only numbers and '"
WEIGHT_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # PSM 7: Single line, only numbers
AGE_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # PSM 7: Single line, only numbers
PROJ_TALENT_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist="1234567ADEFHILMNORSTUY "'  # PSM 5: Single uniform block of vertically aligned text, only letters, numbers and space
ARCH_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "'  # PSM 7: Single line, only numbers, only letters and space
SKILL_CONFIG = r'--oem 3 --psm 4 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ. "'  # PSM 4: Single column of text of variable sizes, only letters and space
GRADE_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEF+-'  # PSM 7: Single line, only letters A-F, +, -
COMB_TIME_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'  # PSM 7: Single line, only numbers and .
COMB_VERT_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.\"'  # PSM 7: Single line, only numbers, . and "
COMB_BENCH_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # PSM 7: Single line, only numbers

# Lookup dictionaries
PROJ_TALENT_LOOKUP = {
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
    "P.RECOGNITION":      "PRC",

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

POSITION_LOOKUP = {
    'OB': 'QB',
}

# RegEx patterns for matching OCR'd text
ARCH_PATTERN = r'^(Physical|Man To Man|Field General|Scrambler|Strong Arm|West Coast|Agile|Pass Protector|Power|Run Stopper|Speed Rusher|Pass Coverage|Accurate|Elusive|Receiving|Slot|Zone|Blocking|Utility|Hybrid|Run Support|Deep Threat|Route Runner|Possession|Vertical Threat|Improviser)'
PROJ_TALENT_PATTERN = r'^(EARLY|MID|LATE)([1-7])'

# C struct redefinitions needed to send keys
PUL = ctypes.POINTER(ctypes.c_ulong)

BACKSPACE = 0x0E
DOWN_ARROW = 0x50
SPACE = 0x39


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class InputI(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", InputI)]


# Actual functions to send keys
def key_down(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = InputI()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def key_up(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = InputI()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def press(hex_key_code):
    key_down(hex_key_code)
    time.sleep(.05)
    key_up(hex_key_code)


# Functions to process image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    # Threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV * cv2.THRESH_OTSU)[1]
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def invert(image):
    # Invert the colors of the image, making black white and vice versa
    return cv2.bitwise_not(image)


def dilate(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((2, 2), np.uint8)
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


# Helper
def cleanup_proj_talent(proj_talent):
    if proj_talent == 'UNDRAFTED':
        return proj_talent

    # Sometimes OCR gets 1ST wrong as 18T, etc.
    result = re.match(PROJ_TALENT_PATTERN, proj_talent)
    rnd = result.group(2)

    # Determine the suffix. 1 uses ST, 2 ND, 3 RD, 4+ TH
    if rnd == '1':
        suffix = 'ST'
    elif rnd == '2':
        suffix = 'ND'
    elif rnd == '3':
        suffix = 'RD'
    else:
        suffix = 'TH'

    # Use the correct suffix
    prefix_length = len(result.group(1))
    fixed_proj_talent = '{}{}{}{}'.format(result.group(1), result.group(2), suffix, proj_talent[prefix_length+3:])
    return fixed_proj_talent


# Functions to extract data from image
def extract_player_name(image):
    # Get area of picture which contains the player name
    name_crop = image[NAME_Y_START:NAME_Y_START + NAME_Y_SIZE, NAME_X_START:NAME_X_START + NAME_X_SIZE]

    # Invert the image to have black letters on white background
    name_crop = invert(name_crop)

    # OCR the player name. Will contain \n so replace it with space
    player_name = pytesseract.image_to_string(name_crop, config=NAME_CONFIG)
    player_name = player_name.replace('\n', ' ')
    return player_name.strip()


def extract_player_position(image):
    # Get area of picture which contains the position
    pos_crop = image[POSITION_Y_START:POSITION_Y_START + POSITION_Y_SIZE,
               POSITION_X_START:POSITION_X_START + POSITION_X_SIZE]

    # Invert the image to have black letters on white background
    pos_crop = invert(pos_crop)

    # OCR the position
    position = pytesseract.image_to_string(pos_crop, config=POSITION_CONFIG)

    # Lookup some common OCR errors
    position = position.strip()
    position = POSITION_LOOKUP.get(position, position)
    return position


def extract_player_height(image):
    # Get area of picture which contains the height
    height_crop = image[HEIGHT_Y_START:HEIGHT_Y_START + HEIGHT_Y_SIZE, HEIGHT_X_START:HEIGHT_X_START + HEIGHT_X_SIZE]

    # OCR the height
    height = pytesseract.image_to_string(height_crop, config=HEIGHT_CONFIG)
    height = height.replace('"', '').strip()

    # Height requires a ' as the 2nd character. Make sure it's there
    height = "{}'{}".format(height[0], height[1:]) if height[1] != "'" else height
    return height


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
    proj = pytesseract.image_to_string(proj_crop, config=PROJ_TALENT_CONFIG)

    # Remove all whitespaces and newlines
    proj = proj.replace(' ', '').replace('\n', '')

    # For some reason 'MID 3RD ROUNDER' is not OCR'd
    if proj == '':
        proj = 'MID3RDROUNDER'

    # Cleanup OCR
    proj = cleanup_proj_talent(proj)

    # get value from lookup table
    return PROJ_TALENT_LOOKUP[proj]


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

    # Lookup skills in dictionary or assume unscouted
    skill1 = SKILL_LOOKUP.get(skill1.strip(), '')
    skill2 = SKILL_LOOKUP.get(skill2.strip(), '')
    skill3 = SKILL_LOOKUP.get(skill3.strip(), '')

    return (
        (skill1, grade1 if skill1 != '' else ''),
        (skill2, grade2 if skill2 != '' else ''),
        (skill3, grade3 if skill3 != '' else ''),
    )


def extract_player_talent_round(image):
    # Get area of picture which contains the talent round
    talent_crop = image[TALENT_Y_START:TALENT_Y_START + TALENT_Y_SIZE, TALENT_X_START:TALENT_X_START + TALENT_X_SIZE]

    # OCR the talent round
    talent = pytesseract.image_to_string(talent_crop, config=PROJ_TALENT_CONFIG)

    # Remove all whitespaces and newlines
    talent = talent.replace(' ', '').replace('\n', '')

    # Early return if not fully scouted yet
    if talent == '':
        return ''

    # Cleanup OCR
    talent = cleanup_proj_talent(talent)

    # get value from lookup table
    return PROJ_TALENT_LOOKUP.get(talent, '')


def extract_player_combine_stats(image):
    # Combine stats are pretty small. Erode them to make the lines thicker
    eroded = erode(image)

    # Get areas of the image which contain the combine stats
    dash_crop = eroded[COMB_DASH_START:COMB_DASH_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_DASH_SIZE]
    vertical_crop = eroded[COMB_VERT_START:COMB_VERT_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_VERT_SIZE]
    cone_crop = eroded[COMB_CONE_START:COMB_CONE_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_CONE_SIZE]
    shuttle_crop = eroded[COMB_SHUTTLE_START:COMB_SHUTTLE_START + COMB_Y_SIZE,
                   COMB_X_START:COMB_X_START + COMB_SHUTTLE_SIZE]
    bench_crop = eroded[COMB_BENCH_START:COMB_BENCH_START + COMB_Y_SIZE, COMB_X_START:COMB_X_START + COMB_BENCH_SIZE]

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

    # Return empty string if no combine stats are available
    if dash == '':
        assert dash == vertical and dash == cone and dash == shuttle and dash == bench
        return dash, vertical, cone, shuttle, bench

    # Tesseract can't detect double 7s because they are too close to each other
    dash = '{}77'.format(dash) if len(dash) == 1 else dash
    cone = '{}77'.format(cone) if len(cone) == 1 else cone
    shuttle = '{}77'.format(shuttle) if len(shuttle) == 1 else shuttle

    # Dash, cone and shuttle require a . as the 2nd character, vertical as the 3rd. Make sure it's there
    dash = dash[0] + '.' + dash[1:] if dash[1] != '.' else dash
    vertical = vertical[0:2] + '.' + vertical[2:] if vertical[2] != '.' else vertical
    cone = cone[0] + '.' + cone[1:] if cone[1] != '.' else cone
    shuttle = shuttle[0] + '.' + shuttle[1:] if shuttle[1] != '.' else shuttle

    # Make sure the length for all stats is correct. If not, cut off the last character
    dash = dash[0:-1] if len(dash) > 4 else dash
    vertical = vertical[0:-1] if len(vertical) > 4 else vertical
    cone = cone[0:-1] if len(cone) > 4 else cone
    shuttle = shuttle[0:-1] if len(shuttle) > 4 else shuttle
    bench = bench[0:-1] if len(bench) > 2 else bench

    return dash, vertical, cone, shuttle, bench


def get_madden_window_box():
    # Search for the Madden 20 window
    toplist, winlist = [], []

    def enum_cb(hwnd, _):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(enum_cb, toplist)
    madden = [(hwnd, title) for hwnd, title in winlist if 'madden' in title.lower()]
    # Just grab the first hwnd for the first window matching madden
    madden = madden[0]
    hwnd = madden[0]

    # Make madden the foreground window
    win32gui.SetForegroundWindow(hwnd)
    bbox = win32gui.GetWindowRect(hwnd)
    return hwnd, bbox


def main():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    hwnd, bbox = get_madden_window_box()

    # List of all players
    players = []

    # There are 450 players to scout. Repeat the following process for each of them
    error = None
    for i in range(156):
        # Always make sure that Madden is in the foreground (stupid Origin...)
        win32gui.SetForegroundWindow(hwnd)

        # Open the player card by pressing space
        press(SPACE)

        # Wait 1 second until the screen opens
        time.sleep(1)

        # Take a screenshot
        image = ImageGrab.grab(bbox)

        # Press backspace to go back to the scouting board
        press(BACKSPACE)

        # Transform RGB image to OpenCV compatible BGR image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # image = cv2.imread('test6.png')

        # Preprocess the image
        deskewed = deskew(image)
        gray = get_grayscale(deskewed)
        thresh = thresholding(gray)

        # Perform OCR
        processed = thresh
        try:
            position = extract_player_position(processed)
            name = extract_player_name(processed)
            height = extract_player_height(processed)
            weight = extract_player_weight(processed)
            age = extract_player_age(processed)
            proj = extract_player_proj_round(processed)
            arch = extract_player_archetype(processed)
            skills = extract_player_skills(gray)
            combine = extract_player_combine_stats(processed)
            talent = extract_player_talent_round(processed)
        except RuntimeError as e:
            # Save what we have so far
            error = i
            print(e)
            break

        # Save player
        players.append((position, name, height, weight, age, proj, arch, skills, combine, talent))

        # Go to the next player by pressing down
        # Always make sure that Madden is in the foreground (stupid Origin...)
        win32gui.SetForegroundWindow(hwnd)
        press(DOWN_ARROW)
        time.sleep(0.05)

    # Save all players in CSV
    with open('test.csv', 'w') as f:
        csv_out = csv.writer(f, quoting=csv.QUOTE_ALL)
        for player in players:
            csv_out.writerow(player)

    if error is not None:
        print('ERROR at player #{}'.format(error+1))


if __name__ == '__main__':
    main()
