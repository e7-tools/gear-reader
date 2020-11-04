import difflib

import cv2
import imutils
import numpy as np
import pytesseract
import regex as re

from config import *
from gears.helpers import import_temp, locate_template, get_stats, stat_evaluator, validate

gear_type_config = "--psm 6"
gear_ability_config = "--psm 7 -c tessedit_char_whitelist= 0123456789+"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_DIR

top = import_temp(TEMPLATES_DIR + 'Top.png')
top_alt = import_temp(TEMPLATES_DIR + 'Top_alt.png')
side = import_temp(TEMPLATES_DIR + 'Side.png')
mid = import_temp(TEMPLATES_DIR + 'Middle.png')
bottom = import_temp(TEMPLATES_DIR + 'Bottom.png')
plus = import_temp(TEMPLATES_DIR + 'Pluslvl.png')
ilvl = import_temp(TEMPLATES_DIR + 'Itemlvl.png')
itype = import_temp(TEMPLATES_DIR + 'Type.png')
indi = import_temp(TEMPLATES_DIR + 'indicator.png')


# Process
# 1. Read and Save frame of Gear from video
# 2. Find Area of interest ( Gear box) in an image
# 3. Find side and middle template and separate gear box into 2 parts
# 4. Read text from each part.
# 5: Reformat values to json, add fake values if missing
# 6: Export json

###--------
def get_approx_xy(approx):
    app_loc = {'x': [], 'y': []}
    for loc in approx:
        app_loc['y'].append(loc[0][1])
        app_loc['x'].append(loc[0][0])
    return app_loc


def find_item_box(image):
    """
    Locate gear box using contours
    :param image:
    :return:
    """
    height, width, _ = image.shape
    # Transform image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]

    # Finding Contours
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Contour approximation
    all_approx = []
    for cnt in contours:
        e = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, e, True)
        app_loc = get_approx_xy(approx)

        # Filter out relevant contour
        if max(app_loc['y']) - min(app_loc['y']) > height / 2 and \
                min(app_loc['x']) > width / 4 and \
                max(app_loc['x']) < width * 3 / 4:
            all_approx.append(approx)

    # Find largest contour
    if len(all_approx) > 0:
        best = max(all_approx, key=lambda x: cv2.arcLength(x, True))
    else:
        raise Exception

    # Draw a rectangle around the contour
    x, y, w, h = cv2.boundingRect(best)
    x_start, y_start, x_end, y_end = x, y, x + w, y + h

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)  # red

    # Gear box location
    gear_box = gray[y_start:y_end, x_start:x_end]
    gear_box_col = image[y_start:y_end, x_start:x_end]

    global indi_y_start, indi_y_end
    indi_y_start = y_start
    indi_y_end = y_start + 200

    return gear_box, gear_box_col


def get_gearbox(image):
    """
    Locate gear box in the frame
    :param image:
    :return: gear box in grayscale and color
    """
    global indi_y_start, indi_y_end
    # Middle of the image
    x_middle = int(image.shape[1] / 2)
    y_middle = int(image.shape[0] / 2)

    # Find location of the indicator
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray_image = imutils.resize(gray_image, height=1440)

    (_, loc_max, r) = locate_template(indi, gray_image[:y_middle, ])
    indi_x_start = int(loc_max[0] * r)
    indi_y_start = int(loc_max[1] * r)
    indi_y_end = int(indi_y_start + indi.shape[0] * r)
    indi_x_end = int(indi_x_start + indi.shape[1] * r)

    # Since the gear is located in the middle.
    # I use location of the indicator in the corner to calculate the box for gear.
    x_start = x_middle - (indi_x_end - x_middle)
    y_start = indi_y_start
    y_end = y_middle + (y_middle - indi_y_start)
    x_end = indi_x_end

    # Gear box location
    gear_box = gray_image[y_start:y_end, x_start:x_end]
    gear_box_col = image[y_start:y_end, x_start:x_end]

    return gear_box, gear_box_col


def split_gearbox(gear_box, gear_box_col):
    """
    Divide gear box into smaller parts
    :param gear_box:
    :param gear_box_col:
    :return:
    """
    # Find side template in the gear box
    (_, loc_max, r) = locate_template(side, gear_box)
    side_x_start = int(loc_max[0] * r)
    side_y_start = int(loc_max[1] * r)
    side_y_end = int(side_y_start + side.shape[0] * r)
    side_x_end = int(side_x_start + side.shape[1] * r)

    # Find mid template in the gear box to the left of the side
    (_, loc_max, r) = locate_template(mid, gear_box[:, :side_x_end])
    mid_x_start = int(loc_max[0] * r)
    mid_y_start = int(loc_max[1] * r)
    mid_y_end = int(mid_y_start + mid.shape[0] * r)
    mid_x_end = int(mid_x_start + mid.shape[1] * r)

    ## Combine all locations
    # Top
    gear_top = gear_box[:mid_y_start, :side_x_start]
    gear_top_col = gear_box_col[:mid_y_start, :side_x_start]

    # Bottom
    gear_bot = gear_box[mid_y_end:, :side_x_start]
    return gear_bot, gear_top, gear_top_col


def read_stats(gear_bot, gear_info, errors):
    # Process image
    gear_bot = cv2.resize(gear_bot, (0, 0), fx=3, fy=3)
    ret, item_stat = cv2.threshold(gear_bot, 40, 255, cv2.THRESH_BINARY_INV)
    item_stat = cv2.medianBlur(item_stat, 5)

    # Read text
    raw_value = pytesseract.image_to_string(item_stat)

    # Clean results and update gear info
    try:
        stat_value = get_stats(raw_value)
    except:
        try:
            print("Trying other method for stats")
            raw_value = pytesseract.image_to_string(item_stat, config="--psm 6")
            stat_value = get_stats(raw_value)
        except:
            print("Error: getting stats")
            stat_value = {}
            errors['stats'] = 1

    gear_info.update(stat_value)
    return item_stat, gear_info, errors, raw_value


def read_ilevel(gear_top, gear_info, errors, low_threshold=170):
    # Find ilvl in top part
    (_, loc_max, r) = locate_template(ilvl, gear_top)
    (x_start, y_start) = (int((loc_max[0] + 17) * r), int((loc_max[1] + 17) * r))
    (x_end, y_end) = (int((loc_max[0] + plus.shape[1]) * r), int((loc_max[1] + plus.shape[0]) * r))
    gear_top_ed = gear_top[y_start:y_end, x_start:x_end]

    # Process img
    gear_top_ed = cv2.resize(gear_top_ed, (0, 0), fx=3, fy=3)
    _, gear_top_ed = cv2.threshold(gear_top_ed, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    gear_top_ed = cv2.medianBlur(gear_top_ed, 1)
    gear_top_ed = cv2.cvtColor(gear_top_ed, cv2.COLOR_GRAY2RGB)

    # Get text
    raw_value = pytesseract.image_to_string(gear_top_ed, config="--psm 13")

    try:
        value = re.search(r".*(\d{2,2}).*", raw_value).group(1)
        gear_info.update({"level": value})
    except:
        print("Error in item level.")
        errors['level'] = 1
    return gear_top_ed, gear_info, errors, raw_value


def read_ability(gear_top, gear_info, errors, low_threshold=190):
    global ability_found, ability_x_end

    # Get ability icon location
    (_, loc_max, r) = locate_template(plus, gear_top[:(indi_y_end - indi_y_start), :])
    (ability_x_start, ability_y_start) = (int(loc_max[0] * r), int(loc_max[1] * r))
    (ability_x_end, ability_y_end) = (int((loc_max[0] + plus.shape[1]) * r), int((loc_max[1] + plus.shape[0]) * r))
    gear_ability = gear_top[ability_y_start:ability_y_end, ability_x_start:ability_x_end]

    # Process image
    gear_ability = cv2.resize(gear_ability, (0, 0), fx=5, fy=5)
    (thresh, gear_ability) = cv2.threshold(gear_ability, low_threshold, 255, cv2.THRESH_BINARY)
    gear_ability = cv2.medianBlur(gear_ability, 5)
    gear_ability = cv2.bitwise_not(gear_ability)

    # Get text
    raw_value = pytesseract.image_to_string(gear_ability, config=gear_ability_config)

    if re.search(r".*?(\d{1,2}).*", raw_value) is None:
        value = "0"
        ability_found = False
    else:
        if re.search(r".*?(\d{3,3}).*", raw_value) is None:
            value = re.search(r".*?(\d{1,2}).*", raw_value).group(1)
        else:
            value = re.search(r".*?(\d{3,3}).*", raw_value).group(1)[1:3]
        ability_found = True

    gear_info.update({"ability": value})
    return gear_ability, gear_info, errors, raw_value


def clean_gear_name(text):
    for ch in ['\\', '`', '*', '_', '{', '}', '[', ']', ',', ';', '"', '/', '|', '»',
               '(', ')', '>', '#', '+', '-', '.', '!', '$', '\n', '~', '‘']:
        if ch in text:
            text = text.replace(ch, " ")
    text = re.sub(r' {2,}', " ", text)
    return text


def read_gear_name(gear_top_col, gear_info, errors):
    (gear_top_x, gear_top_y) = gear_top_col.shape[:2]
    name_y_end = int(gear_top_y * 0.6)
    name_x_start = int(gear_top_x * 0.55)

    if ability_found:
        gear_name = gear_top_col[0:name_y_end, (ability_x_end - 5):]
    else:
        gear_name = gear_top_col[0:name_y_end, name_x_start:]

    hsv = cv2.cvtColor(gear_name, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    gear_name = cv2.bitwise_and(gear_name, gear_name, mask=mask)
    gear_name = cv2.cvtColor(gear_name, cv2.COLOR_BGR2GRAY)
    (thresh, gear_name) = cv2.threshold(gear_name, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # gear_name = cv2.medianBlur(gear_name, 3)
    gear_name = cv2.bitwise_not(gear_name)

    raw_value = pytesseract.image_to_string(gear_name, config=gear_type_config)

    value = clean_gear_name(raw_value)
    gear_info.update({'name': value.strip()})
    return gear_name, gear_info, errors, raw_value


def read_rarity(gear_top_col, gear_info, errors):
    # Locate gear rarity and slot
    low_threshold = 170
    (gear_top_x, gear_top_y) = gear_top_col.shape[:2]
    type_y_end = int(gear_top_y * 0.25)
    type_x_start = int(gear_top_x * 0.55)

    if ability_found:
        gear_type = gear_top_col[0:type_y_end, (ability_x_end - 5):]
    else:
        gear_type = gear_top_col[0:type_y_end, type_x_start:]

    ## Process img
    # Find red and purple color in img
    gear_type = cv2.resize(gear_type, (0, 0), fx=5, fy=5)
    type_hsv = cv2.cvtColor(gear_type, cv2.COLOR_BGR2HSV)

    red_lower = np.array([0, 70, 50])
    red_upper = np.array([10, 255, 255])

    purple_lower = np.array([140, 70, 50])
    purple_upper = np.array([179, 255, 255])

    blue_lower = np.array([90, 50, 0])
    blue_upper = np.array([140, 255, 255])

    mask1 = cv2.inRange(type_hsv, red_lower, red_upper)
    mask2 = cv2.inRange(type_hsv, purple_lower, purple_upper)
    mask3 = cv2.inRange(type_hsv, blue_lower, blue_upper)
    mask12 = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask3, mask12)
    gear_type = cv2.bitwise_and(gear_type, gear_type, mask=mask)

    gear_type = cv2.cvtColor(gear_type, cv2.COLOR_BGR2GRAY)
    (thresh, gear_type) = cv2.threshold(gear_type, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gear_type = cv2.medianBlur(gear_type, 3)
    gear_type = cv2.bitwise_not(gear_type)

    # Get text
    raw_value = pytesseract.image_to_string(gear_type, config=gear_type_config)

    # Find rarity
    rarity_found = re.search(r".*(Rare|Epic|Heroic|Good|Normal).*", raw_value)

    try:
        if rarity_found:
            rarity = rarity_found.group(1)
        else:
            print("Getting closest match for rarity")
            rarity = difflib.get_close_matches(raw_value.rsplit(" ", 1)[0],
                                               ['Rare', 'Epic', 'Good', 'Normal', 'Heroic'],
                                               cutoff=0.5, n=1)[0]
        gear_info.update({"rarity": rarity})
    except:
        print("error in gear rarity")
        errors['rarity'] = 1

    # Find slot
    try:
        slot = re.search(r".*(Weapon|Helmet|Armor|Ring|Necklace|Boots).*", raw_value).group(1)
        gear_info.update({"slot": slot})
    except:
        print("error in gear slot")
        errors['slot'] = 1
    return gear_type, gear_info, errors, raw_value


def value_extractor(input_file, debug=False):
    """
    Read the exported frames of equipment from video.
    Find Area of interest ( Gear box) in a frame then
    find side and middle templates to separate gear box into top and bottom parts.
    Read text from each part. Review text and reformat.
    :param input_file: jpg image
    :param debug: change to True to show image parts
    :return:
    """

    errors = {}
    # Read frame
    image = cv2.imread(input_file)

    """
    Find gear box in the frame
    """
    gear_box, gear_box_col = find_item_box(image)

    # Save gear box
    if not debug:
        cv2.imwrite(input_file.replace(".jpg", "-box.jpg"), gear_box_col)
        pass

    """
    Divide gear box into smaller parts
    """

    gear_bot, gear_top, gear_top_col = split_gearbox(gear_box, gear_box_col)

    if debug:
        cv2.imshow("gear col", imutils.resize(gear_box_col, width=360))

    ####
    gear_info = {}

    ## Add get hero here ##

    """
    Retrieving gear stats
    """

    ## Getting stats from bottom part
    gear_bot_edited, gear_info, errors, raw_stats = read_stats(gear_bot, gear_info, errors)

    if debug:
        cv2.imshow("stat", imutils.resize(gear_bot_edited, width=360))
        print("----- Stats raw:", raw_stats)

    """
    Retrieving gear level
    """
    gear_top_ilvl, gear_info, errors, raw_ilevel = read_ilevel(gear_top, gear_info, errors)

    if debug:
        cv2.imshow('Item Level After Processing', gear_top_ilvl)
        print('----- Item level raw: ', raw_ilevel)

    """
    Retrieving gear ability
    """
    gear_ability, gear_info, errors, raw_ability = read_ability(gear_top, gear_info, errors)

    if debug:
        cv2.imshow('Item Enhance After Processing', gear_ability)
        print('----- Raw ability:', raw_ability)

    """
    Retrieving gear rarity and slot
    """
    gear_type, gear_info, errors, raw_value = read_rarity(gear_top_col, gear_info, errors)
    if debug:
        print('----- Gear type raw:', raw_value)
        cv2.imshow('Item Type After Processing', imutils.resize(gear_type, width=360))

    gear_name, _, _, raw_value1 = read_gear_name(gear_top_col, gear_info, errors)

    if debug:
        print('----- Gear name raw:', raw_value1)
        cv2.imshow('Item Name After Processing', imutils.resize(gear_name, width=360))

    """
    Review gear info and rating
    """
    # Validate gear info
    gear_info, errors = validate(gear_info, errors)

    # Compute rating
    gear_rating = stat_evaluator(gear_info)
    gear_info.update({'rating': gear_rating})

    """
    Transform gear info dictionary to other formats
    """
    # This is for the web app
    # for key, value in gear_info.items():
    # if "Stat" in key:
    #         gear_info.update({key: json.dumps(value)})

    # This is for the program
    for key in list(gear_info):
        if "Stat" in key:
            value = gear_info[key]
            if "main" in key:
                gear_info.update({"main": value[0]})
                gear_info.update({"value": int(value[1])})
            else:
                gear_info.update({value[0]: int(value[1])})
            del gear_info[key]

    if debug:
        print(gear_info, '\n')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gear_info, errors
