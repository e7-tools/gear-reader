import difflib
import json
import random
import string

import cv2
import imutils
import numpy as np
import pandas as pd
import regex as re

sets_name = ['Speed', 'Critical', 'Hit', 'Attack', 'Defense', 'Health',
             'Destruction', 'Immunity', 'Lifesteal', 'Rage', 'Resist', 'Unity', 'Counter']

stats_name = ["HP", "HPP", "Def", "DefP", "Atk", "AtkP", "Spd", "CChance", "CDmg", "Res", "Eff"]

slots_name = ['Weapon', 'Helmet', 'Armor', 'Necklace', 'Ring', 'Boots']
# List of common OCR error to change
change_lst = {"§": "5"}


###### Find Template ############

def import_temp(file):
    """
    Read image file and convert it to Canny for template matching.
    """
    temp = cv2.imread(file)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = cv2.Canny(temp, 10, 25)
    return temp


def locate_template(template, img):
    """
    Find location of a template in an image
    :param template: take processed template as input
    :param img: image to perform the search in
    :return: location of the template
    """
    temp_found = None
    (height, width) = template.shape[:2]

    for scale in np.linspace(0.1, 3, 10)[::-1]:
        # resize the image and store the ratio
        resized_img = imutils.resize(img, width=int(img.shape[1] * scale))
        ratio = img.shape[1] / float(resized_img.shape[1])
        if resized_img.shape[0] < height or resized_img.shape[1] < width:
            break
        # Convert to edged image for checking
        e = cv2.Canny(resized_img, 10, 25)
        match = cv2.matchTemplate(e, template, cv2.TM_CCOEFF)
        (_, val_max, _, loc_max) = cv2.minMaxLoc(match)
        if temp_found is None or val_max > temp_found[0]:
            temp_found = (val_max, loc_max, ratio)
    return temp_found


def rename_stat(stat):
    """
    Rename stat to JSON compatible format
    """
    rename_lst = [("Health", "HP"), ("Defense", "Def"), ("Attack", "Atk"),
                  ("Speed", "Spd"), ("Critical Hit Chance", "CChance"),
                  ("Critical Hit Damage", "CDmg"), ("Effect Resistance", "Res"),
                  ("Effectiveness", "Eff")]
    for old, new in rename_lst:
        if old in stat:
            stat = new
    return stat


# ------------------------------

def match_gear_name(name, data):
    name_lst = data['name'].tolist()
    gear_name = difflib.get_close_matches(name, name_lst, n=1, cutoff=0.1)[0]
    print('Raw name', name, '\nFound name', gear_name)
    idx = name_lst.index(gear_name)
    return data['level'][idx]


def validate(gear_info, errors):
    """
    Check to see if any stat values, ability, level is too high and add misisng data
    :param gear_info: dict from value_extractor function
    :param errors: dict
    :return:
    """

    checklist = [("set", "NA"), ("rarity", "NA"), ("slot", "NA"),
                 ("level", "999"), ("ability", "999"), ("mainStat", ["NA", 999])]

    for key, val in checklist:
        # Add fake data if not found
        if key not in gear_info:
            gear_info[key] = val

            if "mainStat" in key:
                errors['stats'] = 1
            else:
                errors[key] = 1
            print(f"{key} is not in info.")

    for key, val in gear_info.items():
        # Check stats values if values are too high then change to 999
        if "Stat" in key:
            if val[0][-1] == "P" and len(val[0]) > 2 and val[1] > 100:
                val[1] = 0
                gear_info[key] = val
                errors['stats'] = 1
                print("changed stat")

        # Check other values in gear info
        if key == 'mainStat' and val[0] == "NA":
            if gear_info['slot'] == "Weapon":
                gear_info = "Atk"
            elif gear_info['slot'] == "Helmet":
                gear_info = "HP"
            elif gear_info['slot'] == "Armor":
                gear_info = "Def"

        elif key == "level" and val == "999" and len(gear_info['name']) > 1:
            matched = match_gear_name(gear_info['name'])
            if matched > 0:
                print(matched)
                gear_info[key] = str(int(matched))
                errors[key] = 0
                print("Fixed level")
            else:
                gear_info[key] = "999"
                errors[key] = 1
                print("Found error in level. Change to 999")

        elif key == 'ability' and int(val) > 15:
            gear_info[key] = "999"
            errors[key] = 1
            print("Found error in ability. Change to 999")

    return gear_info, errors


def get_stats(text):
    """
    Convert raw text from main, sub stats and set sections to a dictionary
    with relevant data
    :param text: raw text
    :return: dictionary
    """
    stats_wP = ['HP', 'Def', 'Atk']
    regex = r"(Speed|Health|Defense|Attack|Critical Hit Damage|Critical Hit Chance|Effectiveness|Effect Resistance) \d.*"

    line_num = 0
    sub_num = 1
    text = text.strip()
    item_stats = {}
    for key, value in change_lst.items():
        text = text.replace(key, str(value))

    for line in text.split("\n"):
        line_num += 1
        if re.search(r"set", line, re.IGNORECASE) is not None:  # "Set" in line:
            set_found = re.search(r"([a-zA-Z]+) Set", line, re.IGNORECASE).group(1).capitalize()
            if set_found in sets_name:
                item_stats["set"] = set_found
            else:
                item_stats["set"] = difflib.get_close_matches(set_found, sets_name, n=1, cutoff=0.1)[0]

        if re.search(regex, line):
            stat, value = line.rsplit(" ", 1)
            stat = rename_stat(stat)
            # if " " in stat:
            #     stat = stat.rsplit(" ", 1)[1]

            if "," in value:
                value = re.sub(",", "", value)

            if "%" in value:
                value = int(value[:-1])
                if stat in stats_wP:
                    stat += "P"

            if line_num == 1:
                # Record Main Stat
                item_stats["mainStat"] = [stat, int(value)]
                # item_stats[str("Main " + stat)] = value
            else:
                # Record substats
                item_stats["subStat" + str(sub_num)] = [stat, int(value)]
                sub_num += 1

    # if no subStats, raise error
    if sub_num <= 2:
        raise

    return item_stats


def process_img(img):
    """
    To prepare image for OCR
    """
    # Resize
    img = cv2.resize(img, (0, 0), fx=2, fy=2)
    # Threshold
    img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Blur
    img = cv2.medianBlur(img[1], 3)
    # Convert to Color
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def stat_evaluator(gear_info, spd_weight=2, crit_weight=1.5, hp_weight=0.02, atk_weight=0.1):
    rating = 0
    for key, value in gear_info.items():
        if "subStat" in key:
            stat, number = value
            if stat == "Spd":
                rating += number * spd_weight
            elif stat == "CChance":
                rating += number * crit_weight
            elif stat in ['HPP', 'CDmg', 'DefP', 'AtkP', 'Eff', 'Res']:
                rating += number
            elif stat == "HP":
                rating += number * hp_weight
            elif stat == "Atk":
                rating += number * atk_weight
            else:  # other flat stats
                rating += 0
    return int(rating)


def prep_json(gears):
    # Export json
    keep_keys = ['ability', 'level', 'set', 'slot', 'rarity', 'mainStat', 'subStat1',
                 'subStat2', 'subStat3', 'subStat4']
    my_inventory = []
    for item in gears:
        gear = {k: v for k, v in item.items() if k in keep_keys}
        for k, v in gear.items():
            if "Stat" in k:
                if v is not None:
                    if '[' in v:
                        gear.update({k: json.loads(v)})
                else:
                    continue
        lettersAndDigits = string.ascii_lowercase + string.digits
        item_id = 'dt' + ''.join(random.choice(lettersAndDigits) for l in range(6))
        gear.update({"locked": False, "efficiency": 0, "id": item_id})
        my_inventory.append(gear)
    output_json = {'processVersion': '1', 'heroes': [], 'items': []}

    output_json['items'].extend(my_inventory)
    # with open(json_path, 'w') as outfile:
    #     json.dump(output_json, outfile)
    #
    return json.dumps(output_json)  # json.dumps(output_json)


# Gear statistic
def analysis(df):
    df=df[(df['rating']<100) & (df['rating'] > 10)]
    statistic = {"n": df.shape[0],
                 "avg_playTime": round(df.playTime.mean(), 2),
                 "avg_rating": round(df.rating.mean(), 2),
                 "med_rating": round(df.rating.median(), 2),
                 "min_rating": df.rating.min(),
                 "max_rating": df.rating.max(),
                 "user_count": df.groupby("user_id").count().shape[0],
                 }
    return statistic


def export_json(data):
    """
    Convert to gear optimizer format
    :param data:
    :return:
    """
    keep_keys = ['ability', 'level', 'set', 'slot', 'rarity']
    stats = ["Atk", "AtkP", "CChance", "CDmg", "Def", "DefP", "Eff", "HP", "HPP", "Res", "Spd"]
    my_inventory = {'processVersion': '1', 'heroes': [], 'items': []}
    for item in data:
        try:
            gear = {}
            i = 1
            for key, val in item.items():
                if key in keep_keys:
                    gear.update({key: val})

                elif key in stats:
                    substat = f"subStat{i}"
                    gear.update({substat: [key, int(val)]})
                    i += 1

            gear.update({"mainStat": [item['main'], item['value']]})

            lettersAndDigits = string.ascii_lowercase + string.digits
            item_id = 'dt' + ''.join(random.choice(lettersAndDigits) for l in range(6))
            gear.update({"locked": False, "efficiency": 0, "id": item_id})
            my_inventory['items'].append(gear)
        except Exception as err:
            print(err)
            continue
    return my_inventory
