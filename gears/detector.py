import datetime
import os

import cv2
import numpy as np


def detector(video, out_path, progress_callback=None, seconds=0.1, brightness_cutoff=0.8):
    """
    Loop over the frames of the video to capture frames with equipment.
    :param video: mp4 file location
    :param out_path: output path to save frames as jpg
    :param seconds: only analyze frame by a number of second of a fraction of second
    :param brightness_cutoff: set % change in brightness. If higher than this, save frame
    """
    vs = cv2.VideoCapture(video)
    firstFrame = None
    newGear = True
    fps = vs.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
    v_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    v_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    multiplier = int(fps * seconds)
    count = 0
    out_files = []
    if v_height > v_width:
        print("Vertical video")
        raise ValueError("Vertical video")

    while True:
        frameId = int(round(vs.get(1)))

        # grab the current frame
        frame = vs.read()
        frame = frame[1]
        text = f"Exported {count} gears."

        # if last frame
        if frame is None:
            break

        # If not skipped frame
        if frameId % multiplier == 0:
            # resize the frame, convert it to grayscale, and blur it
            # frame = imutils.resize(frame, width=720)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = hsv
                _, _, first_v = cv2.split(firstFrame)
                first_brightness = np.mean(first_v)
                continue

            _, _, v = cv2.split(hsv)
            brightness = np.mean(v)

            if brightness < first_brightness * brightness_cutoff:
                if newGear:
                    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    file_name = os.path.join(out_path, f"gear_{date_str}_{count}.jpg")
                    text = "Found new gear"
                    cv2.imwrite(file_name, frame)  # save frame as JPEG file
                    out_files.append(file_name)

                    count += 1
                    # print(text)
                    if progress_callback:
                        progress_callback.emit("detecting", None, count, None)
                    else:
                        print(text)
                    newGear = False
                else:
                    text = "Saved. Waiting for new one"

            else:
                newGear = True

    print(text)
    vs.release()
    cv2.destroyAllWindows()

    return out_files
