import cv2
import os
import sys
import math
import numpy as np

def load_img(dir, files, rgb):
    img = []
    for f in files:
        img.append(cv2.imread(os.path.join(dir, f))[:, :, rgb])
    return img

def read_exposure_times(dir):
    with open(os.path.join(dir, 'image_list.txt'), 'r') as f:
        list_lines = f.readlines()
    files = []
    exposures = []
    for line in list_lines:
        s = line.split()
        files.append(s[0])
        exposures.append(float(s[1]))
    return files, exposures

def progress(count, total, status):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ... %s \r' % (bar, percents, '%', status))
    sys.stdout.flush()