import cv2
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

"""
In this file, you will define your own segment_and_recognize function.
To do:
    1. Segment the plates character by character
    2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
    3. Recognize the character by comparing the distances
Inputs:(One)
    1. plate_imgs: cropped plate images by Localization.plate_detection function
    type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
    1. recognized_plates: recognized plate characters
    type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
    You may need to define other functions.
"""


def segment_and_recognize(plate_imgs):
    recognized_plates = 0

    return recognized_plates


def show_image(image, label):
    cv2.imshow(label, image)
    k = cv2.waitKey(0)

def preprocess(frame):
    gaussianKernel = cv2.getGaussianKernel(9, 1)

    blur = cv2.filter2D(frame, -1, gaussianKernel)

    mask = np.zeros(frame.shape, dtype=np.uint8)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # initialize kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # perform opening on the mask
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    show_image(thresh, "thresh")

    show_image(opening, "opening")
    color = ('b', 'g', 'r')

    # initialize kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # perform opening on the mask
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

    # perform closing on the mask
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


def optical_character_recognition(frame):
    # first perform preprocessing on licenseplate
    #show_image(frame, "frame")
    binary = preprocess(frame)
    #show_image(binary, "binary")


if __name__ == '__main__':
    with open("license_plates.txt", "rb") as fp:
        plates = pickle.load(fp)

    for plate in plates:
        print(plate[0], end=", ")
        optical_character_recognition(plate[1])
