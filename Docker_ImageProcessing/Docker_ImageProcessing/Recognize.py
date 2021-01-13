import cv2
import numpy as np
import os
import pickle

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

def show_image(image):
    cv2.imshow('Image', image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour(frame, y, x, height, width):
    contour = []

    queue = []
    queue.append((y, x))

    while (len(queue) > 0):
        (i, j) = queue.pop()
        frame[i, j] = 0
        contour.append((i, j))

        # check if left pixel is 1 and add to queue if true
        if j > 0:
            if frame[i, j - 1] == 255:
                queue.append((i, j - 1))

        # check if right pixel is 1 and add to queue if true
        if j < width - 1:
            if frame[i, j + 1] == 255:
                queue.append((i, j + 1))

        # check if down pixel is 1 and add to queue if true
        if i < height - 1:
            if frame[i + 1, j] == 255:
                queue.append((i + 1, j))

        # check if up pixel is 1 and add to queue if true
        if i > 0:
            if frame[i - 1, j] == 255:
                queue.append((i - 1, j))

    return (frame, contour)

def find_contours(frame):
    duplicate = np.copy(frame)
    contours = []
    for y in range(duplicate.shape[0]):
        for x in range(duplicate.shape[1]):
            if duplicate[y, x] == 255:
                duplicate, cnt = contour(duplicate, y, x, duplicate.shape[0], duplicate.shape[1])
                contours.append(cnt)

    return contours


def preprocess(plate):
    blur = cv2.medianBlur(plate,5)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2)

    # initialize kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    # perform opening on the mask
    opening = cv2.morphologyEx(thresh_mean, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    height, width = plate.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    border_width = round(0.1 * height)
    mask[border_width:height - border_width, border_width:width - border_width] = 255

    masked = cv2.bitwise_and(closing, mask, mask=None)

    return masked


def draw_contour(contour, plane):
    
    count = 0
    for point in contour:
        plane[point[0], point[1]] = (255, 255, 255)
    
    return plane


def find_borders(character):
    y_values = [c[0] for c in character]
    x_values = [c[1] for c in character]

    l_min = np.min(x_values)
    r_max = np.max(x_values)

    t_min = np.min(y_values)
    b_max = np.max(y_values)

    return l_min, r_max, t_min, b_max


def filter_contours(plate, contours):
    # minimum relative area of character contour is 0.04585152838427948 based on this a threshold was made
    area_list = []
    height, width = plate.shape[0:2]

    character_list = []

    for cnt in contours:
        area = len(cnt)
        relative_area = area / (height * width)
        l, r, t, b = find_borders(cnt)
        relative_width = (r - l) / width
        if relative_width < 0.20:
            if relative_area > 0.03:
                character_list.append(cnt)

    return character_list
    

def bounding_box(plate, characters):
    bounding_boxes = []
    height, width = plate.shape[0:2]
    blank_lp = np.zeros((height, width, 3), np.uint8)

    constant = 2

    sum = 0
    for c in characters:
        blank_lp = draw_contour(c, blank_lp)
        l, r, t, b = find_borders(c)

        l = l - constant
        r = r + constant
        t = t - constant
        b = b + constant

        blank_lp[t, l:r] = (255, 0 , 255)
        blank_lp[b, l:r] = (255, 0 , 255)
        blank_lp[t:b, l] = (255, 0 , 255)
        blank_lp[t:b, r] = (255, 0 , 255)

        box = blank_lp[t:b, l:r]
        bounding_boxes.append((l, box))

    show_image(blank_lp)

    return sorted(bounding_boxes, key=lambda x: x[0]) 

def read_boxes():
    """ THIS NEEDS TO BE IMPLEMENTED LAST PART """
    """ THIS NEEDS TEMPLATE MATCHING  """


def read_plate(license_plate): 
    show_image(license_plate)
    # preprocess the image and output a binary image of license plate
    binary_plate = preprocess(license_plate)

    # find contours in binary image
    contours = find_contours(binary_plate)

    # filter out the characters from the contours
    characters = filter_contours(license_plate, contours)

    # create bounding boxes around contours and output to a list 
    # (which is sorted from left-most character to right-most)
    character_boxes = bounding_box(license_plate, characters)
    
    # read the characters 
    #LP_text = read_boxes(character_boxes)

if __name__ == '__main__':
    with open("license_plates.txt", "rb") as fp:
        plates = pickle.load(fp)

    for plate in plates:
        if plate[0] > 1000:
            break
        read_plate(plate[1])


