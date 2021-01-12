import cv2
import pickle
import numpy as np
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

"""
In this file, you need to define plate_detection function.
To do:
1. Localize the plates and crop the plates
2. Adjust the cropped plate images
Inputs:(One)
1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
type: Numpy array (imread by OpenCV package)
Outputs:(One)
1. plate_imgs: cropped and adjusted plate images
type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
1. You may need to define other functions, such as crop and adjust function
2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def show_image(image, label):
    cv2.imshow(label, image)
    k = cv2.waitKey(0)


def yellow_mode(frame):
    # Blur the image to uniformize color of the plate
    gaussianKernel = cv2.getGaussianKernel(9, 2)
    blur = cv2.filter2D(frame, -1, gaussianKernel)

    # Convert to HSV color model
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Yellow parts extraction
    light_orange = (15, 60, 70)
    dark_orange = (37, 255, 220)
    mask = cv2.inRange(hsv_img, light_orange, dark_orange)

    # initialize kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # perform opening on the mask
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # perform closing on the mask
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # perform dilation on the mask
    dilation = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel)

    return dilation


def edge_detection(frame):
    sobel_Gx1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_Gy1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_Gx2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_Gy2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    frame_x1 = cv2.filter2D(frame, -1, sobel_Gx1)
    frame_y1 = cv2.filter2D(frame, -1, sobel_Gy1)
    frame_x2 = cv2.filter2D(frame, -1, sobel_Gx2)
    frame_y2 = cv2.filter2D(frame, -1, sobel_Gy2)

    one = cv2.bitwise_or(frame_x1, frame_y1)
    two = cv2.bitwise_or(frame_x2, frame_y2)
    edges = cv2.bitwise_or(one, two)

    return edges


def contour(frame, y, x):
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
        if j < 639:
            if frame[i, j + 1] == 255:
                queue.append((i, j + 1))

        # check if down pixel is 1 and add to queue if true
        if i < 479:
            if frame[i + 1, j] == 255:
                queue.append((i + 1, j))

        # check if up pixel is 1 and add to queue if true
        if i > 0:
            if frame[i - 1, j] == 255:
                queue.append((i - 1, j))

    return (frame, contour)


def debug(contour):
    plane = np.zeros((480, 640))

    count = 0
    for point in contour:
        plane[point[0], point[1]] = 255
        cv2.imshow("Plane", plane)
        cv2.waitKey(1)

def find_contours(frame):
    duplicate = np.copy(frame)
    contours = []
    for y in range(duplicate.shape[0]):
        for x in range(duplicate.shape[1]):
            if duplicate[y, x] == 255:
                duplicate, cnt = contour(duplicate, y, x)
                contours.append(cnt)

    return contours


def find_corners(contour):
    s = np.sum(contour, axis=1)
    top_left = contour[np.argmin(s)]
    bottom_right = contour[np.argmax(s)]

    d = np.diff(contour, axis=1)
    bottom_left = contour[np.argmin(d)]
    top_right = contour[np.argmax(d)]

    return (top_left, top_right, bottom_right, bottom_left)

def verify_contour(contour):
    (tl, tr, br, bl) = find_corners(contour)

    # Computes the width of the plate
    lower_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    upper_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    Width = max(int(lower_width), int(upper_width))

    # Computes the height of the plate
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    Height = max(int(right_height), int(left_height))

    # Calculate aspect_ratio of the plate
    if Width and Height:
        aspect_ratio = Height / Width
    else:
        aspect_ratio = 1

    area = Width * Height

    # condition
    return (Width > 100) and (aspect_ratio < 0.33) and (area > 2600)

def homography(frame, width, height, tl, tr, br, bl):
    TL = [0, 0]
    TR = [0, width - 1]
    BL = [height - 1, 0]
    BR = [height - 1, width - 1]

    A = np.array([
        [TL[1], TL[0], 1, 0, 0, 0, -tl[1] * TL[1], -tl[1] * TL[0]],
        [0, 0, 0, TL[1], TL[0], 1, -tl[0] * TL[1], -tl[0] * TL[0]],
        [TR[1], TR[0], 1, 0, 0, 0, -tr[1] * TR[1], -tr[1] * TR[0]],
        [0, 0, 0, TR[1], TR[0], 1, -tr[0] * TR[1], -tr[0] * TR[0]],
        [BL[1], BL[0], 1, 0, 0, 0, -bl[1] * BL[1], -bl[1] * BL[0]],
        [0, 0, 0, BL[1], BL[0], 1, -bl[0] * BL[1], -bl[0] * BL[0]],
        [BR[1], BR[0], 1, 0, 0, 0, -br[1] * BR[1], -br[1] * BR[0]],
        [0, 0, 0, BR[1], BR[0], 1, -br[0] * BR[1], -br[0] * BR[0]]
    ])

    b = np.array([tl[1], tl[0], tr[1], tr[0], bl[1], bl[0], br[1], br[0]])

    H = np.linalg.lstsq(A, b, rcond=None)[0]

    TM = np.array([
        [H[0], H[1], H[2]],
        [H[3], H[4], H[5]],
        [H[6], H[7], 1]
    ])

    frame_copy = frame.copy()
    license_plate = np.zeros((height, width, 3), np.uint8)

    for i in range(height):
        for j in range(width):
            xy1 = [j, i, 1]
            result = np.matmul(TM, xy1)
            P = [int(round(result[0] / result[2])), int(round(result[1] / result[2]))]
            if (P[1] < 480) & (P[0] < 640):
                license_plate[i, j] = frame[P[1], P[0]]

    return license_plate

def affine(frame, width, height, tl, tr, br, bl):
    TL = [0, 0]
    TR = [0, width]
    BL = [height, 0]

    A = np.array([
        [TL[1], TL[0], 1, 0, 0, 0],
        [0, 0, 0, TL[1], TL[0], 1],
        [TR[1], TR[0], 1, 0, 0, 0],
        [0, 0, 0, TR[1], TR[0], 1],
        [BL[1], BL[0], 1, 0, 0, 0],
        [0, 0, 0, BL[1], BL[0], 1]
    ])

    b = np.array([tl[1], tl[0], tr[1], tr[0], bl[1], bl[0]])

    H = np.linalg.lstsq(A, b, rcond=None)[0]
    TM = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]],[0, 0, 1]])

    frame_copy = frame.copy()
    license_plate = np.zeros((height, width, 3), np.uint8)

    for i in range(height):
        for j in range(width):
            xy1 = [j, i, 1]
            result = np.matmul(TM, xy1)
            P = [int(round(result[0])), int(round(result[1]))]
            if (P[1] < 480) & (P[0] < 640):
                license_plate[i, j] = frame[P[1], P[0]]

    return license_plate


def transform_frame(corners, frame):
    (tl, tr, br, bl) = corners

    const = 0
    # pad the corners with a constant const
    tl = [tl[0] - const, tl[1] - const]
    tr = [tr[0] - const, tr[1] + const]
    bl = [bl[0] + const, bl[1] - const]
    br = [br[0] + const, br[1] + const]

    # Computes the width of the plate
    lower_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    upper_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(lower_width), int(upper_width))

    # Computes the height of the plate
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(right_height), int(left_height))

    license_plate = affine(frame, width, height, tl, tr, br, bl)
    return license_plate

def extract_plate(frame, frame_number):
    # show_image(frame, "frame{nr}".format(nr = frame_number))
    # try to detect a yellow LP
    yellow_plate = yellow_mode(frame)
    #show_image(yellow_plate, "Yellow plate")

    # extract edges out of binary image
    edges = edge_detection(yellow_plate)
    edgesclone = edges.copy()

    # find contours in binary image
    contours = find_contours(edges)

    final = []
    for cnt in contours:
        if (verify_contour(cnt)):
            final.append(cnt)

    # finds the corners of the contour that has met all criteria

    licenseplates = []

    while (len(final) > 0):
        corners = find_corners(final.pop())
        lp = transform_frame(corners, frame)
        licenseplates.append(lp)

    frame_im = Image.fromarray(frame)
    frame_im.save('licenseplates/frame{frame_number}.jpg'.format(frame_number=frame_number), 'JPEG')

    edges_im = Image.fromarray(edges)
    edges_im.save('licenseplates/edges{frame_number}.jpg'.format(frame_number=frame_number), 'JPEG')

    count = 0
    for lp in licenseplates:
        lp_im = Image.fromarray(lp)
        lp_im.save('licenseplates/lp{frame_number}_{count}.jpg'.format(frame_number=frame_number, count=count), 'JPEG')
        count += 1

    return licenseplates


if __name__ == '__main__':
    # simply to have "checkpoints" in this code, faster for debugging and stuff
    # retrieves video frames in video_arr.txt file
    with open("video_arr.txt", "rb") as fp:
        images = pickle.load(fp)

    # images is an array with selected frames, where at each index i there is a tuple (frame_index_i, frame_i)
    plates = []

    not_found = cv2.imread('error.png')
    print("[", end="")
    for image in images:
        print("#", end="")
        result = extract_plate(image[1], image[0])
        for plate in result:
            if len(plate) != 0:
                plates.append((image[0], plate))

    print("]", end="")

    # save the cropped license plates in license_plates.txt file
    with open("license_plates.txt", "wb") as fp:
        pickle.dump(plates, fp)
