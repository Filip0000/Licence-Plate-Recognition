import cv2
import pickle
import numpy as np
import matplotlib as plt

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

def show_image(image):
    cv2.imshow('Image', image)
    k = cv2.waitKey(0)

def find_corners2(contour):
    hull = cv2.convexHull(contour)
    rect = np.zeros((4, 2))
    pts = []
    for pt in hull:
        pts.append(pt[0])

    s = np.sum(pts, axis=1)
    # Top-left
    rect[0] = pts[np.argmin(s)]
    # Bottom-right
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def verify_plate(box):
    """
    Verifies that a plate correspond to basic standards and has appropriate properties
    :param box: coordinates of the four vertices of the bounding rectangle
    :return: boolean value: True if plate is acceptable, False otherwise
    """
    rect = order_points(box)
    (tl, tr, br, bl) = rect

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
        aspect_ratio = Height/Width
    else:
        aspect_ratio = 1

    # Calculate Area of the plate
    area = cv2.contourArea(box)

    # Set conditions for an acceptable plate
    return (Width > 100) and (aspect_ratio < 0.3) and (area > 2600)


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def plate_transform(image, box):
    """
    Transforms a inclined plate into a straight plate
    :param image: plate image
    :param box: list of the four vertices' coordinates of the plate's bounding rectangle
    :return: straightened image
    """

    # obtain the bounding rectangle's vertices and order them
    rect = order_points(box)
    (tl, tr, br, bl) = rect

    # Computes the width of the plate
    lower_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    upper_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(lower_width), int(upper_width))

    # Computes the height of the plate
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(right_height), int(left_height))

    # Construct the set of destination points to obtain a "birds eye view" of the plate
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (width, height))

    # return the warped image
    return warped


def plate_detection(image, contours):
    """
    Detects the plate on the frame
    :param image: frame to be analyzed
    :param contours: contours retrieved of the pre-processed frame
    :return: list containing images of all plates detected
    """
    final_contours = []
    i = 0
    corner_table = np.zeros((4, 2))
    # hull_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for cnt in contours: # Loops and verify all contours for acceptable plates
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if verify_plate(box):
            corners = find_corners2(cnt)

            """corners = find_corners(cnt)
            for i in range(4):
                corner_table[i, 0] = int(corners[i, 0, 0])
                corner_table[i, 1] = int(corners[i, 0, 1])
            print(corner_table)
            for pnt in cross_points:
                cv2.circle(houghed, pnt, 5, (0,0,255), -1)
            cv2.imshow("hough",houghed)
            cv2.waitKey(0)"""
            
            final_contours.append(box)
        i += 1

    if not final_contours: # Returns None if no acceptable plate found
        return None

    # localized = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(localized, final_contours, 0, (0, 255, 0), 3)

    # Transforms and straighten each acceptable contours
    plate_img = []
    for cnt in final_contours:
        plate_img.append(plate_transform(image, cnt))

        # Show each localized plates
        #show_image(plate_img[len(plate_img)-1])
    return plate_img


def yellow_mode(frame):
    """
    Localize Dutch yellow license plates and recognize them.
    :param frame: Actual frame extracted from the video.
    :return: list containing all plates recognized
    """

    # Blur the image to uniformize color of the plate
    blur = cv2.GaussianBlur(frame, (9, 9), 0)

    # Keep record of gray_scales frame for window detection

    # Convert to HSV color model
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Yellow parts extraction
    light_orange = (15, 60, 70)
    dark_orange = (37, 255, 220)
    mask = cv2.inRange(hsv_img, light_orange, dark_orange)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # initialize kernel for morphological operations
    kernel = np.ones((5,5),np.uint8)

    # perform opening on the mask
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    
    masked_opening = cv2.bitwise_and(frame, frame, mask=opening)

    # perform dilation on the mask
    dilate = cv2.dilate(opening, kernel, iterations = 1)
    LP_I = cv2.bitwise_and(frame, frame, mask=dilate)

    show_image(LP_I)

       
    # BGR to gray scale conversion
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # Binarize frame with very low threshold to ease edge detection
    (thresh, binary) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Perform canny edge detection
    edged = cv2.Canny(binary, 50, 100)

    # retrieve contours of the plate
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Change original image to gray scale
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Localize all plates in the frame and return them in a single list
    plates =  plate_detection(gray_original, contours)

    return plates





if __name__ == '__main__':
    # retrieves video frames in video_arr.txt file
    with open("video_arr.txt", "rb") as fp:
        images = pickle.load(fp)

    #plate_detection(cv2.imread('licenseplate.png'))

    plates = []

    not_found = cv2.imread('error.png')

    for image in images:
        image = np.array(image[1])
        plates.append(yellow_mode(image))

    for plate in plates:
        if(plate is not None):
            show_image(plate[len(plate) - 1])

        else:
            show_image(not_found)

















"""def plate_detection(img):
    # preprocess the image
    #gray = preprocess(img)
    #show_image(gray)

    print(np.average(img[:, :, 0]))
    print(np.average(img[:, :, 1]))
    print(np.average(img[:, :, 2]))

    lower = np.array([8, 73, 85])
    upper = np.array([95, 200, 190])

    lower = np.array([8, 73, 95])
    upper = np.array([86, 166, 199])

    mask = cv2.inRange(img, lower, upper)

    show_image(mask)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    dilation = cv2.dilate(opening, kernel, iterations=1)
    show_image(dilation)

    cnts, tree = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0,255,0), 3)
    show_image(img)

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(img, screenCnt, -1, (0,255,0), 3)
    show_image(img)

    plate_imgs = img
    return plate_imgs"""



