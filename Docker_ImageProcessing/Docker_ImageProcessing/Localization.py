import cv2
import pickle
import numpy as np

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
    cv2.waitKey(2000)

# Preprocess' the frames in the video (grayscale, gaussian blur)
def preprocess(image):

    # make the image gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur the imagw
    gaussianblur = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_DEFAULT)

    return gaussianblur

def edge_detection(image):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    IGx =  cv2.filter2D(image, -1, Gx)
    show_image(IGx)

    IGy = cv2.filter2D(image, -1, Gy)
    show_image(IGy)


def plate_detection(image):
    # preprocess the image
    pre_processed = preprocess(image)
    show_image(pre_processed)

    # edge detection part
    edges = edge_detection(pre_processed)

    plate_imgs = image
    return plate_imgs


if __name__ == '__main__':
    # save video frames in video_arr.txt file
    with open("video_arr.txt", "rb") as fp:
        images = pickle.load(fp)

    for image in images:
        plate_detection(image[1])