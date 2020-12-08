import cv2
import numpy as np
import pickle
import os
import pandas as pd
import Localization
import Recognize

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
1. file_path: video path
2. sample_frequency: second
3. save_path: final .csv file path
Output: None
"""

# samples the video with the given sample frequency, adds the selected frames to a list together with the index
def sample_video(file_path, sample_frequency):
    video = cv2.VideoCapture(file_path)

    # get the amount of frames per second of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    video_arr = []

    sum = 0
    while (video.isOpened()):
        ret, frame = video.read()

        if (ret == True):

            if ((sum / fps) % sample_frequency == 0):
                video_arr.append((sum, frame))

        if(sum == 48):
            break
        sum += 1

    return video_arr


def CaptureFrame_Process(file_path, sample_frequency, save_path):

    # converts the video to an array of tuples containing the index (or frame number) and image on the corresponding image
    # the video is subsampled with a sample_frequency specified in the arguments of main.py
    video_arr = sample_video(file_path, sample_frequency)

    # save video frames in video_arr.txt file
    with open("video_arr.txt", "wb") as fp:
        pickle.dump(video_arr, fp)

    """for frame in video_arr:
        localization = Localization.plate_detection(frame[1])"""

    pass
