"""
Created on Mon Feb 26 15:15:30 2018

@author: The students of spring semester 2018 LS100
"""

import imageio
#import matplotlib
#matplotlib.use("qt5agg")
import pylab as pl
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import cv2


def get_fish_position_and_angle(frame, background, threshold, filter_width, display):

    # we substract the background and frame,
    # the fish is normally darker than the background, so we take the absolute value to
    # make the fish the brightest area in the image
    # because movie and background are likely of type unsigned int, for the sake
    # of the substraction, make them signed integers
    background_substracted_image = np.abs(background.astype(np.int) -
                                          frame.astype(np.int))

    # threshold, play around with the tresholding paramater for optimal results
    fish_image_thresholded = background_substracted_image.copy()
    fish_image_thresholded[fish_image_thresholded < threshold] = 0
    fish_image_thresholded[fish_image_thresholded >= threshold] = 255

    # apply a weak gaussian blur to the thresholded image to get rid of noisy pixels
    # play around with the standard deviation for optimal results, normally, if little noise
    # use small values
    fish_image_blurred = gaussian_filter(fish_image_thresholded, (filter_width, filter_width))

    # find the position of the maximum with
    x, y = np.unravel_index(np.argmax(fish_image_blurred), fish_image_blurred.shape)

    # cut out a region of interest square 100 pixels around the fish
    x_left = x - 50
    x_right = x + 50
    y_up = y - 50
    y_down = y + 50

    # we have to take same care of rare incidences where the region of interest would fall outside the movie
    if x_left < 0:
        x_left = 0
        x_right = x_left + 100

    if x_right >= background_substracted_image.shape[0]:
        x_right = background_substracted_image.shape[0] - 1
        x_left = x_right - 100

    if y_up < 0:
        y_up = 0
        y_down = y_up + 100

    if y_down >= background_substracted_image.shape[1]:
        y_down = background_substracted_image.shape[1] - 1
        y_up = y_down - 100

    # copy that region from the thresholded image
    fish_roi_cutout = fish_image_thresholded[x_left:x_right, y_up:y_down].copy()

    # translate the image pixels into a vector of x,y
    coordinates_original_cutout = np.array(np.where(fish_roi_cutout == 255))

    # if we found not enough fish points
    if coordinates_original_cutout.shape[1] < 10:
        print("No fish found. Please check thresholding parameters, etc.")
        return np.nan, np.nan, np.nan

    # determine the center of mass of the cutout
    x_cutout = coordinates_original_cutout[0].mean()
    y_cutout = coordinates_original_cutout[1].mean()

    # move the fish coordinates to the center
    coordinates_moved_cutout = coordinates_original_cutout.copy()
    coordinates_moved_cutout[0] = coordinates_moved_cutout[0] - x_cutout
    coordinates_moved_cutout[1] = coordinates_moved_cutout[1] - y_cutout

    # the eigenvectors of the covariance matrix determine
    # the axes of a coordinate system along the largest/smallest variance
    cov = np.cov(coordinates_moved_cutout)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    # what is the angle of largest axes relative to our original axis? - this is the fish orienteation
    fish_orientation = np.tanh(x_v1 / y_v1)

    # create a rotation matrix with that angle
    rotation_mat = np.matrix([[np.cos(fish_orientation), -np.sin(fish_orientation)],
                              [np.sin(fish_orientation), np.cos(fish_orientation)]])


    # rotate the mean-centered fish points according to that angle
    coordinates_rotated_cutout = np.array(rotation_mat * coordinates_moved_cutout)

    # if more dots are negative than positive in the rotated fish, the head looks downwards.
    # In that case add 180 degrees to the fish orientation
    ind_positive = coordinates_rotated_cutout[1] > 0
    ind_negative = coordinates_rotated_cutout[1] < 0

    if ind_negative.sum() > ind_positive.sum():
        fish_orientation += np.pi

    # we are done. ;-)

    # if desired, display all the processing stages
    if display:

        stiched_images = np.concatenate((image, fish_image_thresholded, fish_image_blurred), axis=1)
        image_for_display = cv2.cvtColor(stiched_images.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # draw into the frame for displaying
        dy = int(np.cos(fish_orientation) * 10)
        dx = int(np.sin(fish_orientation) * 10)

        print(fish_image_thresholded.dtype)
        cv2.line(image_for_display, (y, x), (y + dy, x + dx), (0, 0, 255), thickness=1)
        cv2.circle(image_for_display, (y + dy, x + dx), 3, (0, 0, 255), thickness=-1)

        image_for_display = cv2.resize(image_for_display, None, fx=0.5, fy=0.5)
        cv2.imshow("fish image", image_for_display)
        cv2.waitKey(1)

        """
        pl.title("Stage 0: Original frame")
        pl.imshow(frame, cmap='gray')
        pl.show()

        ####
        pl.title("Stage 1: Background substracted frame")
        pl.imshow(background_substracted_image, cmap='gray')
        pl.show()

        ####
        pl.title("Stage 2: Thresholding")
        pl.imshow(fish_image_thresholded, cmap='gray')
        pl.show()

        ####
        pl.title("Stage 3: Blurring")
        pl.imshow(fish_image_blurred, cmap='gray')
        pl.show()

        ####
        pl.title("Stage 4: Cut out region of interest")
        pl.imshow(fish_roi_cutout, cmap='gray')
        pl.show()

        ####
        # original fish coordinates and center-moved corrdinates
        pl.title("Stage 5: Extract coordinates and move to center of mass")
        pl.plot(coordinates_original_cutout[0], coordinates_original_cutout[1], '.')
        pl.plot(coordinates_moved_cutout[0], coordinates_moved_cutout[1], '.')

        # the principle axes of the new coordinate system
        scale = 20
        pl.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
                [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')

        pl.plot([x_v2 * -scale, x_v2 * scale],
                [y_v2 * -scale, y_v2 * scale], color='blue')

        pl.show()

        pl.title("Stage 6: Rotate and determine number of points below and above zero")
        pl.plot(coordinates_rotated_cutout[0], coordinates_rotated_cutout[1], '.')
        pl.plot(coordinates_rotated_cutout[0][ind_positive], coordinates_rotated_cutout[1][ind_positive], '+')  # color the positive ones
        pl.plot(coordinates_rotated_cutout[0][ind_negative], coordinates_rotated_cutout[1][ind_negative], '_')  # color the negative ones
        pl.show()

        ################
        pl.title("Displaying results")
        dx = np.sin(fish_orientation)*scale*2
        dy = np.cos(fish_orientation)*scale*2

        # original fish coordinates
        pl.plot(coordinates_original_cutout[0], coordinates_original_cutout[1], '.')
        pl.plot([x_cutout], [y_cutout], 'o')
        pl.plot([x_cutout + dx], [y_cutout + dy], 'o')
        pl.plot([x_cutout, x_cutout + dx], [y_cutout, y_cutout + dy])
        pl.show()
        """

    # swap y, and x,
    return int(y), int(x), fish_orientation

# display an example fish
#
# # load the background subtracted image (or loop through a movie)
# print("Analyzing test fish.....")
#
# fish_image = imageio.imread("fish.png")[:, :, 0]
# background = np.zeros_like(fish_image)
#
# x, y, fish_angle = get_fish_position_and_angle(fish_image,
#                                                background,
#                                                display=True)
#
# print("Fish position in image: ", (x, y))
# print("Fish angle in image (in degrees): ", fish_angle * 180/np.pi)
#



###########################
# Analyzing the fish list

# this is the path where all the fish movies reside
root_path = r"/Users/arminbahl/Dropbox/fish_traking_yasuko"

# a list of all the fish where the background should be calculated
fish_names = ["180207_15.mov"]

# loop through all those fish names, and calculate their backgroud images
for fish_name in fish_names:

    print("Extracting position and orientation information for fish", fish_name)

    # concatenate the root path, and the fish name
    path = os.path.join(root_path, fish_name)

    # load the background
    background = np.load(path[:-4] + "_background.npy")

    # load the fish movie
    movie = imageio.get_reader(path)

    xs = []
    ys = []
    fish_orientations = []

    frame_counter = 0
    for frame in movie:
        print("Analyzing", frame_counter)
        image = frame[:, :, 0]
        x, y, fish_orientation = get_fish_position_and_angle(image,
                                                             background,
                                                             threshold = 35,
                                                             filter_width = 2,
                                                             display=False)


        xs.append(x)
        ys.append(y)
        fish_orientations.append(fish_orientation)

        frame_counter += 1

    # Saving the data in the same folder and the same base file name as the fish movie
    np.save(path[:-4] + "_extracted_x_y_ang.npy", np.c_[xs, ys, fish_orientations])