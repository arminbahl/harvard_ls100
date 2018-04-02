"""
@author: The students of spring semester 2018 LS100
"""

import imageio                  # this library allows us to load movies of various compression formats
import numpy as np              # this is the standard python library for doing data data analysis of matrices
import os                       # this is a helper library for the operating system, which allows to concatenate a path, for example
from scipy.stats import mode    # scipy included many functions for statistics, including a function for mode calculation
import pylab as pl
import cv2
import sys

#imageio.plugins.ffmpeg.download()

def calculate_background(root_path, fish_names):
    # loop through all those fish names, and calculate their backgroud images
    for fish_name in fish_names:

        print("Calculating the background image for fish", fish_name)

        # concatenate the root path, and the fish name
        path = os.path.join(root_path, fish_name)

        # load the fish movie
        movie = imageio.get_reader(path)

        # Initialize a counter so that we can create the background based on every 5th
        # frame, in order to save memory.
        list_of_selected_frames = []

        for counter, frame in enumerate(movie):

            print("Loading frame", counter)

            # Here, we grab only every n th frame from the movie for background substraction
            # the background.
            # for large long movies, you should take even less frames
            if counter % 50 == 0:
                list_of_selected_frames.append(frame[:, :, 0])

        print("Performing mode calculation on ", len(list_of_selected_frames), "frames ...")

        # the mode is the best function to calculate a background of a movie,
        # as it find the most often occurring pixel value at a given location
        # Hence, when a fish swims through the background, it does not change that value
        # the mean instead would change, and, to some better lesser extent, also the median
        modal_values, modal_count = mode(list_of_selected_frames, axis=0)

        # This saves the background as both a photo to look at, and as an array
        # to be used by the subtraction program.
        imageio.imwrite(path[:-4] + "_background.png", modal_values[0])
        np.save(path[:-4] + "_background.npy", modal_values[0])


def get_fish_position_and_angle(frame, background, threshold, filter_width, display):

    # we subtract the background and frame,
    # the fish is normally darker than the background, so we take the absolute value to
    # make the fish the brightest area in the image
    # because movie and background are likely of type unsigned int, for the sake
    # of the subtraction, make them signed integers
    background_substracted_image = np.abs(background.astype(np.int) - frame.astype(np.int)).astype(np.uint8)

    # threshold, play around with the thresholding parameter for optimal results
    ret, fish_image_thresholded = cv2.threshold(background_substracted_image, threshold, 255, cv2.THRESH_BINARY)

    # apply a weak gaussian blur to the thresholded image to get rid of noisy pixels
    # play around with the standard deviation for optimal results, normally, if little noise
    # use small values
    fish_image_blurred = cv2.GaussianBlur(fish_image_thresholded, (filter_width, filter_width), 0)

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

    # copy that region from the thresholded image and resize a little
    fish_roi_cutout = cv2.resize(fish_image_thresholded[x_left:x_right, y_up:y_down], (200, 200))

    im2, contours, hierarchy = cv2.findContours(fish_roi_cutout, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # find the biggest contour
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    cnt = contours[np.argmax(contour_areas)]

    # determine a convex hull around that contour
    hull = cv2.convexHull(cnt)

    # if we found not enough fish points
    if hull.shape[0] < 3:
        print("No fish found. Please check thresholding parameters, etc.")
        return np.nan, np.nan, np.nan

    # move the fish coordinates to the center
    hull_moved = hull.copy()
    hull_moved[:, :, 0] = hull_moved[:, :, 0] - hull[:, 0, 0].mean()
    hull_moved[:, :, 1] = hull_moved[:, :, 1] - hull[:, 0, 1].mean()

    # see https://en.wikipedia.org/wiki/Image_moment
    moments = cv2.moments(hull_moved)
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]

    fish_orientation = 0.5*np.arctan2(2 * mu11, mu20 - mu02)

    M = cv2.getRotationMatrix2D((100, 100), fish_orientation * 180/np.pi, 1)

    dummy_image_original = np.zeros((200, 200), dtype=np.uint8)

    cv2.drawContours(dummy_image_original, [hull_moved+100], 0, 255, cv2.FILLED)
    dummy_image_rotated = cv2.warpAffine(dummy_image_original, M, (200, 200))

    fish_width = np.sum(dummy_image_rotated.copy().astype(np.int), axis=0)

    if np.argmax(fish_width) < 100:
        fish_orientation += np.pi

    if display:
        img = np.zeros((200, 200)).astype(np.uint8)
        img = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

        cv2.drawContours(img, [hull_moved + 100], 0, (255, 255, 255))

        dy = int(np.cos(fish_orientation) * 20)
        dx = int(np.sin(fish_orientation) * 20)
        cv2.line(img, (100, 100), (100+dy, 100+dx), (0, 255, 0), thickness=2)

        img2 = np.concatenate((frame, fish_image_thresholded, fish_image_blurred), axis=1)
        img2 = cv2.cvtColor(img2.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # draw into the frame for displaying
        dy = int(np.cos(fish_orientation) * 20)
        dx = int(np.sin(fish_orientation) * 20)

        cv2.line(img2, (y, x), (y + dy, x + dx), (0, 255, 0), thickness=2)
        cv2.circle(img2, (y + dy, x + dx), 3, (0, 0, 255), thickness=-1)

        img2 = cv2.resize(img2, (600, 200))

        image_for_display = np.concatenate((img2, img), axis=1)
        cv2.imshow("fish image", image_for_display)

        if cv2.waitKey(1) == 27:
            sys.exit()

    # swap y, and x,
    return int(y), int(frame.shape[0]-x), (-fish_orientation*180/np.pi + 360) % 360

def extract_position_orientation(root_path, fish_names, threshold, filter_width, display):

    # loop through all those fish names, and calculate their backgroud images
    for fish_name in fish_names:

        print("Extracting position and orientation information for fish", fish_name)

        # concatenate the root path, and the fish name
        path = os.path.join(root_path, fish_name)

        # load the background
        background = np.load(path[:-4] + "_background.npy")

        # load the fish movie
        movie = imageio.get_reader(path)
        dt = 1 / movie.get_meta_data()['fps']

        ts = []
        xs = []
        ys = []
        fish_orientations = []

        for frame_counter, frame in enumerate(movie):
            print("Analyzing", frame_counter)

            image = frame[:, :, 0]
            x, y, fish_orientation = get_fish_position_and_angle(image,
                                                                 background,
                                                                 threshold=threshold,
                                                                 filter_width=filter_width,
                                                                 display=display)

            ts.append(frame_counter * dt)
            xs.append(x)
            ys.append(y)
            fish_orientations.append(fish_orientation)

        # determine the accumulated orientation
        delta_orientations = np.diff(fish_orientations)
        delta_orientations = np.nan_to_num(delta_orientations)

        ind1 = np.where(delta_orientations > 250)
        ind2 = np.where(delta_orientations < -250)

        delta_orientations[ind1] = delta_orientations[ind1] - 360
        delta_orientations[ind2] = delta_orientations[ind2] + 360

        fish_accumulated_orientation = np.cumsum(np.r_[fish_orientations[0], delta_orientations])

        # Saving the data in the same folder and the same base file name as the fish movie
        fish_data = np.c_[ts, xs, ys, fish_orientations, fish_accumulated_orientation]
        np.save(path[:-4] + "_extracted_x_y_ang.npy", fish_data)

        # save a plot to easily check the quality of the tracking
        pl.figure(figsize=(10, 5))
        pl.subplot(121)
        an = np.linspace(0, 2 * np.pi, 100)

        pl.plot(350 + 350 * np.cos(an), 350 + 350 * np.sin(an), label='Dish boundary')
        pl.plot(fish_data[:, 1], fish_data[:, 2], label='Trajectory')
        pl.plot(fish_data[0, 1], fish_data[0, 2], 'o', label='Start')
        pl.plot(fish_data[-1, 1], fish_data[-1, 2], 'o', label='End')

        pl.xlim(0, 700)
        pl.ylim(0, 700)
        pl.axis('equal')
        pl.xlabel("x (Pixel)")
        pl.ylabel("y (Pixel)")
        pl.legend()

        pl.subplot(122)
        pl.xlabel("Time (s)")
        pl.ylabel("Angle (deg)")
        pl.plot(fish_data[:, 0], fish_data[:, 3], label='Raw angle')
        pl.plot(fish_data[:, 0], fish_data[:, 4], label='Accumulated angle change')
        pl.legend()

        pl.savefig(path[:-4] + "_extracted_x_y_ang.png")