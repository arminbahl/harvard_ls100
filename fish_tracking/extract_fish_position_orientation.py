"""
Created on Mon Feb 26 15:15:30 2018

@author: The students of spring semester 2018 LS100
"""

import imageio
import matplotlib
matplotlib.use("qt5agg")
import pylab as pl
import numpy as np
import os
import cv2
import sys

def get_fish_position_and_angle(frame, background, threshold, filter_width, display):

    # we substract the background and frame,
    # the fish is normally darker than the background, so we take the absolute value to
    # make the fish the brightest area in the image
    # because movie and background are likely of type unsigned int, for the sake
    # of the substraction, make them signed integers
    background_substracted_image = np.abs(background.astype(np.int) - frame.astype(np.int)).astype(np.uint8)

    # threshold, play around with the tresholding paramater for optimal results
    #print(background_substracted_image.shape, background_substracted_image.dtype)
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

    moments = cv2.moments(hull_moved)
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]

    fish_orientation = 0.5*np.arctan2(2 * mu11, mu20 - mu02)

    M = cv2.getRotationMatrix2D((200 / 2, 200 / 2), fish_orientation * 180/np.pi, 1)

    dummy_image_original = np.zeros((200, 200), dtype=np.uint8)

    cv2.drawContours(dummy_image_original, [hull_moved+100], 0, 255, cv2.FILLED, 8)
    dummy_image_rotated = cv2.warpAffine(dummy_image_original, M, (200, 200))

    fish_width = np.sum(dummy_image_rotated.copy().astype(np.int), axis=0)

    if np.argmax(fish_width) < 100:
        fish_orientation += np.pi

    if display:
        img = np.zeros((200, 200)).astype(np.uint8)
        cv2.drawContours(img, [hull_moved + 100], 0, 128)

        dy = int(np.cos(fish_orientation) * 20)
        dx = int(np.sin(fish_orientation) * 20)
        cv2.line(img, (100, 100), (100+dy, 100+dx), 255, thickness=2)
        cv2.imshow("test2", img)
        cv2.waitKey(1)

        stiched_images = np.concatenate((image, fish_image_thresholded, fish_image_blurred), axis=1)
        image_for_display = cv2.cvtColor(stiched_images.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # draw into the frame for displaying
        dy = int(np.cos(fish_orientation) * 20)
        dx = int(np.sin(fish_orientation) * 20)

        cv2.line(image_for_display, (y, x), (y + dy, x + dx), (0, 255, 0), thickness=2)
        cv2.circle(image_for_display, (y + dy, x + dx), 3, (0, 0, 255), thickness=-1)

        image_for_display = cv2.resize(image_for_display, None, fx=0.8, fy=0.8)
        cv2.imshow("fish image", image_for_display)

        if cv2.waitKey(1) == 27:
            sys.exit()


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
        #pl.plot(coordinates_original_cutout[0], coordinates_original_cutout[1], '.')
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
        dx = np.cps(fish_orientation)*scale*2
        dy = np.cos(fish_orientation)*scale*2

        # original fish coordinates
        pl.plot(coordinates_original_cutout[0], coordinates_original_cutout[1], '.')
        pl.plot([x_cutout], [y_cutout], 'o')
        pl.plot([x_cutout + dx], [y_cutout + dy], 'o')
        pl.plot([x_cutout, x_cutout + dx], [y_cutout, y_cutout + dy])
        pl.show()
        """


    # swap y, and x,
    return int(y), int(frame.shape[0]-x), -fish_orientation*180/np.pi

#display an example fish
"""
# load the background subtracted image (or loop through a movie)
print("Analyzing test fish.....")

image = imageio.imread("fish.png")[:, :, 0]
background = np.zeros_like(image)

x, y, fish_angle = get_fish_position_and_angle(image, background,
                                                             threshold = 35,
                                                             filter_width = 2,
                                                             display=True)

print("Fish position in image: ", (x, y))
print("Fish angle in image (in degrees): ", fish_angle * 180/np.pi)
"""



###########################
# Analyzing the fish list

# this is the path where all the fish movies reside
root_path = r"/Users/arminbahl/Desktop/ls100"

# a list of all the fish where the background should be calculated
fish_names = ["fish8_0316_20866_7dpf.avi"]

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

    frame_counter = 0
    for frame in movie:
        print("Analyzing", frame_counter)

        image = frame[:, :, 0]
        x, y, fish_orientation = get_fish_position_and_angle(image,
                                                             background,
                                                             threshold=20,
                                                             filter_width=5, # has to be odd
                                                             display=False)

        ts.append(frame_counter * dt)
        xs.append(x)
        ys.append(y)
        fish_orientations.append(fish_orientation)

        frame_counter += 1

        if frame_counter > 2000:
            break


    # determine the accumulated orientation
    delta_orientations = np.diff(fish_orientations)
    delta_orientations = np.nan_to_num(delta_orientations)

    ind1 = np.where(delta_orientations > 300)
    ind2 = np.where(delta_orientations < -300)

    delta_orientations[ind1] = delta_orientations[ind1] - 360
    delta_orientations[ind2] = delta_orientations[ind2] + 360

    fish_accumulated_orientation = np.cumsum(np.r_[fish_orientations[0], delta_orientations])

    #pl.plot(ts, fish_orientations)
    #pl.plot(ts, fish_accumulated_orientation)
    #pl.show()

    # Saving the data in the same folder and the same base file name as the fish movie
    np.save(path[:-4] + "_extracted_x_y_ang.npy", np.c_[ts, xs, ys, fish_orientations, fish_accumulated_orientation])