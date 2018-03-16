"""
Created on Mon Feb 26 15:15:30 2018

@author: The students of spring semester 2018 LS100
"""

# This file will be used to make the background for our movie, and save this
# background as an array (and photo for sanity check) that can be used
# by our movie analyzer.

import imageio                  # this library allows us to load movies of various compression formats
import numpy as np              # this is the standard python library for doing data data analysis of matrices
import os                       # this is a helper library for the operating system, which allows to concatenate a path, for example
from scipy.stats import mode    # scipy included many functions for statistics, including a function for mode calculation

# this is the path where all the fish movies reside
root_path = r"C:\Users\LS100\Desktop\fish_movies"

# a list of all the fish where the background should be calculated
fish_names = ["2min.avi"]

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
    counter = 0
    for frame in movie:

        print("Loading frame", counter)

        # Here, we grab only every 5th frame from the movie for background substraction
        # the background.
        # for large long movies, you should take even less frames
        if counter % 5:
            list_of_selected_frames.append(frame[:, :, 0])

        counter += 1

    # the mode is the best function to calculate a background of a movie,
    # as it find the most often occuring pixel value at a given location
    # Hence, when a fish swims through the background, it does not change that value
    # the mean instead would change, and, to some better lesser extent, also the median
    modal_values, modal_count = mode(list_of_selected_frames, axis=0)

    # This saves the background as both a photo to look at, and as an array
    # to be used by the subtraction program.
    imageio.imwrite(path[:-4] + "_background.png", modal_values)
    np.save(path[:-4] + "_background.npy", modal_values)