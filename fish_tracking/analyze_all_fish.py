"""
@author: The students of spring semester 2018 LS100
"""

import tracking_functions

# this is the path where all the fish movies reside
root_path = r"/Users/arminbahl/Desktop/ls100"

# a list of all the fish where the background should be calculated
fish_names = ["fish8_0316_20866_7dpf.avi"]

print("Calculate backgrounds")
tracking_functions.calculate_background(root_path, fish_names)

tracking_functions.extract_position_orientation(root_path, fish_names=fish_names,
                                                threshold=20, filter_width=5,
                                                display=False) # filter_width has to be odd
