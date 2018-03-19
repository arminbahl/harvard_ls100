import numpy as np
import os
import pylab as pl
root_path = r"D:\Zebrafish"

# a list of all the fish where the background should be calculated
fish_names = ["fc2_save_2018-02-26-150603.avi"]

for fish_name in fish_names:
    path = os.path.join(root_path, fish_name)[:-4] + "_extracted_x_y_ang.npy"
    fish_data = np.load(path)

    print(fish_data.shape)

    pl.plot(fish_data[:,1], fish_data[:,2])
    pl.show()
    pl.plot(fish_data[:, 0], fish_data[:, 3])
    pl.plot(fish_data[:, 0], fish_data[:, 4])
    pl.show()