import numpy as np
import os
import pylab as pl
root_path = r"/Users/arminbahl/Dropbox/fish_traking_yasuko"

fish_names = ["180207_15.mov"]

for fish_name in fish_names:
    path = os.path.join(root_path, fish_name)[:-4] + "_extracted_x_y_ang.npy"
    fish_data = np.load(path)

    print(fish_data.shape)

    pl.plot(fish_data[:,0], fish_data[:,1])
    pl.show()
    pl.plot(fish_data[:,2] * 180/np.pi)
    pl.show()