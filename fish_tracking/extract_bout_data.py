import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import pylab as pl
import pandas as pd

def find_events(data, dt, window, start_threshold, end_threshold):

    data_rolling_var = pd.rolling_var(data, window=int(window / dt), center=True)


    event_start_indices = np.where((data_rolling_var[:-1] <= start_threshold) &
                                   (data_rolling_var[1:] > start_threshold))[0].tolist()
    event_end_indices = []

    i = 0
    while i < len(event_start_indices):

        ind = np.where(data_rolling_var[event_start_indices[i]:] < end_threshold)[0]

        if len(ind) > 0:
            event_end_indices.append(event_start_indices[i] + ind[0])
            i += 1
        else:
            event_start_indices.pop(i) # if we did not find an end, the beginning is also invalid

    return data_rolling_var, np.array(event_start_indices), np.array(event_end_indices)


fish_data = np.load("/Users/arminbahl/Desktop/fish15_0321_tlab_5dpf_extracted_x_y_ang.npy")

print(".........................")
print("This is fish {}".format(fish_data))
print("Data shape", fish_data.shape)

t = fish_data[:, 0]
x = fish_data[:, 1]
y = fish_data[:, 2]
ang = fish_data[:, 3]
accummulated_ang = fish_data[:, 4]
accumulated_path = np.sqrt(np.diff(x)**2 + np.diff(y)**2).cumsum()
dt = np.diff(t).mean()


# find events

data_rolling_var, event_start_indices, event_end_indices = \
    find_events(accummulated_ang, dt, window=0.05, start_threshold=10, end_threshold=5)

# Comment: event_end_indices is faulty, don't use this for now ....

bout_start_times = t[event_start_indices]

bout_angle_changes = accummulated_ang[event_start_indices + int(0.5/dt)] - \
                     accummulated_ang[event_start_indices - int(0.02/dt)]

bout_path_changes = accumulated_path[event_start_indices + int(0.5/dt)] - \
                    accumulated_path[event_start_indices - int(0.02/dt)]

interbout_intervals = np.diff(bout_start_times)


# Plot and extract features

pl.figure()
pl.title("Fish trajectory")
pl.plot(x, y)
pl.plot(x[event_start_indices], y[event_start_indices], 'ro')
pl.show()

pl.figure()
pl.title("Rolling variance")
pl.plot(t, data_rolling_var)
pl.plot(t[event_start_indices], data_rolling_var[event_start_indices], 'ro')

pl.figure()
pl.title("Accumulated path")
pl.plot(t[1:], accumulated_path)
pl.plot(t[event_start_indices], accumulated_path[event_start_indices], 'ro')

pl.figure()
pl.title("Accumulated angle")
pl.plot(t, accummulated_ang)
pl.plot(t[event_start_indices], accummulated_ang[event_start_indices], 'ro')

a,b = np.histogram(bout_path_changes, bins=np.linspace(0, 100, 30), density=True)
pl.figure()
pl.title("Swim forward angle histogram densities")
pl.plot(b[1:], a)
print("Median forward swims: ", np.nanmedian(np.abs(bout_path_changes)))

# plot the histogram of turning angles (density plot, could be averaged over multiple fish)
a, b = np.histogram(bout_angle_changes, bins=np.linspace(-60, 60, 30), density=True)
print("Turn angle histogram densities", a)
pl.figure()
pl.title("Turn angle histogram densities")
pl.plot(b[1:], a)
print("Median absolute turn angle:", np.nanmedian(np.abs(bout_angle_changes)))

### Median to center
center_distance = np.sqrt((x-352)**2 + (y-352)**2)
print("Median distance to the center: ", np.nanmedian(center_distance))

##################
print("Average interbout interval: ", np.nanmedian(interbout_intervals))

a,b = np.histogram(interbout_intervals, bins=np.linspace(0, 3, 30), density=True)
pl.figure()
pl.title("Interbout interval histograms")
pl.plot(b[1:], a)
print("Interbout interval densities: ", a)
pl.show()

