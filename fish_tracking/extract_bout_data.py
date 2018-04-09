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

print(fish_data.shape)

t = fish_data[:, 0]
x = fish_data[:, 1]
y = fish_data[:, 2]
ang = fish_data[:, 3]
accummulated_ang = fish_data[:, 4]
dt = np.diff(t).mean()


data_rolling_var, event_start_indices, event_end_indices = \
    find_events(accummulated_ang, dt, window=0.05, start_threshold=10, end_threshold=5)

bout_start_times = t[event_start_indices]
angles_before_bout = accummulated_ang[event_start_indices - int(0.02/dt)]
angles_after_bout = accummulated_ang[event_start_indices + int(0.5/dt)]

angle_changes = angles_after_bout-angles_before_bout

print(np.nanmedian(np.abs(angle_changes)))

a,b = np.histogram(angle_changes, bins=np.linspace(-60, 60, 30))
pl.figure()
pl.plot(b[1:], a)
pl.show()
n
print(angles_before_bout)
asdf


bout_bout_end_times = t[event_end_indices]


interbout_intervals = np.diff(bout_start_times)

pl.plot(t, accummulated_ang)
pl.plot(t[event_start_indices], accummulated_ang[event_start_indices], 'ro')
#pl.plot(t[event_end_indices], accummulated_ang[event_end_indices], 'go')

pl.plot(t, data_rolling_var)

### Median to center
center_distance = np.sqrt((x-352)**2 + (y-352)**2)
print("Median distance to the center")
print(np.nanmedian(center_distance))


##################
print("Average interbout interval")
print(np.nanmedian(interbout_intervals))

a,b = np.histogram(interbout_intervals, bins=np.linspace(0, 3, 30))
pl.figure()
pl.plot(b[1:], a)
pl.show()

