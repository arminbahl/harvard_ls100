import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import pylab as pl
import pandas as pd
from pandas import DataFrame

fish_name = "fish15_0321_tlab_5dpf"
fish_data = np.load("/Users/arminbahl/Desktop/{}_extracted_x_y_ang.npy".format(fish_name))

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
data_rolling_var = pd.rolling_var(accummulated_ang, window=int(0.05 / dt), center=True)

event_start_indices = np.array(np.where((data_rolling_var[:-1] <= 10) &
                                        (data_rolling_var[1:] > 10))[0])

bout_start_times = t[event_start_indices]

bout_angle_changes = accummulated_ang[event_start_indices + int(0.5/dt)] - \
                     accummulated_ang[event_start_indices - int(0.02/dt)]

bout_path_changes = accumulated_path[event_start_indices + int(0.5/dt)] - \
                    accumulated_path[event_start_indices - int(0.02/dt)]

interbout_intervals = np.diff(bout_start_times)

center_distance = np.sqrt((x-352)**2 + (y-352)**2)


##################
pl.figure()
pl.title("Fish trajectory")
pl.plot(x, y)
pl.plot(x[event_start_indices], y[event_start_indices], 'ro')

##################
pl.figure()
pl.title("Rolling variance")
pl.plot(t, data_rolling_var)
pl.plot(t[event_start_indices], data_rolling_var[event_start_indices], 'ro')

##################
pl.figure()
pl.title("Accumulated path")
pl.plot(t[1:], accumulated_path)
pl.plot(t[event_start_indices], accumulated_path[event_start_indices], 'ro')

##################
pl.figure()
pl.title("Accumulated angle")
pl.plot(t, accummulated_ang)
pl.plot(t[event_start_indices], accummulated_ang[event_start_indices], 'ro')

##################
swims_forward_densities, swims_forward_bins = np.histogram(bout_path_changes, bins=np.linspace(0, 100, 30), density=True)
pl.figure()
pl.title("Swim forward histogram densities")
pl.plot(swims_forward_bins[1:], swims_forward_densities)
print("Swims forward bins: ", swims_forward_bins[1:])
print("Swims forward densities: ", swims_forward_densities)
print("Median forward swims: ", np.nanmedian(np.abs(bout_path_changes)))
print("\n\n")

##################
turn_angle_densties, turn_angle_bins = np.histogram(bout_angle_changes, bins=np.linspace(-60, 60, 30), density=True)
pl.figure()
pl.title("Turn angle histogram densities")
pl.plot(turn_angle_bins[1:], turn_angle_densties)
print("Turn angle histogram bins: ", turn_angle_bins[1:])
print("Turn angle histogram densities: ", turn_angle_densties)
print("Median absolute turn angle: ", np.nanmedian(np.abs(bout_angle_changes)))
print("\n\n")

##################
center_distance_densities, center_distance_bins = np.histogram(center_distance, bins=np.linspace(0, 400, 30), density=True)
pl.figure()
pl.title("Center distance densities")
pl.plot(center_distance_bins[1:], center_distance_densities)
print("Center distance densities bins: ", center_distance_bins[1:])
print("Center distance densities: ", center_distance_densities)
print("Median distance to the center: ", np.nanmedian(center_distance))
print("\n\n")

##################
interbout_interval_densities,interbout_interval_bins = np.histogram(interbout_intervals, bins=np.linspace(0, 3, 30), density=True)
pl.figure()
pl.title("Interbout interval histograms")
pl.plot(interbout_interval_bins[1:], interbout_interval_densities)
print("Interbout interval bins:", interbout_interval_bins[1:])
print("Interbout interval densities: ", interbout_interval_densities)
print("Median Interbout interval:", np.nanmean(interbout_intervals))


pl.show()

dframe = DataFrame({"Swims forward bins: ": swims_forward_bins[1:],
                    "Swims forward densities: ": swims_forward_densities,

                    "Turn angle bins: ": turn_angle_bins[1:],
                    "Turn angle densities: ": turn_angle_densties,

                    'Center distance densities bins: ': center_distance_bins[1:],
                    'Center distance densities: ': center_distance_densities,

                    'Interbout interval bins:': interbout_interval_bins[1:],
                    'Interbout interval densities:': interbout_interval_densities})

dframe.to_excel('{}_histograms.xlsx'.format(fish_name), sheet_name='histograms', index=False)

dframe = DataFrame({"Median forward swims: ": [np.nanmedian(np.abs(bout_path_changes))],
                    "Median absolute turn angle: ": [np.nanmedian(np.abs(bout_angle_changes))],
                    "Median distance to the center: ": [np.nanmedian(center_distance)],
                    "Median Interbout interval: ": [np.nanmean(interbout_intervals)],
                    })

dframe.to_excel('{}_medians.xlsx'.format(fish_name), sheet_name='medians', index=False)



# make an excel file

