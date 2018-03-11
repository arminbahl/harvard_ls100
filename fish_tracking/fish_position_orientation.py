import imageio
import pylab as pl
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def get_fish_position_and_angle(background_substracted_image):
    fish_image = background_substracted_image.copy()


    pl.imshow(fish_image, cmap='gray')
    pl.show()

    # threshold
    fish_image[fish_image < 15] = 0
    fish_image[fish_image >= 15] = 255

    # apply a weak gaussian blur to the thresholded image to get rid of noisy pixels
    fish_image = gaussian_filter(fish_image, (2, 2))

    pl.imshow(fish_image, cmap='gray')
    pl.show()

    # find the position of the maximum with
    x,y = np.unravel_index(np.argmax(fish_image), fish_image.shape)

    # cut out a square
    fish_roi = fish_image[x-50:x+50, y-50:y+50]

    # threshold again
    fish_roi[fish_roi < 15] = 0
    fish_roi[fish_roi >= 15] = 1

    pl.imshow(fish_roi, cmap='gray')
    pl.show()

    # translate the image pixels into a vector of x,y
    coordinates_original = np.array(np.where(fish_roi == 1))

    # move these pixels to the center
    coordinates_moved = coordinates_original.copy()
    coordinates_moved[0] = coordinates_moved[0] - np.mean(coordinates_moved[0])
    coordinates_moved[1] = coordinates_moved[1] - np.mean(coordinates_moved[1])

    # the eigenvectors of the covariance matrix determine the axes of a coordinate system along the largest/smallest variance
    cov = np.cov(coordinates_moved)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    # what is the angle of largest axes relative to our original axis? - this is the fish orienteation
    fish_angle = np.tanh((x_v1)/(y_v1))

    # create a rotation matrix with that angle
    rotation_mat = np.matrix([[np.cos(fish_angle), -np.sin(fish_angle)],
                              [np.sin(fish_angle), np.cos(fish_angle)]])


    # rotate the mean-centered fish points according to that angle
    coordinates_rotated = np.array(rotation_mat * coordinates_moved)

    ################
    # display the original fish and center moved one
    pl.figure()

    # original fish coordinates
    pl.plot(coordinates_original[0], coordinates_original[1], '.')

    # center-moves corrdinates
    pl.plot(coordinates_moved[0], coordinates_moved[1], '.')

    # the principle axes of the new coordinate system
    scale = 20
    pl.plot([x_v1*-scale*2, x_v1*scale*2],
             [y_v1*-scale*2, y_v1*scale*2], color='red')

    pl.plot([x_v2*-scale, x_v2*scale],
             [y_v2*-scale, y_v2*scale], color='blue')

    pl.show()

    ################
    # Display the rotated fish
    pl.figure()

    # if more dots are negative than positive in the rotated fish, the head looks downwards.
    # In that case add 180 degrees to the fish orientation
    ind_positive = (coordinates_rotated[1] > 0)
    ind_negative = (coordinates_rotated[1] < 0)

    if ind_negative.sum() > ind_positive.sum():
        fish_angle += np.pi

    pl.plot(coordinates_rotated[0], coordinates_rotated[1], '.')
    pl.plot(coordinates_rotated[0][ind_positive], coordinates_rotated[1][ind_positive], '+')  # color the positive ones
    pl.plot(coordinates_rotated[0][ind_negative], coordinates_rotated[1][ind_negative], '_')  # color the negative ones
    pl.show()

    ################
    # make a new figure to display angle and head
    x = coordinates_original[0].mean()
    y = coordinates_original[1].mean()

    dx = np.sin(fish_angle)*scale*2
    dy = np.cos(fish_angle)*scale*2

    # original fish coordinates
    pl.figure()
    pl.plot(coordinates_original[0], coordinates_original[1], '.')
    pl.plot([x], [y], 'o')
    pl.plot([x+dx], [y+dy], 'o')
    pl.plot([x, x+dx], [y, y+dy])
    #pl.show()

    return x, y, fish_angle

# load the background subtracted image (or loop through a movie)
print("Analyzing fish.....")

fish_image = imageio.imread("fish.png")[:,:,0]
x, y, fish_angle = get_fish_position_and_angle(fish_image)

print("Fish position in image: ", (x, y))
print("Fish angle in image (in degrees): ", fish_angle * 180/np.pi)

pl.show()