from PIL import Image, ImageSequence
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os

#Upload images from training folder
img = Image.open('/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/input/test/Emb1_t001.tif')
lab = Image.open('/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/labels/test/QCANet/Emb1_t001.tif')
info_dict = {"Image Size": img.size, "Image Height": img.height, "Image Width": img.width, "Frames in Image": getattr(img, "n_frames", 1)}

#Print some information about the file that has just been uploaded
print(info_dict)

#Empty list to read pixel values of original image
array_of_image = []

#Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
for i in range(51):
    img.seek(i)
    lab.seek(i)
    array_img = np.array(img)
    array_lab = np.array(lab)
    array_of_image.append(array_img)
    # print(i, array_img.min(), array_img.max(), array_lab.min(), array_lab.max())

array_of_image = np.array(array_of_image)

#Empty list to append final bicubically interpolated version of image
interpolated_image = []

for i in range(112): #loop in x direction
    for j in range(114): #loop in y direction
        z_holder = []  #empty list for list of points in z direction

        for k in range(51): #loop in z direction
            z_holder.append(array_of_image[k][j][i]) #(z, y, x)

        # empty list for interpolated version of z direction
        interpolated_z = []

        # make spline of the original 51 points, then interpolate to produce 112 points
        spl = CubicSpline(np.arange(0, 51), z_holder)
        new_z_length = np.linspace(0, 50, 112)
        interpolated_z = spl(new_z_length)

        interpolated_image.append(interpolated_z)

# Final array of image is 12768x112, that needs to be reshaped to 112 x 114 x 112 (x, y, z)
interpolated_image = np.array(interpolated_image).reshape((112, 114, 112)) # i, j, k or x, y, z
# Normalize the values based on I = (I - I_min) / (I_max - I_min)
interpolated_image_normalized = (interpolated_image - np.min(interpolated_image)) / ((np.max(interpolated_image) - np.min(interpolated_image)))

