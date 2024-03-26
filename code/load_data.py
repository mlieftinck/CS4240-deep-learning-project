from numba import jit, cuda
from PIL import Image, ImageSequence
import numpy as np
from scipy.interpolate import CubicSpline
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import holoviews as hv
hv.extension('plotly')

def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def preprocessing():
    #Upload images from training folder
    img = Image.open('/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/input/train/Emb01_t001.tif')
    lab = Image.open('/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/labels/train/NSN/Emb01_t001.tif')
    info_dict_img = {"Image Size": img.size, "Image Height": img.height, "Image Width": img.width, "Frames in Image": getattr(img, "n_frames", 1)}
    info_dict_lab = {"Image Size": lab.size, "Image Height": lab.height, "Image Width": lab.width, "Frames in Image": getattr(lab, "n_frames", 1)}

    #Print some information about the file that has just been uploaded
    print(info_dict_img)
    print(info_dict_lab)

    #Empty list to read pixel values of original image
    array_of_image = []
    array_of_truth = []

    #Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
    for i in range(51):
        img.seek(i)
        lab.seek(i)
        array_img = np.array(img)
        array_lab = np.array(lab)
        array_of_image.append(array_img)
        array_of_truth.append(array_lab)
        # print(i, array_img.min(), array_img.max(), array_lab.min(), array_lab.max())

    array_of_image = np.array(array_of_image, dtype=float)
    array_of_image_tensor = torch.tensor(array_of_image)

    array_of_truth = np.array(array_of_truth, dtype=float)
    array_of_truth_tensor = torch.tensor(array_of_truth)

    array_of_image_tensor = torch.unsqueeze(array_of_image_tensor, 0)
    array_of_image_tensor = torch.unsqueeze(array_of_image_tensor, 0)
    array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)
    array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)

    array_of_image_tensor_interpolated = torch.nn.functional.interpolate(array_of_image_tensor, size=(112, 114, 112), mode='trilinear')
    array_of_truth_tensor_interpolated = torch.nn.functional.interpolate(array_of_truth_tensor, size=(112, 114, 112), mode='nearest')

    print(array_of_image_tensor_interpolated.size())
    print(array_of_truth_tensor_interpolated.size())

    z_holder = []  # empty list for list of points in z direction
    interpolated_image_loop = np.empty([112, 114, 112]) #Empty list to append final bicubically interpolated version of image

    for i in range(112): #loop in x direction
        for j in range(114): #loop in y direction
            for k in range(51): #loop in z direction
                z_holder.append(array_of_image[k][j][i]) #(z, y, x)

                if len(z_holder) == 51:
                    # make spline of the original 51 points, then interpolate to produce 112 points
                    spl = CubicSpline(np.arange(0, 51), z_holder)
                    new_z_length = np.linspace(0, 51, 112)
                    interpolated_z = spl(new_z_length)

                    interpolated_image_loop[i,j,:]= interpolated_z

                    z_holder = []  # empty list for list of points in z direction

    # Normalize the values based on I = (I - I_min) / (I_max - I_min)
    interpolated_image_normalized_loop = (interpolated_image_loop - np.min(interpolated_image_loop)) / ((np.max(interpolated_image_loop) - np.min(interpolated_image_loop)))
    interpolated_image_normalized_torch = (array_of_image_tensor_interpolated - torch.min(array_of_image_tensor_interpolated)) / ((torch.max(array_of_image_tensor_interpolated) - torch.min(array_of_image_tensor_interpolated)))

    return interpolated_image_normalized_torch, interpolated_image_normalized_loop
def plot(image):
    # Now we still need to pad the image with 64 voxels... huh?????
    x = np.linspace(0, 112, 112)
    y = np.linspace(0, 114, 114)
    z = np.linspace(0, 112, 112)
    X, Y, Z = np.meshgrid(x, y, z)

    # Reshape to get a list of coordinates
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    values = image

    fig = px.scatter_3d(values.flatten(), x=x_flat, y=y_flat, z=z_flat, color=values.flatten(), opacity=1)
    fig.update_traces(marker=dict(size=3))
    fig.show()
    #
    my_cmap = plt.get_cmap('binary')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sctt = ax.scatter3D(x_flat, y_flat, z_flat, s=0.1, c=values.flatten(), cmap=my_cmap)
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    plt.show()

device = try_gpu()
print(device)
interpolated_image_normalized_torch, interpolated_image_normalized_loop = preprocessing()
# plot(interpolated_image_normalized_torch)