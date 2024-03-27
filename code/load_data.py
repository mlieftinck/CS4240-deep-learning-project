# from numba import jit, cuda
import os

from PIL import Image, ImageSequence
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.express as px

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
    # Upload images from training folder
    project_dir = os.path.dirname(os.getcwd())
    file_dir_train = project_dir + '/data/input/train/'
    file_dir_test = project_dir + '/data/labels/train/NSN/'

    batch_of_images = []
    batch_of_truths = []

    epochs_of_images = []
    epochs_of_truths = []

    train_directory = os.listdir(file_dir_train)
    train_directory.sort()
    test_directory = os.listdir(file_dir_test)
    test_directory.sort()

    for training_file in train_directory:
        img = Image.open(file_dir_train + training_file)

        # Empty list to read pixel values of original image
        array_of_image = []

        # Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
        for i in range(51):
            img.seek(i)
            array_img = np.array(img)
            array_of_image.append(array_img)

        array_of_image = np.array(array_of_image, dtype=float)
        array_of_image_tensor = torch.tensor(array_of_image)

        array_of_image_tensor = torch.unsqueeze(array_of_image_tensor, 0)
        array_of_image_tensor = torch.unsqueeze(array_of_image_tensor, 0)

        array_of_image_tensor_interpolated = torch.nn.functional.interpolate(array_of_image_tensor, size=(112, 114, 112),
                                                                         mode='trilinear')

        array_of_image_tensor_interpolated = torch.squeeze(array_of_image_tensor_interpolated, 0)
        array_of_image_tensor_interpolated = torch.squeeze(array_of_image_tensor_interpolated, 0)

        interpolated_image_normalized_torch = (array_of_image_tensor_interpolated - torch.min(
            array_of_image_tensor_interpolated)) / ((torch.max(array_of_image_tensor_interpolated) -
                                                     torch.min(array_of_image_tensor_interpolated)))

        batch_of_images.append(interpolated_image_normalized_torch)

        if len(batch_of_images) == 11:

            epochs_of_images.append(batch_of_images)
            batch_of_images = []

    epochs_of_images = np.array(epochs_of_images)
    epochs_of_images = torch.Tensor(epochs_of_images)

    for testing_file in os.listdir(file_dir_test):
        lab = Image.open(file_dir_test + testing_file)

        # Empty list to read pixel values of original image
        array_of_truth = []

        for i in range(51):
            lab.seek(i)
            array_lab = np.array(lab)
            array_of_truth.append(array_lab)

        array_of_truth = np.array(array_of_truth, dtype=float)
        array_of_truth_tensor = torch.tensor(array_of_truth)

        array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)
        array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)

        array_of_truth_tensor_interpolated = torch.nn.functional.interpolate(array_of_truth_tensor,
                                                                             size=(112, 114, 112),
                                                                             mode='nearest') / 255

        array_of_truth_tensor_interpolated = torch.squeeze(array_of_truth_tensor_interpolated, 0)
        array_of_truth_tensor_interpolated = torch.squeeze(array_of_truth_tensor_interpolated, 0)

        batch_of_truths.append(array_of_truth_tensor_interpolated)

        if len(batch_of_truths) == 11:
            epochs_of_truths.append(batch_of_truths)
            batch_of_truths = []

    epochs_of_truths = np.array(epochs_of_truths)
    epochs_of_truths = torch.Tensor(epochs_of_truths)

    # info_dict_img = {"Image Size": img.size, "Image Height": img.height, "Image Width": img.width,
    #                  "Frames in Image": getattr(img, "n_frames", 1)}
    # info_dict_lab = {"Image Size": lab.size, "Image Height": lab.height, "Image Width": lab.width,
    #                  "Frames in Image": getattr(lab, "n_frames", 1)}

    return epochs_of_images, epochs_of_truths

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

    fig = px.scatter_3d(values.flatten(), x=x_flat, y=y_flat, z=z_flat, color=values.flatten(), opacity=0.2)
    fig.update_traces(marker=dict(size=3))
    fig.show()
    #
    my_cmap = plt.get_cmap('binary')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sctt = ax.scatter3D(x_flat, y_flat, z_flat, s=0.1, c=values.flatten(), cmap=my_cmap, alpha=0.2)
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    plt.show()

if __name__ == "__main__":
    device = try_gpu()
    print(device)
    pic, truth = preprocessing()
    print(pic.size())
    print(truth.size())
    # plot(interpolated_image_normalized_torch)
