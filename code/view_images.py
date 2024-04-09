import napari
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def view_image_napari(input_array):
    viewer = napari.view_image(input_array, colormap='gray')
    napari.run()


def watershed(nsn, ndn):
    nsn_np = np.array(nsn)
    ndn_np = np.array(ndn)
    distance = ndi.distance_transform_edt(nsn_np)
    markers, _ = ndi.label(nsn_np)
    wsimage = watershed(-distance, markers, mask=nsn_np)
    return wsimage


def plot_3D_image(wsimage):
    num_labels = np.max(wsimage)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Loop over each z-stack
    for i in range(wsimage.shape[0]):
        x = np.arange(0, wsimage.shape[1])
        y = np.arange(0, wsimage.shape[2])
        X, Y = np.meshgrid(x, y)

        colors = plt.cm.nipy_spectral(wsimage[i] / num_labels)
        colors[:, :, 3] = np.where(wsimage[i] == 0, 0, 0.5)

        # Plot the segmented image for the current z-stack
        ax.plot_surface(X, Y, np.full_like(X, i), rstride=1, cstride=1, facecolors=colors, shade=True)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Segmented Images')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    out_nsn = torch.load('/kaggle/input/a-lot-of-cells/nsn_test_t_11_not_flipped.pth')
    w = watershed(out_nsn, out_nsn)
    plot_3D_image(w)