# from numba import jit, cuda
import os

from PIL import Image, ImageSequence
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from dash import Dash, dcc, html

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

def preprocessing(project_dir, model):
    os.chdir('..')

    # Upload images from training folder
    file_dir_train = os.getcwd() + '/data/input/train/'
    file_dir_label = os.getcwd() + '/data/labels/train/' + model + '/'
    file_dir_test = os.getcwd() + '/data/input/test/'
    file_dir_test_label = os.getcwd() + '/data/labels/test/QCANet/'

    batch_of_images = []
    batch_of_truths = []
    batch_of_test_images = []
    batch_of_test_labels = []

    train_directory = os.listdir(file_dir_train)
    train_directory.sort()
    label_directory = os.listdir(file_dir_label)
    label_directory.sort()
    test_directory = os.listdir(file_dir_test)
    test_directory.sort()
    test_label_directory = os.listdir(file_dir_test_label)
    test_label_directory.sort()

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

        array_of_image_tensor_interpolated = torch.nn.functional.interpolate(array_of_image_tensor,
                                                                             size=(112, 114, 112),
                                                                             mode='trilinear')

        reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
        array_of_image_tensor_interpolated_padded = reflection_pad_3d(array_of_image_tensor_interpolated)

        array_of_image_tensor_interpolated_padded = torch.squeeze(array_of_image_tensor_interpolated_padded, 0)
        array_of_image_tensor_interpolated_padded = torch.squeeze(array_of_image_tensor_interpolated_padded, 0)

        interpolated_image_normalized_torch = (array_of_image_tensor_interpolated_padded - torch.min(
            array_of_image_tensor_interpolated_padded)) / ((torch.max(array_of_image_tensor_interpolated_padded) -
                                                            torch.min(array_of_image_tensor_interpolated_padded)))

        image_flipped_x = torch.flip(interpolated_image_normalized_torch, [0])
        image_flipped_y = torch.flip(interpolated_image_normalized_torch, [1])
        image_flipped_xy = torch.flip(image_flipped_x, [1])

        interpolated_image_normalized_torch = torch.unsqueeze(interpolated_image_normalized_torch, 0)
        image_flipped_x = torch.unsqueeze(image_flipped_x, 0)
        image_flipped_y = torch.unsqueeze(image_flipped_y, 0)
        image_flipped_xy = torch.unsqueeze(image_flipped_xy, 0)

        batch_of_images.append(interpolated_image_normalized_torch)
        batch_of_images.append(image_flipped_x)
        batch_of_images.append(image_flipped_y)
        batch_of_images.append(image_flipped_xy)

    batch_of_images = np.array(batch_of_images)
    batch_of_images = torch.Tensor(batch_of_images)

    for label_file in label_directory:
        lab = Image.open(file_dir_label + label_file)

        # Empty list to read pixel values of original image
        array_of_truth = []

        # Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
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

        reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
        array_of_truth_tensor_interpolated_padded = reflection_pad_3d(array_of_truth_tensor_interpolated)

        array_of_truth_tensor_interpolated_padded = torch.squeeze(array_of_truth_tensor_interpolated_padded, 0)
        array_of_truth_tensor_interpolated_padded = torch.squeeze(array_of_truth_tensor_interpolated_padded, 0)

        image_flipped_x = torch.flip(array_of_truth_tensor_interpolated_padded, [0])
        image_flipped_y = torch.flip(array_of_truth_tensor_interpolated_padded, [1])
        image_flipped_xy = torch.flip(image_flipped_x, [1])

        batch_of_truths.append(array_of_truth_tensor_interpolated_padded)
        batch_of_truths.append(image_flipped_x)
        batch_of_truths.append(image_flipped_y)
        batch_of_truths.append(image_flipped_xy)

    batch_of_truths = np.array(batch_of_truths)
    batch_of_truths = torch.Tensor(batch_of_truths)

    for test_file in test_directory:
        lab = Image.open(file_dir_test + test_file)

        # Empty list to read pixel values of original image
        array_of_test = []

        # Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
        for i in range(51):
            lab.seek(i)
            array_lab = np.array(lab)
            array_of_test.append(array_lab)

        array_of_test = np.array(array_of_test, dtype=float)
        array_of_test_tensor = torch.tensor(array_of_test)

        array_of_test_tensor = torch.unsqueeze(array_of_test_tensor, 0)
        array_of_test_tensor = torch.unsqueeze(array_of_test_tensor, 0)

        array_of_test_tensor_interpolated = torch.nn.functional.interpolate(array_of_test_tensor,
                                                                             size=(112, 114, 112),
                                                                             mode='nearest')

        reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
        array_of_test_tensor_interpolated_padded = reflection_pad_3d(array_of_test_tensor_interpolated)

        array_of_test_tensor_interpolated_padded = torch.squeeze(array_of_test_tensor_interpolated_padded, 0)
        array_of_test_tensor_interpolated_padded = torch.squeeze(array_of_test_tensor_interpolated_padded, 0)

        interpolated_test_normalized_torch = (array_of_test_tensor_interpolated_padded - torch.min(
            array_of_test_tensor_interpolated_padded)) / ((torch.max(array_of_test_tensor_interpolated_padded) -
                                                            torch.min(array_of_test_tensor_interpolated_padded)))

        image_flipped_x = torch.flip(interpolated_test_normalized_torch, [0])
        image_flipped_y = torch.flip(interpolated_test_normalized_torch, [1])
        image_flipped_xy = torch.flip(image_flipped_x, [1])

        interpolated_image_normalized_torch = torch.unsqueeze(interpolated_test_normalized_torch, 0)
        image_flipped_x = torch.unsqueeze(image_flipped_x, 0)
        image_flipped_y = torch.unsqueeze(image_flipped_y, 0)
        image_flipped_xy = torch.unsqueeze(image_flipped_xy, 0)

        batch_of_test_images.append(interpolated_image_normalized_torch)
        batch_of_test_images.append(image_flipped_x)
        batch_of_test_images.append(image_flipped_y)
        batch_of_test_images.append(image_flipped_xy)

    batch_of_test_images = np.array(batch_of_test_images)
    batch_of_test_images = torch.Tensor(batch_of_test_images)

    for test_label_file in test_label_directory:
        lab = Image.open(file_dir_test_label + test_label_file)

        # Empty list to read pixel values of original image
        array_of_test_labels = []

        # Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
        for i in range(51):
            lab.seek(i)
            array_lab = np.array(lab)
            array_of_test_labels.append(array_lab)

        array_of_test_labels = np.array(array_of_test_labels, dtype=float)
        array_of_test_labels_tensor = torch.tensor(array_of_test_labels)

        array_of_test_labels_tensor = torch.unsqueeze(array_of_test_labels_tensor, 0)
        array_of_test_labels_tensor = torch.unsqueeze(array_of_test_labels_tensor, 0)

        array_of_test_labels_tensor_interpolated = torch.nn.functional.interpolate(array_of_test_labels_tensor,
                                                                             size=(112, 114, 112),
                                                                             mode='nearest') / 255

        reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
        array_of_test_labels_tensor_interpolated_padded = reflection_pad_3d(array_of_test_labels_tensor_interpolated)

        array_of_test_labels_tensor_interpolated_padded = torch.squeeze(array_of_test_labels_tensor_interpolated_padded, 0)
        array_of_test_labels_tensor_interpolated_padded = torch.squeeze(array_of_test_labels_tensor_interpolated_padded, 0)

        image_flipped_x = torch.flip(array_of_test_labels_tensor_interpolated_padded, [0])
        image_flipped_y = torch.flip(array_of_test_labels_tensor_interpolated_padded, [1])
        image_flipped_xy = torch.flip(image_flipped_x, [1])

        batch_of_test_labels.append(array_of_test_labels_tensor_interpolated_padded)
        batch_of_test_labels.append(image_flipped_x)
        batch_of_test_labels.append(image_flipped_y)
        batch_of_test_labels.append(image_flipped_xy)

    batch_of_test_labels = np.array(batch_of_test_labels)
    batch_of_test_labels = torch.Tensor(batch_of_test_labels)

    torch.save(batch_of_images, '/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/preprocessed_data/batch_of_images.pt')
    torch.save(batch_of_truths, '/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/preprocessed_data/batch_of_truths_'+ model +'.pt')
    torch.save(batch_of_test_images, '/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/preprocessed_data/batch_of_test_images.pt')
    torch.save(batch_of_test_labels, '/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/preprocessed_data/batch_of_test_image_labels.pt')

    return batch_of_images, batch_of_truths, batch_of_test_images, batch_of_test_labels

if __name__ == "__main__":
    device = try_gpu()
    print(device)
    pic, truth, test, test_labels = preprocessing(os.getcwd(), model="QCANet")

    print(pic.size())
    print(truth.size())
    print(test.size())
    print(test_labels.size())
