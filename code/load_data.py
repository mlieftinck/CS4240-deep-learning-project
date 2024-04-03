import os
from PIL import Image
import numpy as np
import torch


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


def preprocessing(source_folder="data/input/train"):
    i = source_folder[4:]
    os.chdir('..')
    project_dir = os.getcwd()

    # Upload images from data folder
    data_path = os.path.join(project_dir, source_folder)
    # file_dir_test = os.path.join(project_dir, 'data/labels/train/' + model + '/')

    image_timeseries = []
    batch_of_truths = []

    batches = []
    epochs_of_truths = []

    data_dir = os.listdir(data_path)
    data_dir.sort()

    for file in data_dir:
        img = Image.open(data_path + "/" + file)

        # Empty list to read pixel values of original image
        image_list = []

        # Append all the pixel values of the image into one (51, 114, 112) array (z, y, x)
        for i in range(51):
            img.seek(i)
            array_img = np.array(img)
            image_list.append(array_img)

        image_array = np.array(image_list, dtype=float)
        image_tensor = torch.tensor(image_array)

        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        image_tensor_interpolated = torch.nn.functional.interpolate(image_tensor,
                                                                    size=(112, 114, 112),
                                                                    mode='trilinear')

        reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
        image_tensor_interpolated_padded = reflection_pad_3d(image_tensor_interpolated)
        image_tensor_interpolated_padded = torch.squeeze(image_tensor_interpolated_padded, 0)

        image_tensor_normalized = (image_tensor_interpolated_padded - torch.min(
            image_tensor_interpolated_padded)) / ((torch.max(image_tensor_interpolated_padded) -
                                                   torch.min(image_tensor_interpolated_padded)))

        image_timeseries.append(image_tensor_normalized)

        print("pre-flip finished!")
        if len(image_timeseries) == 11:
            image_timeseries_flipped_x = []
            image_timeseries_flipped_y = []
            image_timeseries_flipped_xy = []

            for pre_flipped_image in image_timeseries:
                image_flipped_x = torch.flip(pre_flipped_image, [0])
                image_flipped_y = torch.flip(pre_flipped_image, [1])
                image_flipped_xy = torch.flip(image_flipped_x, [1])

                image_timeseries_flipped_x.append(image_flipped_x)
                image_timeseries_flipped_y.append(image_flipped_y)
                image_timeseries_flipped_xy.append(image_flipped_xy)

            batches.append(image_timeseries)
            batches.append(image_timeseries_flipped_x)
            batches.append(image_timeseries_flipped_y)
            batches.append(image_timeseries_flipped_xy)

            image_timeseries = []
            image_timeseries_flipped_x = []
            image_timeseries_flipped_y = []
            image_timeseries_flipped_xy = []

    batches = np.array(batches)
    batches = torch.Tensor(batches)

    # for testing_file in test_directory:
    #     lab = Image.open(file_dir_test + testing_file)
    #
    #     # Empty list to read pixel values of original image
    #     array_of_truth = []
    #
    #     # Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
    #     for i in range(51):
    #         lab.seek(i)
    #         array_lab = np.array(lab)
    #         array_of_truth.append(array_lab)
    #
    #     array_of_truth = np.array(array_of_truth, dtype=float)
    #     array_of_truth_tensor = torch.tensor(array_of_truth)
    #
    #     array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)
    #     array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)
    #
    #     array_of_truth_tensor_interpolated = torch.nn.functional.interpolate(array_of_truth_tensor,
    #                                                                          size=(112, 114, 112),
    #                                                                          mode='nearest') / 255
    #
    #     reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
    #     array_of_truth_tensor_interpolated_padded = reflection_pad_3d(array_of_truth_tensor_interpolated)
    #     array_of_truth_tensor_interpolated_padded = torch.squeeze(array_of_truth_tensor_interpolated_padded, 0)
    #     array_of_truth_tensor_interpolated_padded = torch.squeeze(array_of_truth_tensor_interpolated_padded, 0)
    #
    #     batch_of_truths.append(array_of_truth_tensor_interpolated_padded)
    #
    #     if len(batch_of_truths) == 11:
    #         image_flipped_x_list = []
    #         image_flipped_y_list = []
    #         image_flipped_xy_list = []
    #
    #         for pre_flipped_image in batch_of_truths:
    #             image_flipped_x = torch.flip(pre_flipped_image, [0])
    #             image_flipped_y = torch.flip(pre_flipped_image, [1])
    #             image_flipped_xy = torch.flip(image_flipped_x, [1])
    #
    #             image_flipped_x_list.append(image_flipped_x)
    #             image_flipped_y_list.append(image_flipped_y)
    #             image_flipped_xy_list.append(image_flipped_xy)
    #
    #         epochs_of_truths.append(batch_of_truths)
    #         epochs_of_truths.append(image_flipped_x_list)
    #         epochs_of_truths.append(image_flipped_y_list)
    #         epochs_of_truths.append(image_flipped_xy_list)
    #
    #         batch_of_truths = []
    #         image_flipped_x_list = []
    #         image_flipped_y_list = []
    #         image_flipped_xy_list = []
    #
    # epochs_of_truths = np.array(epochs_of_truths)
    # epochs_of_truths = torch.Tensor(epochs_of_truths)

    os.makedirs(project_dir + "/preprocessed_data/" + source_folder[4:] + "/", exist_ok=True)
    torch.save(batches, "/preprocessed_data" + source_folder[4:] + ".pt")
    # torch.save(epochs_of_truths,
    #            '/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/preprocessed_data/epochs_of_truths_' + model + '.pt')

    return batches


if __name__ == "__main__":
    device = try_gpu()
    print(device)
    batch = preprocessing()
    print(batch.size())
