import os
from PIL import Image
import numpy as np
import torch
import time


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
    start_time = time.time()
    slices_per_image = 51
    images_per_timeseries = 11
    project_dir = os.getcwd()

    # Upload images from data folder
    data_path = os.path.join(project_dir, source_folder)

    image_batches = []
    image_timeseries = []
    timeseries_batches = []

    data_dir = os.listdir(data_path)
    data_dir.sort()

    for file in data_dir:
        img = Image.open(data_path + "/" + file)

        # Empty list to read pixel values of original image
        image_list = []

        # Append all the pixel values of the image into one (51, 114, 112) array (z, y, x)
        for i in range(slices_per_image):
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

        if len(image_timeseries) == images_per_timeseries:
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

            timeseries_batches.append(image_timeseries)
            timeseries_batches.append(image_timeseries_flipped_x)
            timeseries_batches.append(image_timeseries_flipped_y)
            timeseries_batches.append(image_timeseries_flipped_xy)
            image_timeseries = []

    timeseries_batches = np.array(timeseries_batches)
    timeseries_batches = torch.Tensor(timeseries_batches)

    for i in range(len(timeseries_batches)):
        for j in range(len(timeseries_batches[i])):
            if i < len(timeseries_batches) // 4:
                torch.save(timeseries_batches[i][j],
                           project_dir + "/preprocessed_data2" + source_folder[4:] + "/" + data_dir[i] + ".pt")
            elif i < 2 * len(timeseries_batches) // 4:
                torch.save(timeseries_batches[i][j],
                           project_dir + "/preprocessed_data2" + source_folder[4:] + "/" + data_dir[i] + "_flip_x.pt")
            elif i < 3 * len(timeseries_batches) // 4:
                torch.save(timeseries_batches[i][j],
                           project_dir + "/preprocessed_data2" + source_folder[4:] + "/" + data_dir[i] + "_flip_y.pt")
            else:
                torch.save(timeseries_batches[i][j],
                           project_dir + "/preprocessed_data2" + source_folder[4:] + "/" + data_dir[i] + "_flip_xy.pt")

    # os.makedirs(project_dir + "/preprocessed_data/" + source_folder[4:] + "/", exist_ok=True)
    # torch.save(timeseries_batches, project_dir + "/preprocessed_data" + source_folder[4:] + ".pt")

    return time.time() - start_time


if __name__ == "__main__":
    device = try_gpu()
    print(device)
    os.chdir('..')
    t = preprocessing("data/input/test")
    print(f"1/6 completed in: {t // 60:.0f} min {t % 60:.0f} sec")
    t += preprocessing("data/input/train")
    print(f"2/6 completed in: {t // 60:.0f} min {t % 60:.0f} sec")
    t += preprocessing("data/labels/test/QCANet")
    print(f"3/6 completed in: {t // 60:.0f} min {t % 60:.0f} sec")
    t += preprocessing("data/labels/train/NDN")
    print(f"4/6 completed in: {t // 60:.0f} min {t % 60:.0f} sec")
    t += preprocessing("data/labels/train/NSN")
    print(f"5/6 completed in: {t // 60:.0f} min {t % 60:.0f} sec")
    t += preprocessing("data/labels/train/QCANet")
    print(f"6/6 completed in: {t // 60:.0f} min {t % 60:.0f} sec")
