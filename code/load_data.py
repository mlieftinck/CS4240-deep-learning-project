# # from numba import jit, cuda
# import os
#
# from PIL import Image, ImageSequence
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import plotly.express as px
#
# def try_gpu():
#     """
#     If GPU is available, return torch.device as cuda:0; else return torch.device
#     as cpu.
#     """
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu')
#     return device
#
# def preprocessing(project_dir, model="QCANet"):
#     # Upload images from training folder
#     file_dir_train = project_dir + '/data/input/train/'
#     file_dir_test = project_dir + '/data/labels/train/' + model + '/'
#
#     batch_of_images = []
#     batch_of_truths = []
#
#     epochs_of_images = []
#     epochs_of_truths = []
#
#     train_directory = os.listdir(file_dir_train)
#     train_directory.sort()
#     test_directory = os.listdir(file_dir_test)
#     test_directory.sort()
#
#     for training_file in train_directory:
#         img = Image.open(file_dir_train + training_file)
#
#         # Empty list to read pixel values of original image
#         array_of_image = []
#
#         # Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
#         for i in range(51):
#             img.seek(i)
#             array_img = np.array(img)
#             array_of_image.append(array_img)
#
#         array_of_image = np.array(array_of_image, dtype=float)
#         array_of_image_tensor = torch.tensor(array_of_image)
#
#         array_of_image_tensor = torch.unsqueeze(array_of_image_tensor, 0)
#         array_of_image_tensor = torch.unsqueeze(array_of_image_tensor, 0)
#
#         array_of_image_tensor_interpolated = torch.nn.functional.interpolate(array_of_image_tensor,
#                                                                              size=(112, 114, 112),
#                                                                              mode='trilinear')
#
#         reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
#         array_of_image_tensor_interpolated_padded = reflection_pad_3d(array_of_image_tensor_interpolated)
#         array_of_image_tensor_interpolated_padded = torch.squeeze(array_of_image_tensor_interpolated_padded, 0)
#
#         interpolated_image_normalized_torch = (array_of_image_tensor_interpolated_padded - torch.min(
#             array_of_image_tensor_interpolated_padded)) / ((torch.max(array_of_image_tensor_interpolated_padded) -
#                                                             torch.min(array_of_image_tensor_interpolated_padded)))
#
#         batch_of_images.append(interpolated_image_normalized_torch)
#
#         if len(batch_of_images) == 11:
#             image_flipped_x_list = []
#             image_flipped_y_list = []
#             image_flipped_xy_list = []
#
#             for pre_flipped_image in batch_of_images:
#                 image_flipped_x = torch.flip(pre_flipped_image, [0])
#                 image_flipped_y = torch.flip(pre_flipped_image, [1])
#                 image_flipped_xy = torch.flip(image_flipped_x, [1])
#
#                 image_flipped_x_list.append(image_flipped_x)
#                 image_flipped_y_list.append(image_flipped_y)
#                 image_flipped_xy_list.append(image_flipped_xy)
#
#             epochs_of_images.append(batch_of_images)
#             epochs_of_images.append(image_flipped_x_list)
#             epochs_of_images.append(image_flipped_y_list)
#             epochs_of_images.append(image_flipped_xy_list)
#
#             batch_of_images = []
#             image_flipped_x_list = []
#             image_flipped_y_list = []
#             image_flipped_xy_list = []
#
#     epochs_of_images = np.array(epochs_of_images)
#     epochs_of_images = torch.Tensor(epochs_of_images)
#
#     for testing_file in test_directory:
#         lab = Image.open(file_dir_test + testing_file)
#
#         # Empty list to read pixel values of original image
#         array_of_truth = []
#
#         # Append all the pixel values of the image in to one (51, 114, 112) array (z, y, x)
#         for i in range(51):
#             lab.seek(i)
#             array_lab = np.array(lab)
#             array_of_truth.append(array_lab)
#
#         array_of_truth = np.array(array_of_truth, dtype=float)
#         array_of_truth_tensor = torch.tensor(array_of_truth)
#
#         array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)
#         array_of_truth_tensor = torch.unsqueeze(array_of_truth_tensor, 0)
#
#         array_of_truth_tensor_interpolated = torch.nn.functional.interpolate(array_of_truth_tensor,
#                                                                              size=(112, 114, 112),
#                                                                              mode='nearest') / 255
#
#         reflection_pad_3d = torch.nn.ReflectionPad3d((8, 8, 7, 7, 8, 8))
#         array_of_truth_tensor_interpolated_padded = reflection_pad_3d(array_of_truth_tensor_interpolated)
#         array_of_truth_tensor_interpolated_padded = torch.squeeze(array_of_truth_tensor_interpolated_padded, 0)
#         array_of_truth_tensor_interpolated_padded = torch.squeeze(array_of_truth_tensor_interpolated_padded, 0)
#
#         batch_of_truths.append(array_of_truth_tensor_interpolated_padded)
#
#         if len(batch_of_truths) == 11:
#             image_flipped_x_list = []
#             image_flipped_y_list = []
#             image_flipped_xy_list = []
#
#             for pre_flipped_image in batch_of_truths:
#                 image_flipped_x = torch.flip(pre_flipped_image, [0])
#                 image_flipped_y = torch.flip(pre_flipped_image, [1])
#                 image_flipped_xy = torch.flip(image_flipped_x, [1])
#
#                 image_flipped_x_list.append(image_flipped_x)
#                 image_flipped_y_list.append(image_flipped_y)
#                 image_flipped_xy_list.append(image_flipped_xy)
#
#             epochs_of_truths.append(batch_of_truths)
#             epochs_of_truths.append(image_flipped_x_list)
#             epochs_of_truths.append(image_flipped_y_list)
#             epochs_of_truths.append(image_flipped_xy_list)
#
#             batch_of_truths = []
#             image_flipped_x_list = []
#             image_flipped_y_list = []
#             image_flipped_xy_list = []
#
#     epochs_of_truths = np.array(epochs_of_truths)
#     epochs_of_truths = torch.Tensor(epochs_of_truths)
#
#     torch.save(epochs_of_images, '/data/preprocessed_data/epochs_of_images.pt')
#     torch.save(epochs_of_truths, '/Users/aman/Desktop/TUDelft Year 5/Deep Learning/Project/CS4240-deep-learning-project/data/preprocessed_data/epochs_of_truths_'+ model +'.pt')
#
#     return epochs_of_images, epochs_of_truths
#
# if __name__ == "__main__":
#     device = try_gpu()
#     print(device)
#     pic, truth = preprocessing(os.getcwd(), model="QCANet")
#     print(pic.size())
#     print(truth.size())