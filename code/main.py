# import torch
# from train import train, test
# from model import nsn, ndn, mini_ndn
# import torch.nn as nn
# # from load_data import preprocessing
# from view_images import view_image
# from loss import DiceLoss
# import time
# import os
#
#
# def run_training(images, labels, test_images, test_labels, model="NSN", save=False):
#     print(f'------------ RUNNING ------------')
#     NDN = False
#     if model == "NDN":
#         NDN = True
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     if NDN:
#         net = ndn().to(device)
#         optimizer = torch.optim.Adam(net.parameters(), lr=5e-1)
#     else:
#         net = nsn().to(device)
#         optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
#     criterion = DiceLoss()
#
#     time_start = time.time()
#     train_loss, train_acc = train(images, labels, net, optimizer, criterion, device)
#     #     test_loss, test_acc, output_array = test(images, labels, net, criterion)
#     time_end = time.time() - time_start
#
#     print(f'test loss: {train_loss}, test accuracy = {train_acc}')
#     print(f"Total elapsed time: {time_end // 60:.0f} min {time_end % 60:.0f} sec")
#
#     # Save trained model
#     if save:
#         parent_directory = os.path.dirname(os.getcwd())
#         os.makedirs(parent_directory + '/working/trained_models/', exist_ok=True)
#         torch.save(net.state_dict(), parent_directory + '/working/trained_models/' + model + '.pth')
#     return train_loss, train_acc
#
#
# def load_trained_model(model, model_name):
#     parent_dir = os.path.dirname(os.getcwd())
#     model.load_state_dict(torch.load(parent_dir + '/trained_models/' + model_name))
#
#
# def run_training_mini_ndn():
#     images, labels = preprocessing(os.getcwd(), "NDN")
#     print(f'------------ RUNNING ------------')
#     images = images.float()
#     labels = labels.long()
#     net = mini_ndn()
#     optimizer = torch.optim.Adam(net.parameters(), lr=5e-1)
#     criterion = DiceLoss()
#
#     time_start = time.time()
#     train_loss, train_acc = train(images, labels, net, optimizer, criterion)
#     test_loss, test_acc, output_array = test(images, labels, net, criterion)
#     time_end = time.time() - time_start
#
#     print(output_array.size())
#     print(f'test loss: {test_loss}, test accuracy = {test_acc}')
#     print(f"Total elapsed time: {time_end // 60:.0f} min {time_end % 60:.0f} sec")
#
#     view_image(output_array.cpu().detach().numpy()[0])
#
#     return output_array.cpu().detach().numpy()
#
#
# if __name__ == "__main__":
#     o = run_training(os.getcwd())
#     view_image(o[0])
