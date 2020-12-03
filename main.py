import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from dataloader import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from model import CNN3D

# set path
save_model_path = ""  # save Pytorch models

# 3D CNN parameters
fc_hidden1, fc_hidden2 = 256, 256
dropout = 0.0        # dropout probability

# training parameters
k = 3            # number of target category
epochs = 15
batch_size = 30
learning_rate = 1e-4
log_interval = 10
img_x, img_y = 52, 52  # resize video 2d frame size
t_dim = 25

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)
        # loss = F.cross_entropy(output, y)
        # time.sleep(30)
        loss = criterion(output, y)
        # print('after loss {}'.format(loss))
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


if __name__ == '__main__':
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    # load UCF101 actions names
    # params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # # example
    # y = ['HorseRace', 'YoYo', 'WalkingWithDog']
    # y_onehot = labels2onehot(enc, le, y)
    # y2 = onehot2labels(le, y_onehot)




    # list all data files
    # all_X_list = range(0, 1645)              # all video file names
    # all_y_list = range(0, 1645)    # all video labels

    # train, test split
    # train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.20, random_state=5)

    train_list = range(0, 2631)
    valid_list = range(2631, 2989)
    clean_test_list = range(2989, 3290)

    print(len(train_list))
    print(len(valid_list))

    train_set, valid_set = Dataset_3DCNN(train_list), Dataset_3DCNN(valid_list)
    train_loader = data.DataLoader(train_set, batch_size=30, shuffle=True, num_workers=4)
    valid_loader = data.DataLoader(valid_set, batch_size=30, shuffle=True, num_workers=4)

    print('Creating model')
    cnn3d = CNN3D(t_dim=t_dim, img_x=img_x, img_y=img_y,
                  drop_p=dropout, fc_hidden1=fc_hidden1,  fc_hidden2=fc_hidden2, num_classes=k).to(device)
    criterion = nn.CrossEntropyLoss()

    print('Parallelize model to multiple GPUs')
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn3d = nn.DataParallel(cnn3d)

    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=learning_rate)   # optimize all cnn parameters


    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    print('Training start')
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, cnn3d, device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation(cnn3d, device, optimizer, valid_loader)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save('./3DCNN_epoch_training_losses.npy', A)
        np.save('./3DCNN_epoch_training_scores.npy', B)
        np.save('./3DCNN_epoch_test_loss.npy', C)
        np.save('./3DCNN_epoch_test_score.npy', D)

        print('Currently at Epoch {}'.format(epoch))

    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    # plt.plot(histories.losses_val)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    title = "./fig_UCF101_3DCNN.png"
    plt.savefig(title, dpi=600)
    # plt.close(fig)
    plt.show()
