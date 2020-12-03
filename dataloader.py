import torch
import numpy as np
import cv2
import time
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")


class Dataset_3DCNN(Dataset):
    def __init__(self, list_ids):
        self.list_ids = list_ids

    def __len__(self):
        return len(self.list_ids)

    def get_images(self, index):
        # print('     indexing {}'.format(index))
        X = []
        for i in range(0, 25):
            im = cv2.imread('pre_processed_patches/{}/{}.jpg'.format(index, i))
            # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = torch.from_numpy(im).float().to(device)
            X.append(im)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):

        # Load data
        idx = self.list_ids[index]
        X = self.get_images(idx)  # (input) spatial images
        label = np.loadtxt('pre_processed_patches/{}/label.txt'.format(idx))[2]
        label = int(label)
        # print('label size {}'.format(label.shape))
        # time.sleep(30)
        y = torch.LongTensor(np.asarray([label]))  # (labels) LongTensor are for int64 instead of FloatTensor
        # print('label tensor shape {}'.format(y.size()))

        # print(X.shape)
        return X, y
