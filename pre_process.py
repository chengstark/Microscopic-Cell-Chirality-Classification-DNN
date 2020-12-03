import cv2
import numpy as np
import os
from PIL import ImageOps
from PIL import Image


folder_id_counter = 0


def get_patch_from_frame():
    for i in range(1, 33):
        if i < 10:
            idx = '0{}'.format(i)
        else:
            idx = i
        os.mkdir('patches/patch_{}'.format(idx))
        label = np.loadtxt('const_index/lab/{}.txt'.format(idx))
        for cell_id in range(label.shape[0]):
            x, y, l = label[cell_id]
            w = 52
            h = 52
            for frame_id in range(0, 25):
                frame = cv2.imread('/Users/cheng_stark/tmp/Results_with_SEG_05062019/XY{}_video/frame{}.jpg'.format(idx, frame_id))
                # frame = cv2.imread('/Users/cheng_stark/Desktop/a.jpg')
                cell_patch = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                # cell_patch = frame
                # cell_patch = cv2.cvtColor(cell_patch, cv2.COLOR_RGB2GRAY)
                cv2.imwrite('patches/patch_{}/{}_{}.jpg'.format(idx, frame_id, cell_id), cell_patch)
                print('Processed video {} cell {} {}'.format(idx, frame_id, cell_id))


def organize_cells():
    global folder_id_counter
    for i in range(1, 33):
        if i < 10:
            idx = '0{}'.format(i)
        else:
            idx = i
        label = np.loadtxt('const_index/lab/{}.txt'.format(idx))
        for cell_id in range(label.shape[0]):
            single_cell_patches = []
            for frame_id in range(0, 25):
                im = cv2.imread('patches/patch_{}/{}_{}.jpg'.format(idx, frame_id, cell_id))
                single_cell_patches.append(im)
            l = label[cell_id]
            l = np.asarray([l])
            os.mkdir('pre_processed_patches/{}'.format(folder_id_counter))
            frame_idx = 0
            for img in single_cell_patches:
                cv2.imwrite('pre_processed_patches/{}/{}.jpg'.format(folder_id_counter, frame_idx), img)
                frame_idx += 1
            np.savetxt('pre_processed_patches/{}/label.txt'.format(folder_id_counter), l)
            folder_id_counter += 1
            print('Finished processing cell {}'.format(folder_id_counter))


def aug():
    folder_counter = 1645
    for i in range(1645):
        frames = []
        for idx in range(25):
            im = cv2.imread('pre_processed_patches/{}/{}.jpg'.format(i, idx))
            frames.insert(0, im)
        label = np.loadtxt('pre_processed_patches/{}/label.txt'.format(i))
        print(label.shape)
        if label[2] == 1:
            label[2] = 0
        elif label[2] == 0:
            label[2] = 1
        os.mkdir('pre_processed_patches/{}'.format(folder_counter))
        label = label.flatten()
        print(label.shape)
        np.savetxt('pre_processed_patches/{}/label.txt'.format(folder_counter), label)
        l2 = np.loadtxt('pre_processed_patches/{}/label.txt'.format(folder_counter))
        print(l2.shape)

        frame_id = 0
        for im in frames:
            cv2.imwrite('pre_processed_patches/{}/{}.jpg'.format(folder_counter, frame_id), im)
            frame_id += 1
        print('Auged {}'.format(folder_counter))
        folder_counter += 1


def get_four_digit_name(index):
    index_ = '{}'.format(index)
    ret_index = None
    if len(index_) == 1:
        ret_index = '000{}'.format(index_)
    elif len(index_) == 2:
        ret_index = '00{}'.format(index_)
    elif len(index_) == 3:
        ret_index = '0{}'.format(index_)
    elif len(index_) == 4:
        ret_index = '{}'.format(index_)

    return ret_index


def req_folder_arrange():
    os.mkdir('req/train')
    os.mkdir('req/train/cw')
    os.mkdir('req/train/ccw')
    os.mkdir('req/train/oth')

    os.mkdir('req/valid')
    os.mkdir('req/valid/cw')
    os.mkdir('req/valid/ccw')
    os.mkdir('req/valid/oth')

    os.mkdir('req/test')
    os.mkdir('req/test/cw')
    os.mkdir('req/test/ccw')
    os.mkdir('req/test/oth')

    train_list = range(0, 2631)
    valid_list = range(2631, 2989)
    clean_test_list = range(2989, 3290)

    for i in train_list:
        index = get_four_digit_name(i)
        label = np.loadtxt('pre_processed_patches/{}/label.txt'.format(i))
        class_folder = None
        if label[2] == 0:
            class_folder = 'cw'
        if label[2] == 1:
            class_folder = 'ccw'
        if label[2] == 2:
            class_folder = 'oth'
        os.mkdir('req/train/{}/{}'.format(class_folder, index))
        for idx in range(25):
            idx_ = None
            if idx < 10:
                idx_ = '0{}'.format(idx)
            else:
                idx_ = idx
            cell_patch = cv2.imread('pre_processed_patches/{}/{}.jpg'.format(i, idx))
            cv2.imwrite('req/train/{}/{}/{}.jpg'.format(class_folder, index, idx_), cell_patch)
        np.savetxt('req/train/{}/{}/label.txt'.format(class_folder, index), label)
        print('saved_train {}'.format(index))

    for i2 in valid_list:
        index = get_four_digit_name(i2)
        label = np.loadtxt('pre_processed_patches/{}/label.txt'.format(i2))
        class_folder = None
        if label[2] == 0:
            class_folder = 'cw'
        if label[2] == 1:
            class_folder = 'ccw'
        if label[2] == 2:
            class_folder = 'oth'
        os.mkdir('req/valid/{}/{}'.format(class_folder, index))
        for idx in range(25):
            idx_ = None
            if idx < 10:
                idx_ = '0{}'.format(idx)
            else:
                idx_ = idx
            cell_patch = cv2.imread('pre_processed_patches/{}/{}.jpg'.format(i2, idx))
            cv2.imwrite('req/valid/{}/{}/{}.jpg'.format(class_folder, index, idx_), cell_patch)
        np.savetxt('req/valid/{}/{}/label.txt'.format(class_folder, index), label)
        print('saved_valid {}'.format(index))

    for i3 in clean_test_list:
        index = get_four_digit_name(i3)
        label = np.loadtxt('pre_processed_patches/{}/label.txt'.format(i3))
        class_folder = None
        if label[2] == 0:
            class_folder = 'cw'
        if label[2] == 1:
            class_folder = 'ccw'
        if label[2] == 2:
            class_folder = 'oth'
        os.mkdir('req/test/{}/{}'.format(class_folder, index))
        for idx in range(25):
            idx_ = None
            if idx < 10:
                idx_ = '0{}'.format(idx)
            else:
                idx_ = idx
            cell_patch = cv2.imread('pre_processed_patches/{}/{}.jpg'.format(i3, idx))
            cv2.imwrite('req/test/{}/{}/{}.jpg'.format(class_folder, index, idx_), cell_patch)
        np.savetxt('req/test/{}/{}/label.txt'.format(class_folder, index), label)
        print('saved_test {}'.format(index))


if __name__ == '__main__':
    # get_patch_from_frame()
    # organize_cells()
    # aug()
    req_folder_arrange()
