import os
import glob
import collections
import cv2
import numpy as np
import tensorflow as tf

BatchTuple = collections.namedtuple("BatchTuple", ['images', 'labels'])


class Loader:
    RawDataTuple = collections.namedtuple("RawDataTuple", ['path', 'label'])
    image_info = {
        'width': 256,
        'height': 256,
        'channel': 3,
    }
    label_name = []

    def __init__(self, data_path, batch_size):
        self.sess = tf.Session()

        self.data = {}
        self.data_path = data_path
        self.batch_size = batch_size
        self.cur_idx = {}
        self.perm_idx = {}
        self.epoch_counter = 0

        self.load_data()

        for label in self.label_name:
            self.reset(label)
        return

    def load_data(self):
        # Load data from directory
        print("...Loading from %s" % self.data_path)
        dir_name_list = os.listdir(self.data_path)
        dir_name_list.sort()
        for dir_name in dir_name_list:
            dir_path = os.path.join(self.data_path, dir_name)
            self.label_name.append(dir_name)
            self.data[dir_name] = []
            file_name_list = os.listdir(dir_path)
            print("\tNumber of files in %s = %d" % (dir_name, len(file_name_list)))
            for file_name in file_name_list:
                file_path = os.path.join(dir_path, file_name)
                self.data[dir_name].append(self.RawDataTuple(path=file_path, label=self.label_name.index(dir_name)))

        print("...Loading done.")
        return

    def reset(self, label=None):
        if label is None:
            for tmp in self.label_name:
                self.reset(tmp)
            return

        if label not in self.label_name:
            print("Invalid label to reset loader [%s]" % label)
            return

        self.cur_idx[label] = 0
        self.perm_idx[label] = np.random.permutation(len(self.data[label]))
        self.epoch_counter += 1
        return

    def get_empty_batch(self, batch_size):
        batch = BatchTuple(
            images=np.zeros(dtype=np.uint8, shape=[batch_size, self.image_info['height'], self.image_info['width'], self.image_info['channel']]),
            labels=np.zeros(dtype=np.int32, shape=[batch_size])
        )
        return batch

    def get_batch(self, label, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if label not in self.data.keys():
            print("Invalid label to reset loader")
            return None

        if (self.cur_idx[label] + batch_size) > len(self.data[label]):
            # print('reached the end of this data set')
            self.reset(label)
            return None

        batch = self.get_empty_batch(batch_size)
        for idx in range(batch_size):
            single_data = self.data[label][self.perm_idx[label][self.cur_idx[label] + idx]]
            image = cv2.imread(single_data.path, 1)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            batch.images[idx, :, :, :] = image
            batch.labels[idx] = single_data.label

            # Verifying batch
            # print(single_data.path)
            # print(batch.images[idx, 0, 0, 0])
            # print(batch.labels[idx])

        self.cur_idx[label] += batch_size

        return batch
