import tensorflow as tf
import numpy as np
from data_utils.load_data import load_dataset, load_test_data
import data_utils.class_name

class DataManager(object):
    """
    The class of data manager
    """
    def __init__(self, dataset_path, cls_name, val_ratio=0.1, dev_ratio=0.1, is_train=True):
        """
            :param dataset_path:    The path of the dataset
            :param ratio:           (validation_set_size) / (total_dataset_size)
        """
        self.dataset_path = dataset_path
        self.cls_name = cls_name
        self.val_ratio = val_ratio
        self.dev_ratio = dev_ratio
        self.is_train = is_train
        if self.is_train:
            self.data, self.label = load_dataset(cls_name, dataset_path)
            self._train_val_split()
        else:
            self.data = load_test_data(cls_name, dataset_path)
        self.num_data = self.data.shape[0]
        print('From DataManager: Loaded dataset size is %d, name is %s' % (self.num_data, cls_name))
            

    def _train_val_split(self):
        """
        Split the training set ,validation set and the development set
        """
        idx = np.arange(self.num_data)
        np.random.shuffle(idx)
        val_num = int(self.val_ratio * self.num_data)
        dev_num = int(self.dev_ratio * self.num_data)
        self.num_train = self.num_data - val_num

        self.val_data = self.data[idx[:val_num]]
        self.train_data = self.data[idx[val_num:]]
        self.dev_data = self.data[idx[:dev_num]]

        self.val_label = self.label[idx[:val_num]]
        self.train_label = self.label[idx[val_num:]]
        self.dev_label = self.label[idx[:dev_num]]


## Test for data manager
# test_name = class_name.CLASS_NAMES[0]
# manager = DataManager(test_name)
# print(manager.train_data.shape, manager.train_label.shape)
# print(manager.val_data.shape, manager.val_label.shape)
# print(manager.dev_data.shape, manager.dev_label.shape)



# test_name = class_name.CLASS_NAMES[0]
# data, label = load_data(test_name)
# print(data.shape, label.shape)

# x = tf.placeholder(data.dtype, data.shape)
# y = tf.placeholder(label.dtype, label.shape)

# dataset = tf.data.Dataset.from_tensor_slices((x, y))
# dataset = dataset.shuffle(buffer_size=8000)
# batched_dataset = dataset.batch(4)
# print(dataset)

# iterator = batched_dataset.make_initializable_iterator()
# next_data, next_label = iterator.get_next()
# z = next_data + 10

# with tf.Session() as sess:
#     sess.run(iterator.initializer, feed_dict={x:data, y:label})
#     for i in range(3):
#         d, l, ret = sess.run([next_data, next_label, z])
#         print(d.shape, l.shape)
#         print(d, l, ret)