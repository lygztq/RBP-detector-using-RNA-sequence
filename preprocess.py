from data_utils.load_data import read_dirs
from data_utils.class_name import CLASS_NAMES
from data_utils.RNA_process import seq2matrix, shape2matrix
from data_utils.pyrnashapes import rnashapes
from progressbar import progressbar
import numpy as np
import sys
import argparse

def preprocess(is_train=True):
    """
    Preprocessing the dataset, change the char-RNA seq into one-hot matrix.


    Combining all the 37 classes using 0-1 vector to represent the label and
    merge the duplicated seq.

    The preprocessed data:
        one_hot_data:   a (num_of_seq, 300, 4) numpy array that contains num_of_seq
                        one-hot matrix
        labels:         a (num_of_seq, 37) numpy array that contains the labels
    Save the preprocessed data into /data/datas.npy and /data/labels.npy
    """
    # load the origin data
    print('Loading origin data.')
    sys.stdout.flush()
    if is_train:
        data, label = read_dirs('./data')
    else:
        data = read_dirs('./data', with_label=False)


    for i in data.keys():
        print('class name: ', i, '\tdataset size: ', len(data[i]))
        sys.stdout.flush()

    num = len(data)

    class_size = {}

    for i, c in enumerate(data.keys()):
        print('PROCESSING CLASS: %s\t\t\t%d / %d' % (c, i, num))
        sys.stdout.flush()
        num_instance = len(data[c])
        class_size[c] = num_instance
        one_hot_data = np.zeros([num_instance, 4, len(data[c][0]), 1], dtype=np.float32)
        one_hot_shape = np.zeros([num_instance, 3, len(data[c][0]), 1], dtype=np.float32)
        for i in progressbar(range(num_instance)):
            one_hot_data[i] = seq2matrix(data[c][i])
            one_hot_shape[i] = shape2matrix(rnashapes(data[c][i]))
        print('SAVING CLASS: %s' % c)
        sys.stdout.flush()
        if is_train:
            curr_label = np.array(label[c], dtype=np.float32)
            np.save('./data/data_%s' % c, one_hot_data)
            np.save('./data/shape_%s' % c, one_hot_shape)
            np.save('./data/label_%s' % c, curr_label)
        else:
            np.save('./data/test_data_%s' % c, one_hot_data)
            np.save('./data/test_shape_%s' % c, one_hot_shape)
    # print class_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('-test', '-t', action='store_true', help='using test mode')
    args = parser.parse_args()
    preprocess(not args.test)
