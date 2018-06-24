import numpy as np
from data_utils.class_name import CLASS_NAMES, CLASS_NUM
import os

def write_file(file_path, data, with_label=False, label=None, prob=None):
    with open(file_path, 'w') as target_file:
        for i, d in enumerate(data):
            target_file.write(d+'\t')
            if with_label:
                target_file.write(str(label[i]) + '\t' + str(prob[i]))
            target_file.write('\n')


def read_file(file_path, with_label=True):
    """
    Read data from a file
    
    :param file_path: The path of the file
    :param with_label: with or without label
    
    :Return : two numpy array contain the data and the label 
    """
    contain = open(file_path, 'r')
    data = []
    if with_label:
        label = []

    for line in contain:
        # if line == "":
        #     continue
        line = line.split()
        data.append(line[0])
        if with_label:
            label.append(int(line[1]))

    data = np.array(data)
    if with_label:
        label = np.array(label)
        return data, label
    return data

def read_dirs(dir_path, with_label=True):
    """
    Read data from all dataset dir
    :param dir_path: The path that contains the whole dataset.
    :param with_label: with or without label
    :Return : data(num_class, num_instance, len_instance) and label(num_class, num_instance)
    """ 
    names = CLASS_NAMES
    cls_num = CLASS_NUM
    datas = {}
    if with_label:
        labels = {}

    for n in names:
        try:
            if with_label:
                file_name = os.path.join(dir_path, n)
                data, label = read_file(file_name)
                labels[n] = label
            else:
                file_name = os.path.join(dir_path, n)
                data = read_file(file_name, with_label=False)
            datas[n] = data
        except:
            print("Can not find raw data for %s" % n)
        
    if with_label:
        return datas, labels
    else:
        return datas


def load_dataset(cls_name, path='../data'):
    """
    Load the preprocessed data for dataManager
    """
    data_file_name = 'data_%s.npy' % cls_name
    label_file_name = 'label_%s.npy' % cls_name
    data_path = os.path.join(path, data_file_name)
    label_path = os.path.join(path, label_file_name)

    data = np.load(data_path)
    label = np.load(label_path)
    return data, label


def load_test_data(cls_name, path='../data'):
    """
    Load preprocessed test data without label
    """
    test_data_file_name = 'test_data_%s.npy' % cls_name
    test_data_path = os.path.join(path, test_data_file_name)
    
    test_data = np.load(test_data_path)
    return test_data

# test
# data_path = '../data/AGO1/train'
# data, label = read_file(data_path)
# print data[0], '\t', label[0]
# print data.shape, label.shape

# d,l = read_dirs('../data')
# for i in range(len(d)):
#     print class_name.CLASS_NAMES[i], len(d[i])