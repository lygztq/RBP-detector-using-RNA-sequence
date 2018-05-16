import numpy as np
import class_name
import os

def read_file(file_path):
    """
    Read data from a file
    
    :param file_path: The path of the file
    
    :Return : two numpy array contain the data and the label 
    """
    contain = open(file_path)
    data = []
    label = []

    for line in contain:
        line = line.split()
        data.append(line[0])
        label.append(int(line[1]))

    data = np.array(data)
    label = np.array(label)
    return data, label

def read_dirs(dir_path):
    """
    Read data from all dataset dir
    :param dir_path: The path that contains the whole dataset.
    :Return : data(num_class, num_instance, len_instance) and label(num_class, num_instance)
    """ 
    names = class_name.CLASS_NAMES
    cls_num = class_name.CLASS_NUM
    datas = []
    labels = []

    for i in range(cls_num):
        name = names[i]
        train_file = os.path.join(dir_path, name, 'train')
        data, label = read_file(train_file)
        datas.append(data)
        labels.append(label)

    return datas, labels


def load_data(cls_name, path='../data'):
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

# test
# data_path = '../data/AGO1/train'
# data, label = read_file(data_path)
# print data[0], '\t', label[0]
# print data.shape, label.shape

# d,l = read_dirs('../data')
# for i in range(len(d)):
#     print class_name.CLASS_NAMES[i], len(d[i])