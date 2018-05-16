from load_data import read_dirs
import class_name
import numpy as np
import sys

def seq2matrix(seq):
    """
    Change a RNA sequence(len=300) to a one-hot matrix representation.
    e.g.
        if A=0, C=1, G=2, T=3
        ACG --> [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        :param seq: A RNA sequence
        :return:    a one-hot matrix with size (300 * 4)
    """
    mat = np.zeros([4, len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'A':
            mat[0][i] = 1
        elif seq[i] == 'C':
            mat[1][i] = 1
        elif seq[i] == 'G':
            mat[2][i] = 1
        else:
            mat[3][i] = 1
    return mat

def preprocess():
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
    print 'Loading origin data.'
    sys.stdout.flush()
    data, label = read_dirs('../data')
    for i in range(len(data)):
        print 'class name: ', class_name.CLASS_NAMES[i], '\tdataset size: ', len(data[i])
        sys.stdout.flush()

    num = len(data)

    for c in range(num):
        print 'PROCESSING CLASS: %s' % class_name.CLASS_NAMES[c]
        sys.stdout.flush()
        num_instance = len(data[c])
        one_hot_data = np.zeros([num_instance, 4, len(data[c][0])])
        curr_label = np.array(label[c])
        for i in range(num_instance):
            one_hot_data[i] = seq2matrix(data[c][i])
        print 'SAVING CLASS: %s' % class_name.CLASS_NAMES[c]
        sys.stdout.flush()
        np.save('../data/data_%s' % class_name.CLASS_NAMES[c], one_hot_data)
        np.save('../data/label_%s' % class_name.CLASS_NAMES[c], curr_label)

if __name__ == '__main__':
    preprocess()