import numpy as np

def seq2matrix(seq):
    """
    Change a RNA sequence(len=300) to a one-hot matrix representation.
    e.g.
        if A=0, C=1, G=2, T=3
        ACG --> [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        :param seq: A RNA sequence
        :return:    a one-hot matrix with size (300 * 4)
    """
    mat = np.zeros([4, len(seq), 1], dtype=np.float32)
    for i in range(len(seq)):
        if seq[i] == 'A':
            mat[0, i, 0] = 1
        elif seq[i] == 'C':
            mat[1, i, 0] = 1
        elif seq[i] == 'G':
            mat[2, i, 0] = 1
        else:
            mat[3, i, 0] = 1
    return mat

def matrix2seq(matrix):
    """
    Change a one-hot RNA matrix(4x300x1) to a RNA sequence.
    :param matrix:  A one-hot RNA matrix
    :return :       A RNA sequence
    """
    bit2char = ['A', 'C', 'G', 'T']
    seq = ""
    pos = np.argmax(matrix, axis=0).squeeze()
    for i in range(pos.shape[0]):
        seq += bit2char[pos[i]]
    return seq


def process_batch_result(data, result, prob):
    """
    process the result from neural network into seq RNA
    :param data:    a list with len=batch_num, each element is a batch of 
                    RNA matrix with shape (Batch_size, 4, 300, 1)

    :param result:  a list with len=batch_num, each element is a batch of 
                    predicted result with shape (Batch_size, 1)

    :param prob:    a list with len=batch_num, each element is a batch of 
                    predicted probability result with shape (Batch_size, 1)

    :return:        the processed data and label
    """
    batch_num = len(data)
    datas = []
    results = []
    probs = []

    for b in range(batch_num):
        batch_size = data[b].shape[0]
        for e in range(batch_size):
            RNA_seq = matrix2seq(data[b][e])
            datas.append(RNA_seq)
            results.append(int(result[b][e][0]))
            probs.append(prob[b][e][0])
    
    return datas, results, probs



