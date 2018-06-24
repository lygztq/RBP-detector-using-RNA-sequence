import tensorflow as tf
import numpy as np
import sys
from data_utils.class_name import CLASS_NAMES, CLASS_INDICES, CLASS_NUM, CLASS_SIZE
from model.RNA_model import RNA_model

# hyper-parameters
batch_size = 64
basis_learning_rate = 1e-3
learning_rate_decay=0.02
reg_strength = 5e-4
basis_num_epoch = 80

def single_class_train(cls_name):
    num_epoch = basis_num_epoch #* CLASS_SIZE[cls_name] // 8000
    model = RNA_model(
        cls_name=cls_name, batch_size=batch_size,
        num_epoch=num_epoch, learning_rate=basis_learning_rate,
        learning_rate_decay=learning_rate_decay, dataset_path='./data', 
        keep_prob=0.36, use_decay=True, use_rnn=True, use_reg=True,
        reg_strength=reg_strength, data_augmentation=False
    )
    acc = model.train()
    return acc

def train():
    cnt = 1
    with open('val_acc_record', 'w') as acc_file:
        acc_file.write("class_name\t\tclass_size\t\tfinal_val_acc\n")
        acc_file.write("-----------------------------------------\n")
        for cls_name in CLASS_NAMES:
            print('training %s\t\t(%d / %d)' % (cls_name, cnt, CLASS_NUM))
            acc = single_class_train(cls_name)
            cnt+=1
            acc_file.write(cls_name)
            acc_file.write('\t\t\t')
            acc_file.write(str(CLASS_SIZE[cls_name]))
            acc_file.write('\t\t\t')
            acc_file.write(str(acc) + '\n')

if __name__ == '__main__':
    train()
