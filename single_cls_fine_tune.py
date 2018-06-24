import tensorflow as tf
import numpy as np
import sys
from data_utils.class_name import CLASS_NAMES
from model.RNA_model import RNA_model

batch_size = 64
num_epoch = 200
learning_rate = 1e-3
learning_rate_decay = 0.02
reg_strength = 6e-5

test_name = CLASS_NAMES[36]

model = RNA_model(
    cls_name=test_name, batch_size=batch_size,
    num_epoch=num_epoch, learning_rate=learning_rate,
    learning_rate_decay=learning_rate_decay, is_train=True,
    dataset_path='./data', keep_prob=0.225, use_decay=True, use_rnn=False, use_reg=True, reg_strength=reg_strength, data_augmentation=False)

acc = model.train()
print("%s acc changed to %f" % (test_name, acc))
