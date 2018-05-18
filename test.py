import tensorflow as tf
import numpy as np
import sys
from data_utils.class_name import CLASS_NAMES
from model.RNA_model import RNA_model

batch_size = 64
num_epoch = 40
learning_rate = 1e-4
learning_rate_decay = 0.001

test_name = CLASS_NAMES[0]

model = RNA_model(
    cls_name=test_name, batch_size=batch_size,
    num_epoch=num_epoch, learning_rate=learning_rate,
    learning_rate_decay=learning_rate_decay, is_train=True,
    dataset_path='./data', keep_prob=0.25, use_decay=True)

model.train()
