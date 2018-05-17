import tensorflow as tf
import numpy as np
import sys
from data_utils.data_manager import DataManager
from data_utils import class_name
from model.cnn import get_cnn_model
from model.param_init import kaiming_normal

batch_size = 32
num_epoch = 30
learning_rate = 1e-4

test_name = class_name.CLASS_NAMES[1]
manager = DataManager(cls_name=test_name, dataset_path='./data')
sys.stdout.flush()

dev_data, dev_label = manager.train_data, manager.train_label
X = tf.placeholder(dev_data.dtype, [None]+list(dev_data.shape[1:]))
y = tf.placeholder(dev_label.dtype, [None]+list(dev_label.shape[1:]))
dataset = tf.data.Dataset.from_tensor_slices((X,y))
dataset = dataset.shuffle(buffer_size=1000)
batched_dataset = dataset.batch(batch_size)

iterator = batched_dataset.make_initializable_iterator()
next_data, next_label = iterator.get_next()

cnn_out = get_cnn_model(next_data) #(N, D) 
fc_w1 = tf.get_variable('fc_w1', initializer=kaiming_normal((cnn_out.shape[1].value, 64)), dtype=tf.float32)
fc_b1 = tf.zeros(name='fc_b1', shape=(64), dtype=tf.float32)
fc_w2 = tf.get_variable('fc_w2', initializer=kaiming_normal((64, 1)), dtype=tf.float32)
fc_b2 = tf.zeros(name='fc_b2', shape=(1), dtype=tf.float32)

fc1_out = tf.nn.relu(tf.matmul(cnn_out, fc_w1) + fc_b1)
#fc2_out = tf.reshape(tf.matmul(fc1_out, fc_w2) + fc_b2, [-1])
fc2_out = tf.matmul(fc1_out, fc_w2) + fc_b2
result = (tf.sigmoid(fc2_out) > 0.5)
#print(result.shape, result.dtype)

next_label = tf.expand_dims(next_label, -1)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=next_label, logits=fc2_out), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
global_step = tf.get_variable('global_step', initializer=0, trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(iterator.initializer, feed_dict={X:dev_data, y:dev_label})
    for i in range(num_epoch):
        sess.run(iterator.initializer, feed_dict={X:dev_data, y:dev_label})
        cnt = 0
        total = dev_data.shape[0] // batch_size
        last_loss = 0
        acc = 0.0
        last_step=0
        while True:
            try:
                curr_loss, train, ret, curr_label, step = sess.run([loss, train_op, result, next_label, global_step])
                last_loss = curr_loss
                acc = (cnt*acc + np.sum(curr_label==ret, dtype=np.float)/batch_size) / (cnt+1)
                last_step = step
                # print('iterator: %d/%d\t\tloss: %f' % (cnt, total, curr_loss))
                cnt+=1
            except tf.errors.OutOfRangeError:
                print('Epoch %d/%d\t\tloss: %f\t\tacc: %f\t\tstep: %d' % (i+1, num_epoch, last_loss, acc, last_step))
                break

    sess.run(iterator.initializer, feed_dict={X:manager.val_data, y:manager.val_label})
    val_acc = 0.0
    cnt = 0
    while True:
        try:
            ret, curr_label = sess.run([result, next_label])
            val_acc = (cnt*val_acc + np.sum(curr_label==ret, dtype=np.float)/batch_size) / (cnt+1)
            cnt+=1
        except tf.errors.OutOfRangeError:
            print('validation acc: %f' % val_acc)
            break
    

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