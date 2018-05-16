import tensorflow as tf
from load_data import load_data
import class_name

test_name = class_name.CLASS_NAMES[0]
data, label = load_data(test_name)
print(data.shape, label.shape)

x = tf.placeholder(data.dtype, data.shape)
y = tf.placeholder(label.dtype, label.shape)

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=8000)
batched_dataset = dataset.batch(4)
print(dataset)

iterator = batched_dataset.make_initializable_iterator()
next_data, next_label = iterator.get_next()
z = next_data + 10

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={x:data, y:label})
    for i in range(3):
        d, l, ret = sess.run([next_data, next_label, z])
        print(d.shape, l.shape)
        print(d, l, ret)