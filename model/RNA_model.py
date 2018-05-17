import sys, os
import tensorflow as tf
import numpy as np
from model.cnn import get_cnn_model
from model.param_init import kaiming_normal
from data_utils.data_manager import DataManager
from data_utils.class_name import CLASS_NAMES

class RNA_model(object):
    """
    The class of model structure

    cls_name:             The name of class
    kwargs:
    - batch_size:           just size of batch
    - num_epoch:            The number of epoch
    - dataset_path:         the dir path that contain the whole dataset
    - learning_rate:        just learning rate
    - model_save_path:      the place that saves the model parameters and checkpoints
    - learning_rate_decay:  decay rate for learning rate
    - use_decay:            whether or not using learning rate decay
    - fc_hidden_num:        num of hidden layer in fully-connented net
    - is_train:             train or test
    - test_set_path:        the path of test dataset
    - TODO: dropout?
    - TODO: regularization?

    """
    # TODO: test()
    def __init__(self, cls_name, **kwargs):
        # get parameters
        self.cls_name = cls_name
        self.batch_size = kwargs.pop('batch_size', 32) #(name, default_value)
        self.num_epoch = kwargs.pop('num_epoch', 30)
        self.dataset_path = kwargs.pop('dataset_path', None)
        self.learning_rate = kwargs.pop('learning_rate', 1e-4)
        self.model_save_path = kwargs.pop('model_save_path', './checkpoint/%s/model.ckpt'%self.cls_name)
        self.learning_rate_decay = kwargs.pop('learning_rate_decay', 0.0) # 0.05 is ok
        self.fc_hidden_num = kwargs.pop('fc_hidden_num', 64)
        self.is_train = kwargs.pop('is_train', True)
        self.use_decay = kwargs.pop('use_decay', False)
        self.test_set_path = kwargs.pop('test_set_path', None)

        # do some check
        if self.cls_name not in CLASS_NAMES:
            raise ValueError('Invalid class name(or None for class name): %s' % self.cls_name)

        if self.is_train:
            if self.dataset_path == None:
                raise ValueError('No dataset path is given')
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
        else:
            if self.test_set_path == None:
                raise ValueError('No test dataset is given')
            if not os.path.exists(self.model_save_path):
                raise ValueError('Invalid path or can not find checkpoints for this model')
    
    
    def build_model(self, batch_data, is_train=True):
        """
        Build computational graph for model
        :param batch_data: input tensor
        
        Return the output of model
        """
        # cnn part
        cnn_out = get_cnn_model(batch_data) #(N, D)
        # the fully-connected net part
        with tf.variable_scope('fc_net'):
            fc_w1 = tf.get_variable('fc_w1', initializer=kaiming_normal((cnn_out.shape[1].value, self.fc_hidden_num)), dtype=tf.float32)
            fc_b1 = tf.get_variable('fc_b1', initializer=tf.zeros([self.fc_hidden_num]), dtype=tf.float32)
            # fc_b1 = tf.zeros(name='fc_b1', shape=(self.fc_hidden_num), dtype=tf.float32)
            fc_w2 = tf.get_variable('fc_w2', initializer=kaiming_normal((self.fc_hidden_num, 1)), dtype=tf.float32)
            fc_b2 = tf.get_variable('fc_b2', initializer=tf.zeros([1]), dtype=tf.float32)
            # fc_b2 = tf.zeros(name='fc_b2', shape=(1), dtype=tf.float32)
            fc1_out = tf.nn.relu(tf.matmul(cnn_out, fc_w1) + fc_b1)
            fc2_out = tf.matmul(fc1_out, fc_w2) + fc_b2

        result = (tf.sigmoid(fc2_out) > 0.5)
        return fc2_out, result



    def train(self):
        if not self.is_train:
            raise TypeError('This model is not for training.')
        # get the dataset
        data_manager = DataManager(cls_name=self.cls_name, dataset_path=self.dataset_path)
        sys.stdout.flush()
        train_data, train_label = data_manager.train_data, data_manager.train_label
        num_train_data = train_data.shape[0]
        X = tf.placeholder(train_data.dtype, [None]+list(train_data.shape[1:]))
        y = tf.placeholder(train_label.dtype, [None]+list(train_label.shape[1:]))
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=8000)
        batched_dataset = dataset.batch(self.batch_size)

        iterator = batched_dataset.make_initializable_iterator()
        batch_data, batch_label = iterator.get_next()
        batch_label = tf.expand_dims(batch_label, -1)
        
        # build model and get parameters
        fc2_out, result = self.build_model(batch_data)
        with tf.variable_scope('cnn', reuse=True):
            conv_w1 = tf.get_variable('conv_w1')
            conv_b1 = tf.get_variable('conv_b1')
        with tf.variable_scope('fc_net', reuse=True):
            fc_w1 = tf.get_variable('fc_w1')
            fc_b1 = tf.get_variable('fc_b1')
            fc_w2 = tf.get_variable('fc_w2')
            fc_b2 = tf.get_variable('fc_b2')
        weights = [conv_w1, conv_b1, fc_w1, fc_b1, fc_w2, fc_b2]
        saver = tf.train.Saver(weights)

        # training part
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_label, logits=fc2_out), name='loss')
        global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        learning_rate = self.learning_rate
        if self.use_decay:
            learning_rate = tf.train.natural_exp_decay(
                learning_rate, global_step, 
                decay_rate=self.learning_rate_decay,
                name='learning_rate', decay_steps=num_train_data//self.batch_size)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_epoch):
                sess.run(iterator.initializer, feed_dict={X:train_data, y:train_label})
                cnt = 0
                last_loss, acc = 0.0, 0.0
                while True:
                    try:
                        curr_loss, train, curr_result, curr_label = sess.run([loss, train_op, result, batch_label])
                        acc = (cnt*acc + np.sum(curr_label==curr_result, dtype=np.float)/self.batch_size) / (cnt+1)
                        last_loss = curr_loss
                        cnt+=1
                    except tf.errors.OutOfRangeError:
                        print('Epoch %d/%d\t\tloss: %f\t\tacc: %f' % (i+1, self.num_epoch, last_loss, acc))
                        break
            
            # validation
            sess.run(iterator.initializer, feed_dict={X:data_manager.val_data, y:data_manager.val_label})
            val_acc = 0.0
            cnt = 0
            while True:
                try:
                    curr_result, curr_label = sess.run([result, batch_label])
                    val_acc = (cnt*val_acc + np.sum(curr_label==curr_result, dtype=np.float)/self.batch_size) / (cnt+1)
                    cnt+=1
                except tf.errors.OutOfRangeError:
                    print('validation acc: %f' % val_acc)
                    break
            saver.save(sess, self.model_save_path)
        print('Finish training')
    
    def test(self):
        if self.is_train:
            raise TypeError('This model is not for test')
        
        # get the dataset
        data_manager = DataManager(self.cls_name, self.test_set_path, is_train=False)
        test_data = data_manager.data
        num_test_data = test_data.shape[0]
        X = tf.placeholder(test_data.dtype, [None]+list(test_data.shape[1:]))
        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
        
        fc2_out, result = self.build_model(batch_data)
        with tf.variable_scope('cnn', reuse=True):
            conv_w1 = tf.get_variable('conv_w1')
            conv_b1 = tf.get_variable('conv_b1')
        with tf.variable_scope('fc_net', reuse=True):
            fc_w1 = tf.get_variable('fc_w1')
            fc_b1 = tf.get_variable('fc_b1')
            fc_w2 = tf.get_variable('fc_w2')
            fc_b2 = tf.get_variable('fc_b2')
        weights = [conv_w1, conv_b1, fc_w1, fc_b1, fc_w2, fc_b2]
        saver = tf.train.Saver(weights)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={X:test_data})
            saver.restore(sess, self.model_save_path)
            while True:
                try:
                    curr_result = sess.run(result)
                    curr_result_int = curr_result.astype(np.int)
                except tf.errors.OutOfRangeError:
                    break

        

    
                