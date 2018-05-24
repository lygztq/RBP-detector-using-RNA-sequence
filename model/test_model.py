import sys, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from model.cnn import get_cnn_model
from model.rnn import biLSTM
from model.param_init import kaiming_normal
from data_utils.data_manager import DataManager
from data_utils.class_name import CLASS_NAMES

class test_RNA_model(object):
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
    - keep_prob:            the keep probability for dropout 
    - use_rnn:              whether use rnn
    - reg_strength:         the strength of regularization
    - use_reg:              use regularization or not
    - leaky_relu_alpha:     alpha for leaky ReLU

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
        self.fc_hidden_num = kwargs.pop('fc_hidden_num', 48)
        self.is_train = kwargs.pop('is_train', True)
        self.use_decay = kwargs.pop('use_decay', False)
        self.test_set_path = kwargs.pop('test_set_path', None)
        self.keep_prob = kwargs.pop('keep_prob', 1.0)
        self.use_rnn = kwargs.pop('use_rnn', False)
        self.reg_strength = kwargs.pop('reg_strength', 0.0)
        self.use_reg = kwargs.pop('use_reg', False)
        self.leaky_relu_alpha = kwargs.pop('leaky_relu_alpha', 0.01)

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
    
    
    def build_model(self, batch_data, keep_prob, reuse=False):
        """
        Build computational graph for model
        :param batch_data: input tensor
        :param keep_prob:  the keep probability of dropout
        :reuse:            use new parameters or old parameters(i.e. if new, for training,
                           if old, for test)
        
        Return the output of model
        """
        # cnn part
        cnn_out = get_cnn_model(batch_data, reuse=reuse) #(N, 99, 64)


        # rnn part
        if self.use_rnn:
            rnn_out = biLSTM(cnn_out, reuse=reuse)
            nn_out = rnn_out
        else:
            new_dim = np.prod([i.value for i in cnn_out.shape[1:]])
            nn_out = tf.reshape(cnn_out, [-1, new_dim])

        # the fully-connected net part
        with tf.variable_scope('fc_net', reuse=reuse):
            fc_w1 = tf.get_variable('fc_w1', initializer=kaiming_normal((nn_out.shape[1].value, self.fc_hidden_num)), dtype=tf.float32)
            fc_b1 = tf.get_variable('fc_b1', initializer=tf.zeros([self.fc_hidden_num]), dtype=tf.float32)
            fc_w2 = tf.get_variable('fc_w2', initializer=kaiming_normal((self.fc_hidden_num, self.fc_hidden_num)), dtype=tf.float32)
            fc_b2 = tf.get_variable('fc_b2', initializer=tf.zeros([self.fc_hidden_num]), dtype=tf.float32)
            fc_w3 = tf.get_variable('fc_w3', initializer=kaiming_normal((self.fc_hidden_num, self.fc_hidden_num)), dtype=tf.float32)
            fc_b3 = tf.get_variable('fc_b3', initializer=tf.zeros([self.fc_hidden_num]), dtype=tf.float32)
            fc_w4 = tf.get_variable('fc_w4', initializer=kaiming_normal((self.fc_hidden_num, 1)), dtype=tf.float32)
            fc_b4 = tf.get_variable('fc_b4', initializer=tf.zeros([1]), dtype=tf.float32)

            dropout_nn_out = tf.nn.dropout(nn_out, 2*keep_prob)
            
            fc1_out = tf.nn.leaky_relu(tf.matmul(dropout_nn_out, fc_w1) + fc_b1, alpha=self.leaky_relu_alpha)
            dropout_fc1_out = tf.nn.dropout(fc1_out, keep_prob)
            
            fc2_out = tf.nn.leaky_relu(tf.matmul(dropout_fc1_out, fc_w2) + fc_b2, alpha=self.leaky_relu_alpha)
            dropout_fc2_out = tf.nn.dropout(fc2_out, keep_prob)

            fc3_out = tf.nn.leaky_relu(tf.matmul(dropout_fc2_out, fc_w3) + fc_b3, alpha=self.leaky_relu_alpha)
            dropout_fc3_out = tf.nn.dropout(fc3_out, keep_prob)

            fc4_out = tf.matmul(dropout_fc3_out, fc_w4) + fc_b4

        result = (tf.sigmoid(fc4_out) > 0.5)
        return fc4_out, result



    def train(self):
        # check
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
        dropout_keep_prob = tf.placeholder(tf.float32)
        fc2_out, result = self.build_model(batch_data, dropout_keep_prob)
        saver = tf.train.Saver()

        # training part
        data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_label, logits=fc2_out), name='data_loss')

        # add regularization
        if self.use_reg:
            with tf.variable_scope('cnn', reuse=True):
                conv_w1 = tf.get_variable('conv_w1')
            with tf.variable_scope('fc_net', reuse=True):
                fc_w1 = tf.get_variable('fc_w1')
                fc_w2 = tf.get_variable('fc_w2')
                fc_w3 = tf.get_variable('fc_w3')
                fc_w4 = tf.get_variable('fc_w4')
            reg_loss = tf.nn.l2_loss(conv_w1) + tf.nn.l2_loss(fc_w1) + tf.nn.l2_loss(fc_w2) \
                        + tf.nn.l2_loss(fc_w3) + tf.nn.l2_loss(fc_w4)
            loss = data_loss + self.reg_strength * reg_loss
        else:
            loss = data_loss

        
        global_step = tf.get_variable('global_step', initializer=0.0, trainable=False)
        # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        learning_rate = tf.get_variable('learning_rate', initializer=self.learning_rate, dtype=tf.float32)
        if self.use_decay:
            learning_rate = tf.train.natural_exp_decay(
                self.learning_rate, global_step, 
                decay_rate=self.learning_rate_decay,
                name='learning_rate', decay_steps=1)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        global_step_add = tf.assign_add(global_step, 1)

        train_acc_hist = []
        val_acc_hist = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_epoch):
                sess.run(iterator.initializer, feed_dict={X:train_data, y:train_label})
                cnt = 0
                last_loss, acc = 0.0, 0.0
                while True:
                    try:
                        curr_loss, train, curr_result, curr_label = sess.run(
                            [loss, train_op, result, batch_label],
                            feed_dict={dropout_keep_prob: self.keep_prob}
                            )
                        acc = (cnt*acc + np.sum(curr_label==curr_result, dtype=np.float)/self.batch_size) / (cnt+1)
                        last_loss = curr_loss
                        cnt+=1
                    except tf.errors.OutOfRangeError:
                        print('Epoch %d/%d\t\tloss: %f\t\tacc: %f' % (i+1, self.num_epoch, last_loss, acc))
                        train_acc_hist.append(acc)
                        sess.run([global_step_add])
                        g_s, lr = sess.run([global_step, learning_rate])
                        print ('learning_rate: %f' % lr)
                        break
                # validation
                sess.run(iterator.initializer, feed_dict={X:data_manager.val_data, y:data_manager.val_label})
                val_acc = 0.0
                cnt = 0
                while True:
                    try:
                        curr_result, curr_label = sess.run([result, batch_label], feed_dict={dropout_keep_prob: 1.0})
                        val_acc = (cnt*val_acc + np.sum(curr_label==curr_result, dtype=np.float)/self.batch_size) / (cnt+1)
                        cnt+=1
                    except tf.errors.OutOfRangeError:
                        print('validation acc: %f' % val_acc)
                        val_acc_hist.append(val_acc)
                        break
            saver.save(sess, self.model_save_path)
        plt.figure(1)
        plt.grid(True)
        plt.plot(np.array(train_acc_hist), 'b-', label="train_acc")
        plt.plot(np.array(val_acc_hist), 'y-', label="val_acc")
        plt.legend()
        plt.show()
        plt.savefig('./train_record.png', format='png')
        print('Finish training')
        sys.stdout.flush()
    
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
        
        dropout_keep_prob = tf.placeholder(tf.float32)
        _, result = self.build_model(batch_data, dropout_keep_prob, reuse=True)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={X:test_data})
            saver.restore(sess, self.model_save_path)
            while True:
                try:
                    curr_result = sess.run(result, feed_dict={dropout_keep_prob: 1.0})
                    curr_result_int = curr_result.astype(np.int)
                except tf.errors.OutOfRangeError:
                    break
        

        

    
                