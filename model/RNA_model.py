import sys, os
import tensorflow as tf
from model.cnn import get_cnn_model
from model.param_init import kaiming_normal
from data_utils.data_manager import DataManager
from data_utils.class_name import CLASS_NAMES

class RNA_model(object):
    """
    The class of model structure
    """
    # TODO: build_model() and test()
    def __init__(self, **kargs):
        """
        kargs:
        - cls_name:             The name of class
        - batch_size:           just size of batch
        - dataset_path:         the dir path that contain the whole dataset
        - learning_rate:        just learning rate
        - model_save_path:      the place that saves the model parameters and checkpoints
        - learning_rate_decay:  decay rate for learning rate
        - fc_hidden_num:        num of hidden layer in fully-connented net
        - is_train:             train or test
        - test_set_path:        the path of test dataset
        - TODO: dropout?
        - TODO: regularization?
        """
        # get parameters
        self.cls_name = kargs.get('cls_name', default=None)
        self.batch_size = kargs.get('batch_size', default=32)
        self.dataset_path = kargs.get('dataset_path', default=None)
        self.learning_rate = kargs.get('learning_rate', default=1e-4)
        self.model_save_path = kargs.get('model_save_path', default='./checkpoint')
        self.learning_rate_decay = kargs.get('learning_rate_decay', default=1.0)
        self.fc_hidden_num = kargs.get('fc_hidden_num', default=64)
        self.is_train = kargs.get('is_train', default=True)
        self.test_set_path = kargs.get('test_set_path', default=None)

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
    
    def train(self):
        # get the dataset
        data_manager = DataManager(cls_name=self.cls_name, dataset_path=self.dataset_path)
        sys.stdout.flush()
        train_data, train_label = data_manager.train_data, data_manager.train_label
        num_train_data = train_data.shape[0]
        X = tf.placeholder(train_data.dtype, [None]+list(train_data.shape[1:]))
        y = tf.placeholder(train_label.dtype, [None]+list(train_label[1:]))
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=8000)
        batched_dataset = dataset.batch(self.batch_size)

        iterator = batched_dataset.make_initializable_iterator()
        batch_data, batch_label = iterator.get_next()
        batch_label = tf.expand_dims(batch_label, -1)

        # model structure
        cnn_out = get_cnn_model(batch_data) #(N, D)
        # the fully-connected net part
        with tf.variable_scope('fc_net'):
            fc_w1 = tf.get_variable('fc_w1', initializer=kaiming_normal((cnn_out.shape[1].value, self.fc_hidden_num)), dtype=tf.float32)
            fc_b1 = tf.zeros(name='fc_b1', shape=(self.fc_hidden_num), dtype=tf.float32)
            fc_w2 = tf.get_variable('fc_w2', initializer=kaiming_normal((self.fc_hidden_num, 1)), dtype=tf.float32)
            fc_b2 = tf.zeros(name='fc_b2', shape=(1), dtype=tf.float32)
            fc1_out = tf.nn.relu(tf.matmul(cnn_out, fc_w1) + fc_b1)
            fc2_out = tf.matmul(fc1_out, fc_w2) + fc_b2
        
        result = (tf.sigmoid(fc2_out) > 0.5)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_label, logits=fc2_out), name='loss')
        
        # training part
        global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        learning_rate = self.learning_rate
        learning_rate = tf.train.natural_exp_decay(
            learning_rate, global_step, 
            decay_rate=self.learning_rate_decay,
            name='learning_rate', decay_steps=num_train_data//self.batch_size)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
                