# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import resnet
from config import Config
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.util.nest import *
from options import parser

activation = tf.nn.relu
slim = tf.contrib.slim
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
HEIGHT = 480
WIDTH = 640
RNN_SIZE = 32
RNN_PROJ = 32
LEFT_CONTEXT = 0
SEQ_LEN = 10
BATCH_SIZE = 16
CHANNELS = 3
NUM_EPOCHS = 100
KEEP_PROB_TRAIN = 0.25
CSV_HEADER = "num,index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[7:]  # angle,torque,speed
OUTPUT_DIM = 3  # predict steering angle
args = parser.parse_args()


class BatchGenerator(object):
    def __init__(self, sequence, seq_len, batch_size, offset):
        self.sequence = sequence
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.chunk_size = 1 + (len(sequence) - 1) / batch_size
        self.indices = [(i * self.chunk_size + offset) % len(sequence) for i in range(batch_size)]
        self.length = len(sequence)

    def next(self, offset, flag):
        while True:
            output = []
            for i in range(self.batch_size):
                idx = int((self.indices[i] + offset) % self.length)
                left_pad = self.sequence[idx - LEFT_CONTEXT:idx]
                if len(left_pad) < LEFT_CONTEXT:
                    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
                assert len(left_pad) == LEFT_CONTEXT
                leftover = len(self.sequence) - idx
                if leftover >= self.seq_len:
                    result = self.sequence[idx:idx + self.seq_len]
                else:
                    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
                assert len(result) == self.seq_len
                if flag == 0:
                    self.indices[i] = (self.indices[i] + 1) % len(self.sequence)
                else:
                    self.indices[i] = (self.indices[i] + self.seq_len) % len(self.sequence)
                images, targets = zip(*result)
                # images_left_pad, _ = zip(*left_pad)
                output.append((np.stack(images), np.stack(targets)))
            output = zip(*output)
            output[0] = np.stack(output[0])  # batch_size x (LEFT_CONTEXT + seq_len)
            output[1] = np.stack(output[1])  # batch_size x seq_len x OUTPUT_DIM
            return output


def read_csv(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip().split(",")[6:10] for ln in f.readlines()]
        lines = map(lambda x: (x[0], np.float32(x[1:])), lines)  # imagefile, outputs
        return lines


def process_csv(filename, val=0):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    print ("output_dim: %d" % OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation
    train_seq = []
    valid_seq = []
    num = 0
    for ln in lines:
        train_seq.append(ln)
        num += 1
    print ("training seq:%d" % num)

    for cnt in range(len(train_seq)):
        sum_f += train_seq[cnt][1][:]
        sum_sq_f += train_seq[cnt][1][:] * train_seq[cnt][1][:]
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    print (len(train_seq), len(valid_seq))
    print ("current mean, std")
    print (mean, std)
    return (train_seq, valid_seq), (mean, std)


(train_seq, valid_seq), (mean, std) = process_csv(filename="complete_dataset.csv",
                                                  val=0)  # concatenated interpolated.csv from rosbags, total_training_dataset.csv
test_seq = read_csv("exampleSubmissionInterpolatedFinal.csv")  # interpolated.csv for testset filled with dummy values
layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)

def get_optimizer(loss, lrate):
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
    gradvars = optimizer.compute_gradients(loss)
    gradients, v = zip(*gradvars)
    # print ([x.name for x in v])
    gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
    return optimizer.apply_gradients(zip(gradients, v))


class SamplingRNNCell(tf.nn.rnn_cell.RNNCell):
    """Simple sampling RNN cell."""

    def __init__(self, num_outputs, use_ground_truth, internal_cell):
        """
        if use_ground_truth then don't sample
        """
        self._num_outputs = num_outputs
        self._use_ground_truth = use_ground_truth  # boolean
        self._internal_cell = internal_cell  # may be LSTM or GRU or anything

    @property
    def state_size(self):
        return self._num_outputs, self._internal_cell.state_size  # previous output and bottleneck state

    @property
    def output_size(self):
        return self._num_outputs  # steering angle, torque, vehicle speed

    def __call__(self, inputs, state, scope=None):
        (visual_feats, current_ground_truth) = inputs
        prev_output, prev_state_internal = state
        context = tf.concat([prev_output, visual_feats], 1)
        new_output_internal, new_state_internal = internal_cell(context,
                                                                prev_state_internal)  # here the internal cell (e.g. LSTM) is called
        new_output = tf.contrib.layers.fully_connected(
            inputs=tf.concat([new_output_internal, prev_output, visual_feats], 1),
            num_outputs=self._num_outputs,
            activation_fn=None,
            scope="OutputProjection")
        # if self._use_ground_truth == True, we pass the ground truth as the state; otherwise, we use the model's predictions
        return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck is
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = resnet._get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = resnet._get_variable('gamma',
                         params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = resnet._get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = resnet._get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x

def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = resnet._get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv3d(x, weights, [1, 1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool3d(x,
                          ksize=[1, ksize, ksize, ksize, 1],
                          strides=[1, 1, stride, stride, 1],
                          padding='SAME')

if args.flag == 'train':
    with tf.Session() as sess:
        # build 3D ResNet-50
        print('Load pre-trained model')
        saver = tf.train.import_meta_graph(args.path_pre_trained + '.meta') 
        saver.restore(sess, args.path_pre_trained + '.ckpt')
        var_list = saver._var_list
        value=[]
        for i in range(265):
            tmp = sess.run(var_list[i], feed_dict={})
            if tmp.shape[0] < 10:
                tmp = [tmp / 1.0 / tmp.shape[0] for _ in range(tmp.shape[0])]
                tmp = np.stack(tmp, axis=0)
            value.append(tmp)


graph = tf.Graph()

with graph.as_default():
    # inputs
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)  # tf.placeholder_with_default(input=1e-4, shape=())
    keep_prob = tf.placeholder_with_default(input=1.0, shape=())
    aux_cost_weight = tf.placeholder_with_default(input=1.0, shape=())

    inputs = tf.placeholder(shape=(BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN),
                            dtype=tf.string)  # pathes to png files from the central camera
    targets = tf.placeholder(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM),
                             dtype=tf.float32)  # seq_len x batch_size x OUTPUT_DIM

    targets_normalized = (targets - mean) / std
    input_images = tf.stack([tf.image.decode_png(tf.read_file(x))
                            for x in tf.unstack(tf.reshape(inputs, shape=[(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE]))])
    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    input_images = tf.reshape(input_images, [BATCH_SIZE*(LEFT_CONTEXT + SEQ_LEN), HEIGHT, WIDTH, CHANNELS])
    input_images = tf.image.resize_images(input_images, [int(HEIGHT / 3), int(WIDTH / 4)])
    input_images = tf.reshape(input_images, [BATCH_SIZE, (LEFT_CONTEXT + SEQ_LEN), int(HEIGHT / 3), int(WIDTH / 4), CHANNELS])

    num_classes = 1000
    num_blocks = [3, 4, 6, 3]  # defaults to 50-layer network
    use_bias = False  # defaults to using batch norm
    bottleneck = True

    x = input_images
    c = Config()
    c['bottleneck'] = bottleneck
    is_training = tf.placeholder(shape=(), dtype='bool', name='is_training')
    c['is_training'] = is_training
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = 1
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = resnet.bn(x, c)
        x = resnet.activation(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = resnet.stack(x, c)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = resnet.stack(x, c)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = resnet.stack(x, c)

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = resnet.stack(x, c)

    variable_map = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
    op_list = []
    list_tmp = []
    for i in range(265):
        tmp = tf.placeholder(shape=variable_map[i].shape, dtype=tf.float32, name=str(i))
        op_list.append(tf.assign(ref= variable_map[i], value= tmp))
        list_tmp.append(tmp)

    net_tmp = x
    
    net = slim.fully_connected(tf.reshape(net_tmp, [BATCH_SIZE, SEQ_LEN, -1]), 1024, activation_fn=tf.nn.relu)
    net = tf.nn.dropout(x=net, keep_prob=keep_prob)
    net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
    net = tf.nn.dropout(x=net, keep_prob=keep_prob)
    net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
    net = tf.nn.dropout(x=net, keep_prob=keep_prob)
    net = slim.fully_connected(net, 128, activation_fn=None)
    net = layer_norm(tf.nn.elu(net))

    cnn_output = tf.reshape(net, [BATCH_SIZE, SEQ_LEN, -1])
    cnn_output = tf.nn.dropout(x=cnn_output, keep_prob=keep_prob)
    rnn_inputs_with_ground_truth = (cnn_output, targets_normalized)
    rnn_inputs_autoregressive = (cnn_output, tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32))

    internal_cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
    cell_with_ground_truth = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=True, internal_cell=internal_cell)
    cell_autoregressive = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=False, internal_cell=internal_cell)


    def get_initial_state(complex_state_tuple_sizes):
        flat_sizes = flatten(complex_state_tuple_sizes) 
        init_state_flat = [tf.tile(
            multiples=[BATCH_SIZE, 1],
            input=tf.get_variable("controller_initial_state_%d" % i, initializer=tf.zeros_initializer, shape=([1, s]),
                                  dtype=tf.float32))
            for i, s in enumerate(flat_sizes)]
        init_state = pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
        return init_state, flat_sizes, init_state_flat


    def deep_copy_initial_state(complex_state_tuple):
        flat_state = flatten(complex_state_tuple)
        flat_copy = [tf.identity(s) for s in flat_state]
        deep_copy = pack_sequence_as(complex_state_tuple, flat_copy)
        return deep_copy


    controller_initial_state_variables, tmp_0, tmp_1 = get_initial_state(cell_autoregressive.state_size)
    controller_initial_state_autoregressive = deep_copy_initial_state(controller_initial_state_variables)
    controller_initial_state_gt = deep_copy_initial_state(controller_initial_state_variables)

    with tf.variable_scope("predictor"):
        out_gt, controller_final_state_gt = tf.nn.dynamic_rnn(cell=cell_with_ground_truth,
                                                              inputs=rnn_inputs_with_ground_truth,
                                                              sequence_length=[SEQ_LEN] * BATCH_SIZE,
                                                              initial_state=controller_initial_state_gt,
                                                              dtype=tf.float32,
                                                              swap_memory=True, time_major=False)
    with tf.variable_scope("predictor", reuse=True):
        out_autoregressive, controller_final_state_autoregressive = tf.nn.dynamic_rnn(cell=cell_autoregressive,
                                                                                      inputs=rnn_inputs_autoregressive,
                                                                                      sequence_length=[SEQ_LEN] * BATCH_SIZE,
                                                                                      initial_state=controller_initial_state_autoregressive,
                                                                                      dtype=tf.float32,
                                                                                      swap_memory=True,
                                                                                      time_major=False)

    mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
    mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_autoregressive, targets_normalized))
    mse_autoregressive_steering = tf.reduce_mean(tf.squared_difference(out_autoregressive[:, :, 0], targets_normalized[:, :, 0]))
    steering_predictions = (out_autoregressive[:, :, 0] * std[0]) + mean[0]
    total_loss = mse_autoregressive_steering + aux_cost_weight * (mse_autoregressive + mse_gt)
    # print("Parameter size:")
    # print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])) # calculate parameter size
    optimizer = get_optimizer(total_loss, learning_rate)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
global_train_step = 0
global_valid_step = 0
global_test_step = 0

def do_epoch(session, sequences, mode):
    global global_train_step, global_test_step
    train_predictions = {}
    test_predictions = {}
    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, offset=0)
    total_num_steps = 1 + int ((batch_generator.chunk_size - 1) / SEQ_LEN)
    acc_loss = np.float128(0.0)
    for step in range(total_num_steps):
        if mode == "train":
            feed_inputs, feed_targets = batch_generator.next(np.random.randint(108000), 1)  # randomly select training sequences
            feed_dict = {learning_rate: 1e-4, inputs: feed_inputs, targets: feed_targets, is_training: True}
        if mode == "test":
            feed_inputs, feed_targets = batch_generator.next(0, 1)
            feed_dict = {learning_rate: 0.0, inputs: feed_inputs, targets: feed_targets, is_training: True}
        if mode == "train":
            feed_dict.update({keep_prob: KEEP_PROB_TRAIN})
            _, loss, model_predictions_train = \
                session.run([optimizer, mse_autoregressive_steering, steering_predictions],
                            feed_dict=feed_dict)
            global_train_step += 1
            feed_inputs_train = feed_inputs[:, LEFT_CONTEXT:].flatten()
            steering_targets_train = feed_targets[:, :, 0].flatten()
            model_predictions_train = model_predictions_train.flatten()
            stats_train = np.stack(
                [steering_targets_train, model_predictions_train, abs(steering_targets_train - model_predictions_train),
                 (steering_targets_train - model_predictions_train) ** 2])
            for i, img in enumerate(feed_inputs_train):
                train_predictions[img] = stats_train[:, i]
        elif mode == "test":
            model_predictions= \
                session.run(steering_predictions,
                            feed_dict=feed_dict)
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            model_predictions = model_predictions.flatten()
            global_test_step = global_test_step + 1
            for i, img in enumerate(feed_inputs):
                test_predictions[img] = model_predictions[i]
        if mode != "test":
            acc_loss += loss
            if (step + 1) % 40 == 0:
                print (step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step + 1)),)
    print ('')
    if mode == "test":
        return (None, test_predictions)
    return (np.sqrt(acc_loss / total_num_steps), train_predictions)


best_testing_score = None
test_label ={}
count = 0
num_test = 0
img_name = []

with open("CH2_final_evaluation.csv", "r") as f:
    for line in f.readlines():
        test_label[line.strip().split(",")[0] + '.png'] = float(line.strip().split(",")[1])
        img_name.append(line.strip().split(",")[0] + '.png')
        count = count + 1

with graph.as_default():
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        if args.flag == 'train':
            print("Training:")
            session.run(tf.initialize_all_variables())
            for i in range(265):
                _ = session.run([op_list[i]],feed_dict={list_tmp[i]: value[i]})
        if args.flag == 'test':
            print("Testing:")
            saver.restore(session, args.path_trained)

        for epoch in range(NUM_EPOCHS):
	    print("Starting epoch %d" % epoch)
	    mae_error = 0.0
	    rmse_error = 0.0
	    test_predictions = []
	    num_test = 0
            start = time.time()
	    _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
            end = time.time()
            print("Total testing time :" + str(end - start))
            
	    for img, pred in test_predictions.items():
	        img = img.replace("center/", "")
		mae_error = mae_error + abs(pred - test_label[img])
		rmse_error = rmse_error + (pred - test_label[img]) ** 2
		num_test += 1
            print("number of test: %d" % num_test)
	    mae_error = mae_error / 1.0 * 180 / 3.1415 / num_test
	    rmse_error = np.sqrt(rmse_error / 1.0 / num_test) * 180 / 3.1415
	    print("Testing mae error: %.4f, rmse error: %.4f" % (mae_error, rmse_error))
            if args.flag == 'test':
                print('finish testing')
                break
	    if best_testing_score is None:
		best_testing_score = mae_error
	    if mae_error < best_testing_score:
	        saver.save(session, args.path_save)
		best_testing_score = mae_error
		print('\r', " Model has become better, SAVED at epoch %d" % epoch,)
	    if epoch != NUM_EPOCHS - 1:
		print ("Training")
		_, train_predictions = do_epoch(session=session, sequences=train_seq, mode="train")
		result = np.float128(0.0)
		mae_train = np.float128(0.0)
		for img, stats in train_predictions.items():
		    result += stats[-1]
		    mae_train += stats[-2]
		print ("Unnormalized MAE(train):", mae_train / len(train_predictions))
		print ("Training unnormalized RMSE:", np.sqrt(result / len(train_predictions)))
