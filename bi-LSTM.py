import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle


def load_X(X_attribute):
    """Given attribute(train or test) of feature, and read all 9 features into a ndarray,
    shape is [sample_num,time_steps,feature_num]
        argument: X_path str attribute of feature: train or test
        return:  ndarray tensor of features
    """
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]
    X_path = './data/UCI HAR Dataset/' + X_attribute + '/Inertial Signals/'
    X = []  # define a list to store the final features tensor
    for name in INPUT_SIGNAL_TYPES:
        absolute_name = X_path + name + X_attribute + '.txt'
        f = open(absolute_name, 'rb')
        # each_x shape is [sample_num,each_steps]
        each_X = [np.array(serie, dtype=np.float32) for serie in [
            row.replace("  ", " ").strip().split(" ") for row in f]]
        # add all feature into X, X shape [feature_num, sample_num, time_steps]
        X.append(each_X)
        f.close()
    # trans X from [feature_num, sample_num, time_steps] to [sample_num,
    # time_steps,feature_num]
    X = np.transpose(np.array(X), (1, 2, 0))
    # print X.shape
    return X


def load_Y(Y_attribute):
    """ read Y file and return Y 
        argument: Y_attribute str attibute of Y('train' or 'test')
        return: Y ndarray the labels of each sample,range [0,5]
    """
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]
    Y_path = './data/UCI HAR Dataset/' + Y_attribute + '/y_' + Y_attribute + '.txt'
    f = open(Y_path)
    # create Y, type is ndarray, range [0,5]
    Y = np.array([int(row) for row in f], dtype=np.int32) - 1
    f.close()
    return Y


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 300
        self.batch_size = 100
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None

        # LSTM structure
        # Features count is of 9: three 3D sensors features over time
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 28  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        self.n_residual_layers = 2  # Residual LSTMs, highway-style
        self.n_stacked_layers = 2  # Stack multiple blocks of residual/highway


def one_hot(Y):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    Y_num = len(Y)
    return np.eye(6)[np.array(Y)]


def linear(input_2D_tensor_list, features_len, new_features_len):
    """make linear operator, mainly change the shape of tensor
       both input and output is a list of tensor
        argument: 
            input_2D_tensor_list: list shape is [batch_size,feature_num]
            features_len: int the initial features length of input_2D_tensor
            new_feature_len: int the final features length of output_2D_tensor
        return:
            output_2D_tensor_list lit shape is [batch_size,new_feature_len]
    """
    # Linear activation, reshaping inputs to each LSTMs
    # according to their number of hidden:

    # Note: it might be interesting to try different initializers.
    W = tf.Variable(tf.random_normal([features_len, new_features_len]))
    b = tf.Variable(tf.random_normal([new_features_len]))
    # W = tf.get_variable('linear_weights', initializer=tf.random_normal(
    #     [features_len, new_features_len]))
    # b = tf.get_variable(
    #     'linear_biases', initializer=tf.random_normal([new_features_len]))

    output_2D_tensor_list = [tf.nn.relu(tf.matmul(input_2D_tensor, W) + b)
                             for input_2D_tensor in input_2D_tensor_list]
    # make ReLU to the output
    return output_2D_tensor_list


def LSTM_cell(input_hidden_tensor, n_outputs):
    """ define the basic LSTM layer 
        argument:
            input_hidden_tensor: list a list of tensor, 
                                 shape: time_steps*[batch_size,n_inputs]
            n_outputs: int num of LSTM layer output
        return: 
            outputs: list a time_steps list of tensor, 
                     shape: time_steps*[batch_size,n_outputs]
    """
    # print input_hidden_tensor
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_outputs, forget_bias=1.0)
    outputs, _ = tf.nn.rnn(lstm_cell, input_hidden_tensor, dtype=tf.float32)

    return outputs


def bidirectional_LSTM_cell(input_hidden_tensor, n_inputs, n_outputs):
    """build bi-LSTM, it seems like stack two layers LSTM by contraries.
    so ,the output num is 2*n_outputs
        argument:
            input_hidden_tensor: list a time_steps series of tensor, shape: [sample_num,n_inputs]
            n_inputs: int units of input tensor
            n_outputs: int units of output tensor
        return: 
            layer_hidden_outputs: list a time_steps series of tensor, shape: [sample_num,2*n_outputs]          
    """

    with tf.name_scope("pass_forward") as pass_forward:
        # hidden_forward shape: time_steps*[batch_size,n_outputs]
        hidden_forward = linear(input_hidden_tensor, n_inputs, n_outputs)
        # forward shape: time_steps*[batch_size,n_outpus]
        forward = LSTM_cell(hidden_forward, n_outputs)
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope("pass_backward") as pass_backward:
        # hidden backward shape: time_steps*[batch_size,n_outputs]
        hidden_backward = linear(input_hidden_tensor, n_inputs, n_outputs)
        # backward shape: time_steps*[batch_size,n_outpus]
        backward = list(
            reversed(LSTM_cell(list(reversed(hidden_backward)), n_outputs)))
    # Simply concatenating cells' outputs at each timesteps on the innermost
    # dimension, like if the two cells acted as one cell
    # with twice the n_hidden size:
    layer_hidden_outputs = [tf.concat(len(f.get_shape()) - 1, [f, b])
                            for f, b in zip(forward, backward)]
    # print len(layer_hidden_outputs), layer_hidden_outputs[0].get_shape()
    return layer_hidden_outputs


def LSTM_network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, 
                     shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : ndarray  output shape [batch_size,n_classes]
    """
    #------reshape input data---------------------
    # feature_mat shape : [time_steps,batch_size,n_inputs]
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    tf.transpose
    # fature_mat shape: [time_steps*batch_size,n_inputs]
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # feature_mat shape: time_steps*[batch_size,n_inputs]
    feature_mat = tf.split(0, config.n_steps, feature_mat)
    #--------stack bi-LSTM layer 1-----------------
    with tf.name_scope("Bi-LSTM_layer_1"):
        hidden1 = bidirectional_LSTM_cell(
            feature_mat, config.n_inputs, config.n_hidden)

    #--------stack bi-LSTM layer 2-----------------
    with tf.name_scope("Bi-LSTM_layer_2"):
        hidden2 = bidirectional_LSTM_cell(
            hidden1, config.n_hidden * 2, config.n_hidden)

    #--------stack full connection layer 3---------
    with tf.name_scope("full_connection_layer"):
        # output_list is a time_steps series of shape [batch_size,n_classes]
        output_list = linear(hidden2, config.n_hidden * 2, config.n_classes)
    return output_list[-1]


def main():
    #-----------------------------
    # step1: load and prepare data
    #-----------------------------
    # Those are separate normalised input features for the neural network
    # shape [sample_num,time_steps,feature_num]=[7352,128,9]
    X_train = load_X('train')
    # shape [sample_num,time_steps,feature_num]=[1947,128,9]
    X_test = load_X('test')
    Y_train = load_Y('train')  # shape [sample_num,]=[7352,]
    Y_test = load_Y('test')  # shape [sample_num,]=[2947]
    Y_train = one_hot(Y_train)
    Y_test = one_hot(Y_test)
    # print X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
    # Output classes to learn how to classify
    #-----------------------------------
    # step2: define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)

    #------------------------------------------------------
    # step3: Let's get serious and build the neural network
    #------------------------------------------------------
    # change input data to a time_steps series of [batch_size,feature_nums]
    X = tf.placeholder(dtype=tf.float32, shape=[
                       None, config.n_steps, config.n_inputs], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[
                       None, config.n_classes], name="Y")

    pred_Y = LSTM_network(X, config)
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
    # optimize = tf.contrib.layers.optimize_loss(
    #     loss,
    #     global_step=tf.Variable(0),
    #     learning_rate=config.learning_rate,
    #     optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
    #     clip_gradients=config.clip_gradients,
    #     gradient_noise_scale=config.gradient_noise_scale
    # )
    optimize = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    sess = tf.InteractiveSession(
        config=tf.ConfigProto(log_device_placement=False))
    tf.initialize_all_variables().run()
    best_accuracy = 0.0
    for i in range(config.training_epochs):
        shuffled_X, shuffled_y = shuffle(X_train, Y_train, random_state=i * 42)
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(
                optimize,
                feed_dict={
                    X: shuffled_X[start:end],
                    Y: shuffled_y[start:end]
                }
            )
        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, loss],
            feed_dict={X: X_test, Y: Y_test}
        )

        print("train iter: {},".format(i) +
              " test accuracy: {},".format(accuracy_out) +
              " loss: {}".format(loss_out))

        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
if __name__ == "__main__":
    main()
