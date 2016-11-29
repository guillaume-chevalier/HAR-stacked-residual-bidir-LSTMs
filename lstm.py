
# Thanks to Zhao Yu for converting the .ipynb notebook to
# this simplified Python script that I edited a little.

# Note that the dataset must be already downloaded for this script to work, do:
#     $ cd data/
#     $ python download_dataset.py

import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle

import os


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'rb')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'rb')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


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
        self.training_epochs = 100
        self.batch_size = 100
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        self.n_residual_layers = 2  # Residual LSTMs, highway-style
        self.n_stacked_layers = 2  # Stack multiple blocks of residual/highway


def linear(input_2D_tensor_list, features_len, new_features_len):
    # Linear activation, reshaping inputs to each LSTMs
    # according to their number of hidden:

    # Note: it might be interesting to try different initializers.
    W = tf.get_variable(
        "linear_weights",
        initializer=tf.random_normal([features_len, new_features_len])
    )
    b = tf.get_variable(
        "linear_biases",
        initializer=tf.random_normal([new_features_len])
    )
    # The following step could probably be optimized by multiplying
    # once with surrounded packing/unpacking operations:

    print "Linear matmul shape: "
    print ([features_len, new_features_len])
    input_2D_tensor_list = [
        tf.nn.relu(tf.matmul(input_2D_tensor, W) + b)
            for input_2D_tensor in input_2D_tensor_list
    ]
    return input_2D_tensor_list


def LSTM_cell(input_hidden_tensor, n_outputs):
    # Define LSTM cell hidden layer:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_outputs, forget_bias=1.0)

    # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
    outputs, _ = tf.nn.rnn(lstm_cell, input_hidden_tensor, dtype=tf.float32)
    # outputs' shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_classes]

    return outputs


def bidirectional_LSTM_cell(input_hidden_tensor, n_inputs, n_outputs):
    """
    The input_hidden_tensor should be a list of lenght "time_step"
    containing tensors of shape [batch_size, n_lower_features]
    """
    print "bidir:"
    print (len(input_hidden_tensor), str(input_hidden_tensor[0].get_shape()))

    with tf.variable_scope('bidir_concat') as scope:
        with tf.variable_scope('pass_forward') as scope2:
            hidden = linear(input_hidden_tensor, n_inputs, n_outputs)
            print (len(hidden), str(hidden[0].get_shape()))
            forward = LSTM_cell(hidden, n_outputs)
            print (len(forward), str(forward[0].get_shape()))

        # Backward pass is as simple as surrounding the cell with a double inversion:
        with tf.variable_scope('pass_backward') as scope2:
            hidden = linear(input_hidden_tensor, n_inputs, n_outputs)
            print (len(hidden), str(hidden[0].get_shape()))
            backward = list(reversed(LSTM_cell(list(reversed(hidden)), n_outputs)))
            print (len(backward), str(backward[0].get_shape()))


        # Simply concatenating cells' outputs at each timesteps on the innermost
        # dimension, like if the two cells acted as one cell
        # with twice the n_hidden size:
        layer_hidden_outputs = [
            tf.concat(len(f.get_shape())-1, [f, b])
                for f, b in zip(forward, backward)
        ]

    return layer_hidden_outputs


def add_highway_redisual(layer, residual_minilayer):
    return [a + b for a, b in zip(layer, residual_minilayer)]


def residual_bidirectional_LSTM_layers(input_hidden_tensor, n_input, n_output, layer_level):
    with tf.variable_scope('layer_{}'.format(layer_level)) as scope:

        hidden_LSTM_layer = bidirectional_LSTM_cell(input_hidden_tensor, n_input, n_output)

        # Adding K new residual bidir connections to this first layer:
        for i in range(config.n_residual_layers):
            with tf.variable_scope('LSTM_residual_{}'.format(i)) as scope2:

                hidden_LSTM_layer = add_highway_redisual(
                    hidden_LSTM_layer,
                    bidirectional_LSTM_cell(input_hidden_tensor, n_input, n_output)
                )

    return hidden_LSTM_layer


def LSTM_Network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """

    with tf.variable_scope('LSTM_Network') as scope:  # TensorFlow graph naming

        # Exchange dim 1 and dim 0
        feature_mat = tf.transpose(feature_mat, [1, 0, 2])
        print feature_mat.get_shape()
        # New feature_mat's shape: [time_steps, batch_size, n_inputs]

        # Temporarily crush the feature_mat's dimensions
        feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
        print feature_mat.get_shape()
        # New feature_mat's shape: [time_steps*batch_size, n_inputs]

        # Split the series because the rnn cell needs time_steps features, each of shape:
        hidden = tf.split(0, config.n_steps, feature_mat)
        print (len(hidden), str(hidden[0].get_shape()))
        # New shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

        # Stack two residual bidirectional LSTM cells:

        print "\nCreating hidden 1:"
        hidden = residual_bidirectional_LSTM_layers(hidden, config.n_inputs, config.n_hidden, 1)
        print (len(hidden), str(hidden[0].get_shape()))

        for stacked_hidden_index in range(1, config.n_stacked_layers):

            print "\nCreating hidden {}:".format(stacked_hidden_index+1)
            hidden = residual_bidirectional_LSTM_layers(hidden, 2*config.n_hidden, config.n_hidden, stacked_hidden_index+1)
            print (len(hidden), str(hidden[0].get_shape()))

        print ""

        # Final linear activation logits
        # Get the last output tensor of the inner loop output series, of shape [batch_size, n_classes]
        return linear(
            [hidden[-1]],
            2*config.n_hidden, config.n_classes
        )[0]


def one_hot(label):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    label_num = len(label)
    new_label = label.reshape(label_num)  # shape : [sample_num]
    # because max is 5, and we will create 6 columns
    n_values = np.max(new_label) + 1
    return np.eye(n_values)[np.array(new_label, dtype=np.int32)]


if __name__ == "__main__":

    #-----------------------------
    # step1: load and prepare data
    #-----------------------------
    # Those are separate normalised input features for the neural network
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

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    DATA_PATH = "data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    TRAIN = "train/"
    TEST = "test/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    y_train = one_hot(load_y(y_train_path))
    y_test = one_hot(load_y(y_test_path))

    #-----------------------------------
    # step2: define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    #------------------------------------------------------
    # step3: Let's get serious and build the neural network
    #------------------------------------------------------

    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Loss, optimizer, evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
    # Gradient clipping Adam optimizer with gradient noise
    optimize = tf.contrib.layers.optimize_loss(
        loss,
        global_step=tf.Variable(0),
        learning_rate=config.learning_rate,
        optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
        clip_gradients=config.clip_gradients,
        gradient_noise_scale=config.gradient_noise_scale
    )

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned of for less console spam.
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.initialize_all_variables().run()

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        shuffled_X, shuffled_y = shuffle(X_train, y_train, random_state=i*42)
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            _,train_acc,train_loss=sess.run(
                [optimize,accuracy,loss],
                feed_dict={
                    X: shuffled_X[start:end],
                    Y: shuffled_y[start:end]
                }
            )

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, loss],
            feed_dict={X: X_test, Y: y_test}
        )

        print("train iter: {},".format(i)+\
              "train accuracy: {}".format(train_acc)+\
              "train loss: {}".format(train_loss)+\
              " test accuracy: {},".format(accuracy_out)+\
              " loss: {}".format(loss_out))

        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    #------------------------------------------------------------------
    # step5: Training is good, but having visual insight is even better
    #------------------------------------------------------------------
    # The code is in the .ipynb

    #------------------------------------------------------------------
    # step6: And finally, the multi-class confusion matrix and metrics!
    #------------------------------------------------------------------
    # The code is in the .ipynb
