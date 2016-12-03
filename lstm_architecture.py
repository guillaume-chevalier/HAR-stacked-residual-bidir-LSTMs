
import tensorflow as tf

from sklearn.utils import shuffle
import numpy as np

import os


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


def one_hot(Y):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    return np.eye(6)[np.array(Y)]


def batch_norm(input_tensor, config):
    # Implementing batch normalisation: this is used out of the residual layers
    # to normalise those output neurons by mean and standard deviation.

    if config.n_residual_layers == 1:
        # There is no residual layers, no need for batch_norm:
        return input_tensor

    with tf.variable_scope("batch_norm") as scope:
        # Mean and variance simply crunched over all axes
        axes = list(range(len(input_tensor.get_shape())))

        mean, variance = tf.nn.moments(input_tensor, axes=axes, shift=None, name=None, keep_dims=False)
        stdev = tf.sqrt(variance+0.001)

        # Rescaling
        bn = input_tensor - mean
        bn /= (stdev + 0.0001)
        # Learnable extra rescaling
        bn *= tf.Variable(1.0, name=(scope.name + "/a"))
        bn += tf.Variable(0.0, name=(scope.name + "/b"))

    return bn

def relu_fc(input_2D_tensor_list, features_len, new_features_len, config):
    """make a relu fully-connected layer, mainly change the shape of tensor
       both input and output is a list of tensor
        argument:
            input_2D_tensor_list: list shape is [batch_size,feature_num]
            features_len: int the initial features length of input_2D_tensor
            new_feature_len: int the final features length of output_2D_tensor
            config: Config used for weights initializers
        return:
            output_2D_tensor_list lit shape is [batch_size,new_feature_len]
    """

    W = tf.get_variable(
        "relu_fc_weights",
        initializer=tf.random_normal(
            [features_len, new_features_len],
            mean=0.0,
            stddev=float(config.weights_stddev)
        )
    )
    b = tf.get_variable(
        "relu_fc_biases",
        initializer=tf.random_normal(
            [new_features_len],
            mean=float(config.bias_mean),
            stddev=float(config.weights_stddev)
        )
    )

    # intra-timestep multiplication:
    output_2D_tensor_list = [
        tf.nn.relu(tf.matmul(input_2D_tensor, W) + b)
            for input_2D_tensor in input_2D_tensor_list
    ]

    return output_2D_tensor_list


def single_LSTM_cell(input_hidden_tensor, n_outputs):
    """ define the basic LSTM layer
        argument:
            input_hidden_tensor: list a list of tensor,
                                 shape: time_steps*[batch_size,n_inputs]
            n_outputs: int num of LSTM layer output
        return:
            outputs: list a time_steps list of tensor,
                     shape: time_steps*[batch_size,n_outputs]
    """
    with tf.variable_scope("lstm_cell"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_outputs, forget_bias=1.0)
        outputs, _ = tf.nn.rnn(lstm_cell, input_hidden_tensor, dtype=tf.float32)
    return outputs


def bi_LSTM_cell(input_hidden_tensor, n_inputs, n_outputs, config):
    """build bi-LSTM, concatenating the two directions in an inner manner.
        argument:
            input_hidden_tensor: list a time_steps series of tensor, shape: [sample_num, n_inputs]
            n_inputs: int units of input tensor
            n_outputs: int units of output tensor, each bi-LSTM will have half those internal units
            config: Config used for the relu_fc
        return:
            layer_hidden_outputs: list a time_steps series of tensor, shape: [sample_num, n_outputs]
    """
    n_outputs = int(n_outputs/2)

    print "bidir:"
    print (len(input_hidden_tensor), str(input_hidden_tensor[0].get_shape()))

    with tf.variable_scope('pass_forward') as scope2:
        hidden_forward = relu_fc(input_hidden_tensor, n_inputs, n_outputs, config)
        forward = single_LSTM_cell(hidden_forward, n_outputs)


    # Backward pass is as simple as surrounding the cell with a double inversion:
    with tf.variable_scope('pass_backward') as scope2:
        hidden_backward = relu_fc(input_hidden_tensor, n_inputs, n_outputs, config)
        backward = list(reversed(single_LSTM_cell(list(reversed(hidden_backward)), n_outputs)))

    with tf.variable_scope('bidir_concat') as scope:
        # Simply concatenating cells' outputs at each timesteps on the innermost
        # dimension, like if the two cells acted as one cell
        # with twice the n_hidden size:
        layer_hidden_outputs = [
            tf.concat(len(f.get_shape()) - 1, [f, b])
                for f, b in zip(forward, backward)]

    return layer_hidden_outputs


def residual_bidirectional_LSTM_layers(input_hidden_tensor, n_input, n_output, layer_level, config):
    """This architecture is only enabled if "config.n_residual_layers" has a
    value only greater than int(0). The arguments are same than for bi_LSTM_cell.
    arguments:
        input_hidden_tensor: list a time_steps series of tensor, shape: [sample_num, n_inputs]
        n_inputs: int units of input tensor
        n_outputs: int units of output tensor, each bi-LSTM will have half those internal units
        config: Config used for determining if there are residual connections and if yes, their number and with some batch_norm.
    return:
        layer_hidden_outputs: list a time_steps series of tensor, shape: [sample_num, n_outputs]
    """
    with tf.variable_scope('layer_{}'.format(layer_level)) as scope:

        if config.use_bidirectionnal_cells:
            get_lstm = lambda input_tensor: bi_LSTM_cell(input_tensor, n_input, n_output, config)
        else:
            get_lstm = lambda input_tensor: single_LSTM_cell(relu_fc(input_tensor, n_input, n_output, config), n_output)
        def add_highway_redisual(layer, residual_minilayer):
            return [a + b for a, b in zip(layer, residual_minilayer)]

        hidden_LSTM_layer = get_lstm(input_hidden_tensor)
        # Adding K new (residual bidir) connections to this first layer:
        for i in range(config.n_residual_layers):
            with tf.variable_scope('LSTM_residual_{}'.format(i)) as scope2:
                hidden_LSTM_layer = add_highway_redisual(
                    hidden_LSTM_layer,
                    get_lstm(input_hidden_tensor)
                )

    if config.also_add_dropout_between_stacked_cells:
        hidden_LSTM_layer = [tf.nn.dropout(out, keep_prob_for_dropout) for out in hidden_LSTM_layer]

    return [batch_norm(out, config) for out in hidden_LSTM_layer]


def LSTM_network(feature_mat, config, keep_prob_for_dropout):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : ndarray  output shape [batch_size,n_classes]
    """

    with tf.variable_scope('LSTM_network') as scope:  # TensorFlow graph naming

        feature_mat = tf.nn.dropout(feature_mat, keep_prob_for_dropout)

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

        # Stacking LSTM cells, at least one is stacked:
        print "\nCreating hidden #1:"
        hidden = residual_bidirectional_LSTM_layers(hidden, config.n_inputs, config.n_hidden, 1, config)
        print (len(hidden), str(hidden[0].get_shape()))

        for stacked_hidden_index in range(config.n_stacked_layers):
            # If the config permits it, we stack more lstm cells:
            print "\nCreating hidden #{}:".format(stacked_hidden_index+2)
            hidden = residual_bidirectional_LSTM_layers(hidden, config.n_hidden, config.n_hidden, stacked_hidden_index+2, config)
            print (len(hidden), str(hidden[0].get_shape()))

        print ""

        # Final fully-connected activation logits
        # Get the last output tensor of the inner loop output series, of shape [batch_size, n_classes]
        last_hidden = tf.nn.dropout(hidden[-1], keep_prob_for_dropout)
        last_logits = relu_fc(
            [last_hidden],
            config.n_hidden, config.n_classes, config
        )[0]
        return last_logits


def run_with_config(Config):

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
    Y_train = one_hot(Y_train)  # shape [sample_num, n_classes]=[2947, 6]
    Y_test = one_hot(Y_test)  # shape [sample_num, n_classes]=[2947, 6]

    #-----------------------------------
    # step2: define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, Y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    #------------------------------------------------------
    # step3: Let's get serious and build the neural network
    #------------------------------------------------------
    with tf.device("/cpu:0"):
        X = tf.placeholder(tf.float32, [
                           None, config.n_steps, config.n_inputs], name="X")
        Y = tf.placeholder(tf.float32, [
                           None, config.n_classes], name="Y")

        # is_train for dropout control:
        is_train = tf.placeholder(tf.bool, name="is_train")
        keep_prob_for_dropout = tf.cond(is_train,
            lambda: tf.constant(
                config.keep_prob_for_dropout,
                name="keep_prob_for_dropout"
            ),
            lambda: tf.constant(
                1.0,
                name="keep_prob_for_dropout"
            )
        )

        pred_Y = LSTM_network(X, config, keep_prob_for_dropout)

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
    sess=tf.InteractiveSession(
        config=tf.ConfigProto(log_device_placement=False))
    tf.initialize_all_variables().run()

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        shuffled_X, shuffled_Y = shuffle(X_train, Y_train, random_state=i*42)
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            _, train_acc, train_loss = sess.run(
                [optimize,accuracy,loss],
                feed_dict={
                    X: shuffled_X[start:end],
                    Y: shuffled_Y[start:end],
                    is_train: True
                }
            )

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, loss],
            feed_dict={
                X: X_test,
                Y: Y_test,
                is_train: False
            }
        )

        print("train iter: {}, ".format(i)+\
              "train accuracy: {}, ".format(train_acc)+\
              "train loss: {}, ".format(train_loss)+\
              "test accuracy: {}, ".format(accuracy_out)+\
              "test loss: {}".format(loss_out))

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

    return accuracy_out, best_accuracy
