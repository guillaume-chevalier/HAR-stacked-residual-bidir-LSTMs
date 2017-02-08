# Some parts of the code are taken from: https://github.com/sussexwearlab/DeepConvLSTM

from lstm_architecture import one_hot, run_with_config
from sliding_window import sliding_window

import numpy as np

import cPickle as cp
import time


#--------------------------------------------
# Neural net's config.
#--------------------------------------------

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Data shaping
        self.train_count = len(X_train)  # nb of training series
        self.test_data_count = len(X_test)  # nb of testing series
        self.n_steps = len(X_train[0])  # nb of time_steps per series
        self.n_classes = 18  # Final output classes, one classification per series

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 100
        self.batch_size = 100
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None
        self.keep_prob_for_dropout = 0.85  # **(1/3.0)  # Dropout is added on inputs and after each stacked layers (but not between residual layers).

        # Linear+relu structure
        self.bias_mean = 0.3
        self.weights_stddev = 0.2  # I would recommend between 0.1 and 1.0 or to change and use a xavier initializer

        ########
        # NOTE: I think that if any of the below parameters are changed,
        # the best is to readjust every parameters in the "Training" section
        # above to properly compare the architectures only once optimised.
        ########

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count
        self.n_hidden = 28  # nb of neurons inside the neural network
        self.use_bidirectionnal_cells = True  # Use bidir in every LSTM cell, or not:

        # High-level deep architecture
        self.also_add_dropout_between_stacked_cells = False  # True
        # NOTE: values of exactly 1 (int) for those 2 high-level parameters below totally disables them and result in only 1 starting LSTM.
        # self.n_layers_in_highway = 1  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        # self.n_stacked_layers = 1  # Stack multiple blocks of residual layers.


#--------------------------------------------
# Dataset-specific constants and functions
#--------------------------------------------

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
NB_SENSOR_CHANNELS_WITH_FILTERING = 149

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)
SLIDING_WINDOW_STEP_SHORT = SLIDING_WINDOW_STEP

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128


def load_dataset(filename):

    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

assert (NB_SENSOR_CHANNELS_WITH_FILTERING == X_train.shape[1] or NB_SENSOR_CHANNELS == X_train.shape[1])

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    data_x, data_y = data_x.astype(np.float32), one_hot(data_y.reshape(len(data_y)).astype(np.uint8))
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
    return data_x, data_y


#--------------------------------------------
# Loading dataset
#--------------------------------------------


# Sensor data is segmented using a sliding window mechanism
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP_SHORT)
X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

for mat in [X_train, y_train, X_test, y_test]:
    print mat.shape


#--------------------------------------------
# Training (maybe multiple) experiment(s)
#--------------------------------------------

n_layers_in_highway = 3
n_stacked_layers = 3
trial_name = "{}x{}".format(n_layers_in_highway, n_stacked_layers)

for learning_rate in [0.001]:
    for lambda_loss_amount in [0.005]:
        print "learning_rate: {}".format(learning_rate)
        print "lambda_loss_amount: {}".format(lambda_loss_amount)
        print ""

        class EditedConfig(Config):
            def __init__(self, X, Y):
                super(EditedConfig, self).__init__(X, Y)

                # Edit only some parameters:
                self.learning_rate = learning_rate
                self.lambda_loss_amount = lambda_loss_amount
                # Architecture params:
                self.n_layers_in_highway = n_layers_in_highway
                self.n_stacked_layers = n_stacked_layers

        accuracy_out, best_accuracy, f1_score_out, best_f1_score = run_with_config(EditedConfig, X_train, y_train, X_test, y_test)
        print (accuracy_out, best_accuracy, f1_score_out, best_f1_score)

        with open('{}_result_opportunity_18.txt'.format(trial_name),'a') as f:
            f.write("""str(learning_rate)+' \t'+str(lambda_loss_amount)+' \t'+str(accuracy_out)+' \t'+str(best_accuracy)+' \t'+str(f1_score_out)+' \t'+str(best_f1_score)\n""")
            f.write(   str(learning_rate)+' \t'+str(lambda_loss_amount)+' \t'+str(accuracy_out)+' \t'+str(best_accuracy)+' \t'+str(f1_score_out)+' \t'+str(best_f1_score)+'\n\n' )

        print "________________________________________________________"
    print ""
print "Done."
