
from lstm_architecture import one_hot, run_with_config

import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL

from bson import json_util
import json
import os
import pickle
import traceback

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
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        self.n_classes = 6  # Final output classes

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 200
        self.batch_size = 100
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None
        # Dropout is added on inputs and after each stacked layers (but not
        # between residual layers).
        self.keep_prob_for_dropout = 0.85  # **(1/3.0)

        # Linear+relu structure
        self.bias_mean = 0.3
        # I would recommend between 0.1 and 1.0 or to change and use a xavier
        # initializer
        self.weights_stddev = 0.2

        ########
        # NOTE: I think that if any of the below parameters are changed,
        # the best is to readjust every parameters in the "Training" section
        # above to properly compare the architectures only once optimised.
        ########

        # LSTM structure
        # Features count is of 9: three 3D sensors features over time
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 28  # nb of neurons inside the neural network
        # Use bidir in every LSTM cell, or not:
        self.use_bidirectionnal_cells = False

        # High-level deep architecture
        self.also_add_dropout_between_stacked_cells = False  # True
        # NOTE: values of exactly 1 (int) for those 2 high-level parameters below totally disables them and result in only 1 starting LSTM.
        # self.n_layers_in_highway = 1  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        # self.n_stacked_layers = 1  # Stack multiple blocks of residual
        # layers.


#--------------------------------------------
# Dataset-specific constants and functions + loading
#--------------------------------------------

# Useful Constants

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

TRAIN = "train/"
TEST = "test/"


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    """
    Given attribute (train or test) of feature, read all 9 features into an
    np ndarray of shape [sample_sequence_idx, time_step, feature_num]
        argument:   X_signals_paths str attribute of feature: 'train' or 'test'
        return:     np ndarray, tensor of features
    """
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

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    """
    Read Y file of values to be predicted
        argument: y_path str attibute of Y: 'train' or 'test'
        return: Y ndarray / tensor of the 6 one_hot labels of each sample
    """
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
    return one_hot(y_ - 1)

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)


#--------------------------------------------
# Training (maybe multiple) experiment(s)
#--------------------------------------------

def fine_tune(hyperparams):
    try:
        class EditedConfig(Config):
            def __init__(self, X, Y):
                super(EditedConfig, self).__init__(X, Y)
                # Edit some parameters:
                self.learning_rate = self.learning_rate * hyperparams["lr_rate_multiplier"]
                self.lambda_loss_amount = self.lambda_loss_amount * hyperparams["l2_reg_multiplier"]
                self.n_hidden = int(self.n_hidden * hyperparams["n_hidden_multiplier"])
                # Set anew other parameters:
                self.clip_gradients = hyperparams["clip_gradients"]
                self.keep_prob_for_dropout = hyperparams["dropout_keep_probability"]
                self.n_layers_in_highway = hyperparams["n_layers_in_highway"]
                self.n_stacked_layers = hyperparams["n_stacked_layers"]
                self.bias_mean = hyperparams["bias_mean"]
                self.weights_stddev = hyperparams["weights_stddev"]
                self.use_bidirectionnal_cells = hyperparams["use_bidirectionnal_cells"]
                self.also_add_dropout_between_stacked_cells = hyperparams["also_add_dropout_between_stacked_cells"]

        print("Hyperparams:")
        print(hyperparams)

        accuracy_out, best_accuracy, f1_score_out, best_f1_score = (
            run_with_config(EditedConfig, X_train, y_train, X_test, y_test)
        )
        trial_name = "model_{}x{}_{}_{}".format(
            n_layers_in_highway, n_stacked_layers, best_accuracy, best_f1_score)

        print (accuracy_out, best_accuracy, f1_score_out, best_f1_score)

        results = {
            "loss": -best_accuracy[0],
            "status": STATUS_OK,
            "space": hyperparams,
            "accuracy_end": accuracy_out,
            "accuracy_best": best_accuracy,
            "f1_score_end": f1_score_out,
            "f1_score_best": best_f1_score
        }

    except Exception as err:
        try:
            tf.get_default_session().close()
        except:
            pass
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        results = {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }

    print("RESULTS:")
    print(json.dumps(
        results,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))
    # Save all training results to disks with unique filenames
    if not os.path.exists("results_6/"):
        os.makedirs("results_6/")
    with open('results_6/{}.txt.json'.format(trial_name), 'w') as f:
        json.dump(
            results, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )

    print("\n\n")
    return results


space = {
    "n_layers_in_highway": hp.choice("n_layers_in_highway", [1, 2, 3]),
    "n_stacked_layers": hp.choice("n_stacked_layers", [1, 2, 3]),
    "lr_rate_multiplier": hp.loguniform("lr_rate_multi", -0.3, 0.3),
    "l2_reg_multiplier": hp.loguniform("l2_multi", -0.3, 0.3),
    "n_hidden_multiplier": hp.loguniform("n_hidden_multiplier", -0.3, 0.3),
    "clip_gradients": hp.choice("clip_multi", [5., 10., 15., 20.]),
    "dropout_keep_probability": hp.uniform("dropout_multi", 0.5, 1.0),
    "bias_mean": hp.uniform("bias_mean", 0.0, 1.0),
    "weights_stddev": hp.uniform("weights_stddev", 0.05, 0.5),
    "use_bidirectionnal_cells": hp.choice("use_bidirectionnal_cells", [False, True]),
    "also_add_dropout_between_stacked_cells": hp.choice("also_add_dropout_between_stacked_cells", [False, True])
}


def run_a_trial():
    """
    Run one TPE meta optimisation step and save its results.

    See:
    https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/blob/master/optimize.py
    """
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results_6.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        fine_tune,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results_6.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print("Best results yet (note that this is NOT calculated on the 'loss' "
          "metric despite the key is 'loss' - we rather take the negative "
          "best accuracy throughout learning as a metric to minimize):")
    print(best)


if __name__ == "__main__":
    """
    Plot the model and run the optimisation forever (and saves results).

    See:
    https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/blob/master/optimize.py
    """

    print("Here we train many models, one after the other.")

    print("\nResults will be saved in the folder named 'results_6/'. "
          "it is possible to sort them alphabetically and choose the "
          "best loss for each architecture. As you run the "
          "optimization, results are consinuously saved into a "
          "'results.pkl' file, too. Re-running optimize.py will resume "
          "the meta-optimization.\n")

    while True:
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)
