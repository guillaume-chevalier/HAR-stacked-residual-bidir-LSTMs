
from lstm_architecture import load_X, load_Y, run_with_config

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
        self.training_epochs = 500
        self.batch_size = 100
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None
        self.keep_prob_for_dropout = 0.85  # Dropout is added on inputs and after each stacked layers (but not between residual layers).

        # Linear+relu structure
        self.bias_mean = 0.0  # I would recommend to try 0.0 or 1.0
        self.weights_stddev = 0.2  # I would recommend between 0.1 and 1.0 or to change and use a xavier initializer

        ########
        # NOTE: I think that if any of the below parameters are changed,
        # the best is to readjust every parameters in the "Training" section
        # above to properly compare the architectures only once optimised.
        ########

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 28  # nb of neurons inside the neural network
        self.use_bidirectionnal_cells = False  # Use bidir in every LSTM cell, or not:

        # High-level deep architecture
        self.also_add_dropout_between_stacked_cells = False
        self.n_residual_layers = 0  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        self.n_stacked_layers = 0  # Stack multiple blocks of residual layers.
        # NOTE: values of exactly 0 (int) for those last 2 high-level parameters totally disables them and result in only 1 starting LSTM.


# Train
n_residual_layers = 0
n_stacked_layers = 0
trial_name = "{}x{}".format(n_residual_layers, n_stacked_layers)

for learning_rate in [0.01, 0.007, 0.001, 0.0007]:
    for lambda_loss_amount in [0.01, 0.005, 0.001]:
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
                self.n_residual_layers = n_residual_layers
                self.n_stacked_layers = n_stacked_layers

        try:
            accuracy_out, best_accuracy = run_with_config(EditedConfig)
        except:
            accuracy_out, best_accuracy = -1, -1
        print (accuracy_out, best_accuracy)

        with open('{}_result.txt'.format(trial_name),'a') as f:
            f.write(str(learning_rate)+'\t'+str(lambda_loss_amount)+'\t'+str(accuracy_out)+'\t'+str(best_accuracy)+'\n\n')

        print "________________________________________________________"
    print ""
print "Done."
