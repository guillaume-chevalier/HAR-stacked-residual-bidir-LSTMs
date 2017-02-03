# Adapted from: https://github.com/sussexwearlab/DeepConvLSTM
__author__ = 'fjordonez, gchevalier'

from signal_filtering import filter_opportunity_datasets_accelerometers

import os
import zipfile
import argparse
import numpy as np
import cPickle as cp

from io import BytesIO
from pandas import Series

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
NB_SENSOR_CHANNELS_WITH_FILTERING = 149 # =77 gyros +36*2 accelerometer channels

# Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
OPPORTUNITY_DATA_FILES_TRAIN = [
    'OpportunityUCIDataset/dataset/S1-Drill.dat',
    'OpportunityUCIDataset/dataset/S1-ADL1.dat',
    'OpportunityUCIDataset/dataset/S1-ADL2.dat',
    'OpportunityUCIDataset/dataset/S1-ADL3.dat',
    'OpportunityUCIDataset/dataset/S1-ADL4.dat',
    'OpportunityUCIDataset/dataset/S1-ADL5.dat',
    'OpportunityUCIDataset/dataset/S2-Drill.dat',
    'OpportunityUCIDataset/dataset/S2-ADL1.dat',
    'OpportunityUCIDataset/dataset/S2-ADL2.dat',
    'OpportunityUCIDataset/dataset/S2-ADL3.dat',
    'OpportunityUCIDataset/dataset/S3-Drill.dat',
    'OpportunityUCIDataset/dataset/S3-ADL1.dat',
    'OpportunityUCIDataset/dataset/S3-ADL2.dat',
    'OpportunityUCIDataset/dataset/S3-ADL3.dat'
]

OPPORTUNITY_DATA_FILES_TEST = [
    'OpportunityUCIDataset/dataset/S2-ADL4.dat',
    'OpportunityUCIDataset/dataset/S2-ADL5.dat',
    'OpportunityUCIDataset/dataset/S3-ADL4.dat',
    'OpportunityUCIDataset/dataset/S3-ADL5.dat'
]

def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: tuple((numpy integer 2D matrix, numpy integer 1D matrix))
        (Selection of features (N, f), feature_is_accelerometer (f,) one-hot)
    """

    # In term of column_names.txt's ranges: excluded-included (here 0-indexed)
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])

    # In term of column_names.txt's ranges: excluded-included
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])

    # In term of column_names.txt's ranges: excluded-included
    features_acc = np.arange(1, 37)
    features_acc = np.concatenate([features_acc, np.arange(134, 194)])
    features_acc = np.concatenate([features_acc, np.arange(207, 231)])

    # One-hot for everything that is an accelerometer
    is_accelerometer = np.zeros([243])
    is_accelerometer[features_acc] = 1

    # Deleting some signals to keep only the 113 of the challenge
    data = np.delete(data, features_delete, 1)
    is_accelerometer = np.delete(is_accelerometer, features_delete, 0)

    # Shape `(N, f), (f, )`
    # where N is number of timesteps and f is 113 features, one-hot
    return data, is_accelerometer


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.

    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001
    x /= (std * 2)  # 2 is for having smaller values
    return x

def split_data_into_time_gyros_accelerometers(data, is_accelerometer):
    # Assuming index 0 of features is reserved for time.
    # Splitting data into gyros, accelerometers and time:

    is_accelerometer = np.array(is_accelerometer*2-1, dtype=np.int32)
    # is_accelerometer's zeros have been replaced by -1. 1's are untouched.
    plane = np.arange(len(is_accelerometer)) * is_accelerometer
    delete_gyros = [-e for e in plane if e <= 0]
    delete_accms = [ e for e in plane if e >= 0]

    time  = data[:,0]
    gyros = np.delete(data, delete_accms, 1)
    accms = np.delete(data, delete_gyros, 1)
    return time, gyros, accms

def divide_x_y(data, label, filter_accelerometers):
    """Segments each sample into (time+features) and (label)

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    if filter_accelerometers:
        data_x = data[:, :114]
    else:
        data_x = data[:,1:114]

    # Choose labels type for y
    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, 114]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, 115]  # Gestures label

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def check_data(data_set):
    """Try to access to the file and checks if dataset is in the data directory
       In case the file is not found try to download it from original location

    :param data_set:
            Path with original OPPORTUNITY zip file
    :return:
    """
    print 'Checking dataset {0}'.format(data_set)
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    # When dataset not found, try to download it from UCI repository
    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
        print '... dataset path {0} not found'.format(data_set)
        import urllib
        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        )
        if not os.path.exists(data_dir):
            print '... creating directory {0}'.format(data_dir)
            os.makedirs(data_dir)
        print '... downloading data from {0}'.format(origin)
        urllib.urlretrieve(origin, data_set)

    return data_dir


def process_dataset_file(data, label, filter_accelerometers):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data, is_accelerometer = select_columns_opp(data)

    # Colums are segmentd into features and labels
    data_x, data_y =  divide_x_y(data, label, filter_accelerometers)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation (a.k.a. filling in NaN)
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x)

    if filter_accelerometers:
        # x's accelerometers, are filtered out by some LP passes for noise and gravity.
        # Time is discarded, accelerometers are filtered to
        # split gravity and remove noise.
        _, x_gyros, x_accms = split_data_into_time_gyros_accelerometers(
            data_x, is_accelerometer
        )
        print "gyros' shape: {}".format(x_gyros.shape)
        print "old accelerometers' shape: {}".format(x_accms.shape)
        x_accms = normalize(filter_opportunity_datasets_accelerometers(x_accms))
        print "new accelerometers' shape: {}".format(x_accms.shape)
        # Put features together (inner concatenation with transposals)

        data_x = np.hstack([x_gyros, x_accms])
        print "new total shape: {}".format(data_x.shape)

    return data_x, data_y


def load_data_files(zipped_dataset, label, data_files, filter_accelerometers=False):
    """Loads specified data files' features (x) and labels (y)

    :param zipped_dataset: ZipFile
        OPPORTUNITY zip file to read from
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    :param data_files: list of strings
        Data files to load.
    :return: numpy integer matrix, numy integer array
        Loaded sensor data, segmented into features (x) and labels (y)
    """

    nb_sensors = NB_SENSOR_CHANNELS_WITH_FILTERING if filter_accelerometers else NB_SENSOR_CHANNELS
    data_x = np.empty((0, nb_sensors))
    data_y = np.empty((0))

    for filename in data_files:
        try:
            data = np.loadtxt(BytesIO(zipped_dataset.read(filename)))
            print '... file {0}'.format(filename)
            x, y = process_dataset_file(data, label, filter_accelerometers)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
            print "Data's shape yet: "
            print data_x.shape
        except KeyError:
            print 'ERROR: Did not find {0} in zip file'.format(filename)

    return data_x, data_y


def generate_data(dataset, target_filename, label):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """

    data_dir = check_data(dataset)
    zf = zipfile.ZipFile(dataset)

    print '\nProcessing train dataset files...\n'
    X_train, y_train = load_data_files(zf, label, OPPORTUNITY_DATA_FILES_TRAIN)
    print '\nProcessing test dataset files...\n'
    X_test,  y_test  = load_data_files(zf, label, OPPORTUNITY_DATA_FILES_TEST)

    print "Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape)

    obj = [(X_train, y_train), (X_test, y_test)]
    f = file(os.path.join(data_dir, target_filename), 'wb')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()


def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    parser.add_argument(
        '-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument(
        '-o', '--output', type=str, help='Processed data file', required=True)
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Type of activities to be recognized', default="gestures", choices = ["gestures", "locomotion"], required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    dataset = args.input
    target_filename = args.output
    label = args.task
    # Return all variable values
    return dataset, target_filename, label

if __name__ == '__main__':

    OpportunityUCIDataset_zip, output, l = get_args();
    generate_data(OpportunityUCIDataset_zip, output, l)
