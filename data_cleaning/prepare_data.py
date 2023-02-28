import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def prepare_data(model_type):
    # Create sequences
    data_files = np.load('./data/seq_labels.npz')
    sequences = data_files['arr_0']
    labels = data_files['arr_1']

    # drop altitude, only train on x, y, z, time and distance
    x = sequences
    # Group similar modes of transportation, and drop very infrequent modes
    label_dict = {'walk': 0, 'bus': 1, 'bike': 2, 'train': 3, 'car': 1, 'subway': 3, 'taxi': 1,
                  'run': 0, 'motorcycle': 1}
    y = np.array([label_dict.get(item, item) for item in labels])

    # Split data into train and test, and hold test until after training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    num_classes = len(np.unique(y))

    # shuffle the training data
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # normalize the data
    scalers = {}
    for i in range(x_train.shape[1]):
        scalers[i] = MinMaxScaler()
        x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :])

    for i in range(x_test.shape[1]):
        x_test[:, i, :] = scalers[i].transform(x_test[:, i, :])

    if model_type == 'lstm':
        # prepare for lstm training
        # drop distance, only train on x, y, z, altitude and time
        # split sequences into train sequence and corresponding test sequence
        y_train = x_train[:, :4, 99:]
        x_train = x_train[:, :4, :99]
        y_test = x_test[:, :4, 99:]
        x_test = x_test[:, :4, :99]

    return x_train, y_train, x_test, y_test, scalers, num_classes
