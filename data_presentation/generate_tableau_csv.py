import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_cleaning.prepare_data import prepare_data
from training_models.transformer_classifier import build_model as transformer_build_model
from training_models.lstm_time_series import build_model as lstm_build_model


def generate_tableau_csv():
    x_train, y_train, x_test, y_test, scalers, num_classes = prepare_data('evaluate')

    lstm_model = lstm_build_model((4, 99), 4)
    transformer_model = transformer_build_model((5, 100),
                                                num_heads=4,
                                                num_transformer_blocks=4,
                                                dropout=0.25)

    lstm_model.load_weights('lstm_model.h5')
    transformer_model.load_weights('transformer_model.h5')

    x_test_lstm = x_test[:, :4, :99]
    for idx in range(0, 10):
        x_test_pred = x_test_lstm[:, :, idx:(99 + idx)]
        y_pred = lstm_model.predict(x_test_pred)
        x_test_lstm = np.concatenate((x_test_lstm, y_pred[..., None]), axis=2)

    y_pred_location = x_test_lstm[:, :, 99:109]
    zeros = np.zeros((7436, 4, 90))
    y = np.concatenate((zeros, y_pred_location), axis=2)

    x_test_transformer = x_test[:, :, :]
    y_pred_transportation_mode = transformer_model.predict(x_test_transformer)

    for i in range(y.shape[1]):
        x_test[:, i, :] = scalers[i].inverse_transform(x_test[:, i, :])
        y[:, i, :] = scalers[i].inverse_transform(y[:, i, :])

    label_dict = {'walk': 0, 'motor vehicle': 1, 'bike': 2, 'train': 3}
    inv_map = {v: k for k, v in label_dict.items()}
    y_pred_transformer = np.argmax(y_pred_transportation_mode, axis=1)
    y_pred_transformer = np.array([inv_map.get(item, item) for item in y_pred_transformer])

    # find best estimate to display
    idx = 0
    diff = 1000
    x_compare = x_test[:, :3, 99]
    y_compare = y[:, :3, 90]
    for n in range(x_compare.shape[0]):
        if np.linalg.norm(y_compare[n, :] - x_compare[n, :]) < diff:
            idx = n
            diff = np.linalg.norm(y_compare[n, :] - x_compare[n, :])
    x_test = np.concatenate((x_test[:, :4, :99], y[:, :, 90:]), axis=2)

    # select random sequence to display
    # rand_index = np.random.randint(0, len(x_test))
    rand_index = idx
    df_tableau = pd.DataFrame(x_test[rand_index, :, :].transpose(), columns=['x', 'y', 'z', 'timedelta'])
    df_tableau['mode_prediction'] = y_pred_transformer[rand_index]

    R = 6371
    df_tableau['Latitude'] = np.degrees(np.arcsin(df_tableau['z']/R))
    df_tableau['Longitude'] = np.degrees(np.arctan2(df_tableau['y'], df_tableau['x']))

    df_tableau.drop(['x', 'y', 'z'], axis=1, inplace=True)
    df_tableau['timestamp'] = df_tableau.apply(lambda x: datetime.strptime('2007-08-04 03:30:32', '%Y-%m-%d %H:%M:%S') + timedelta(seconds=x['timedelta']), axis=1)
    df_tableau['time_diff'] = (df_tableau['timedelta']-df_tableau['timedelta'].shift()).fillna(0)

    df_tableau.to_csv('results/tableau_data.csv')


if __name__ == '__main__':
    generate_tableau_csv()
