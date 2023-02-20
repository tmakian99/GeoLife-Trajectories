import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from prepare_data import prepare_data
from transformer_classifier import build_model as transformer_build_model
from lstm_time_series import build_model as lstm_build_model

x_train, y_train, x_test, y_test, scalers, num_classes = prepare_data('evaluate')

lstm_model = lstm_build_model((4, 99), 4)
transformer_model = transformer_build_model((5, 100),
                                            num_heads=4,
                                            num_transformer_blocks=4,
                                            dropout=0.25)

lstm_model.load_weights('lstm_model.h5')
transformer_model.load_weights('transformer_model.h5')

x_test_lstm = x_test[:, :4, :99]
x_test_transformer = x_test[:, :, :]

y_pred_location = lstm_model.predict(x_test_lstm)
y_pred_transportation_mode = transformer_model.predict(x_test_transformer)

# replace last location data in each of the test sequences with the lstm prediction
x_test[:, :4, 99] = y_pred_location

for i in range(x_test.shape[1]):
    x_test[:, i, :] = scalers[i].inverse_transform(x_test[:, i, :])

label_dict = {'walk': 0, 'motor vehicle': 1, 'bike': 2, 'train': 3}
inv_map = {v: k for k, v in label_dict.items()}
y_pred_transformer = np.argmax(y_pred_transportation_mode, axis=1)
y_pred_transformer = np.array([inv_map.get(item, item) for item in y_pred_transformer])

# select random sequence to display
rand_index = np.random.randint(0, len(x_test))
df_tableau = pd.DataFrame(x_test[rand_index, :, :].transpose(), columns=['x', 'y', 'z', 'timedelta', 'distance'])
df_tableau['mode_prediction'] = y_pred_transformer[rand_index]

R = 6371
df_tableau['Latitude'] = np.degrees(np.arcsin(df_tableau['z']/R))
df_tableau['Longitude'] = np.degrees(np.arctan2(df_tableau['y'], df_tableau['x']))

df_tableau.drop(['x', 'y', 'z'], axis=1, inplace=True)
df_tableau['timestamp'] = df_tableau.apply(lambda x: datetime.strptime('2007-08-04 03:30:32', '%Y-%m-%d %H:%M:%S') + timedelta(seconds=x['timedelta']), axis=1)

df_tableau.to_csv('tableau_data.csv')
