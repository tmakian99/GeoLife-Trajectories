import geopy.distance
import pandas as pd
import numpy as np
import math

df_labels = pd.read_pickle('df_labels.pkl')
df_labels.drop(['Altitude', 'Reserved_1', 'Reserved_2', 'user'], axis=1, inplace=True)
df_labels = df_labels[df_labels['label'] != '']
df_labels.dropna(subset=['label'], inplace=True)

# add sequences column
df_labels.loc[:, 'sequence'] = (df_labels['label'] != df_labels['label'].shift()).cumsum()
# split sequence if timedelta makes clear it's a different trip
df_labels.loc[:, 'timedelta'] = (df_labels['Datetime'] - df_labels['Datetime'].shift()).fillna(
    pd.Timedelta('0 days')).dt.seconds

df_labels.loc[:, 'new_sequence'] = (df_labels['timedelta'] > 60).cumsum()
df_labels.loc[:, 'sequence'] = df_labels['sequence'] + df_labels['new_sequence']
df_labels = df_labels[df_labels.Latitude < 90]
df_labels['coordinate'] = list(zip(df_labels['Latitude'], df_labels['Longitude']))
df_labels['prev_coordinate'] = df_labels['coordinate'].shift(1)
df_labels['distance'] = \
    df_labels.apply(lambda x: geopy.distance.geodesic(x['coordinate'], x['prev_coordinate']).km, axis=1)


# split into sequences of 100 datapoints
def split_dataframe(df, chunk_size=100):
    chunks = list()
    num_chunks = math.ceil(len(df) / chunk_size)
    for j in range(num_chunks):
        chunks.append(df.iloc[j * chunk_size:(j + 1) * chunk_size])
    return chunks


def create_sequences():
    sequence_arrays = []
    labels = []
    for i in range(1, 27565):
        new_sequence = df_labels[df_labels['sequence'] == i]
        for sequence in split_dataframe(new_sequence):
            if len(sequence) == 100:
                df_sequence = pd.DataFrame(sequence)
                # encode timedelta information
                df_sequence.loc[:, 'timedelta'] = (df_sequence['Datetime'] - df_sequence['Datetime'].shift()) \
                    .fillna(pd.Timedelta('0 days')).dt.seconds.cumsum()
                df_sequence.at[0, 'distance'] = 0
                sequence_arrays.append(np.array([
                    sequence['Longitude'],
                    sequence['Latitude'],
                    sequence['timedelta'],
                    sequence['distance']]))
                labels.append(sequence['label'].iloc[0])

    sequence_arrays = np.array(sequence_arrays)
    return sequence_arrays, labels
