import os
import numpy as np
import pandas as pd

# Create an empty list to store the DataFrames
df_list = []

# Loop over each of the 181 folders
for i in range(182):
    # Create the folder path for each folder
    folder_path = "./Data/{:03d}/Trajectory".format(i)

    # Get a list of all of the .plt files in the folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".plt")]

    # Loop over each .plt file in the folder
    for file in file_list:
        # Create the file path for each .plt file
        file_path = os.path.join(folder_path, file)

        # Load the .plt file into a DataFrame
        df = pd.read_csv(file_path, delimiter=',', header=None, skiprows=7,
                         names=["Latitude", "Longitude", "Reserved_1", "Altitude", "Reserved_2", "Date", "Time"])

        # Convert the Altitude column from feet to meters
        df['Altitude'] = df['Altitude'] * 0.3048

        # Convert the Date and Time columns into a single datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')

        # Drop the original Date and Time columns
        df = df.drop(['Date', 'Time'], axis=1)

        ## Add user column

        df['user'] = "{:03d}".format(i)

        # Append the loaded DataFrame to the list
        df_list.append(df)

# Concatenate all of the DataFrames in the list into a single DataFrame
df = pd.concat(df_list)

df['label'] = ''


def is_between(time, row_to_check):
    if row_to_check['start_time'] <= time <= row_to_check['end_time']:
        return row_to_check['transportation_mode']


df_list_w_labels = []
# add labels to df if they exist
for i in range(182):
    # Create the folder path for each folder
    folder_path = "./Data/{:03d}".format(i)
    # Check to see if a .txt file exists in the folder
    for f in os.listdir(folder_path):
        if f.endswith('.txt'):
            file_path = os.path.join(folder_path, f)
            # Load the .plt file into a DataFrame
            df_labels = pd.read_csv(file_path, delimiter='\t', header=None, skiprows=1,
                                    names=["start_time", "end_time", "transportation_mode"])
            df_labels['user'] = "{:03d}".format(i)
            df_labels['start_time'] = pd.to_datetime(df_labels['start_time'], format='%Y/%m/%d %H:%M:%S')
            df_labels['end_time'] = pd.to_datetime(df_labels['end_time'], format='%Y/%m/%d %H:%M:%S')
            df_user = df[df['user'] == "{:03d}".format(i)]
            for j, row in df_labels.iterrows():
                print('file: ' + str(i) + ' row ' + str(j) + ' of ' + str(len(df_labels)))
                mask = (df_user['Datetime'] >= row['start_time']) and (df_user['Datetime'])
                df_to_apply = df_user[mask]
                df_user.loc[mask, 'label'] = df_to_apply['Datetime'].apply(lambda x: is_between(x, row))
            df_list_w_labels.append(df_user)
            break
df_labels = pd.concat(df_list_w_labels)
df_labels.dropna(subset=['label'], inplace=True)
df_labels = df_labels[df_labels['label'] != '']
df_labels.loc['sequence'] = (df_labels.loc['label'] != df_labels.loc['label'].shift()).cumsum()

df_labels.to_pickle('df_labels.pkl')
