import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import clean_data

# Create sequences
sequences, labels = clean_data.create_sequences()

# Group similar modes of transportation, and drop very infrequent modes
label_dict = {'walk': 0, 'bus': 1, 'bike': 2, 'train': 3, 'car': 1, 'subway': 3, 'taxi': 1,
              'run': 0, 'motorcycle': 1}
new_labels = np.array([label_dict.get(item, item) for item in labels])

# Split data into train and test, and hold test until after training
x_train, x_test, y_train, y_test = train_test_split(sequences, new_labels, test_size=0.2)
num_classes = len(np.unique(new_labels))

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


# from keras nlp transformer encoder definition, based upon 'Attention is all you need' architecture:
# https://github.com/keras-team/keras-nlp/blob/v0.3.1/keras_nlp/layers/transformer_encoder.py#L24
def transformer_encoder(inputs, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    feature_size = inputs.shape[-1]
    attention_head_size = int(feature_size // num_heads)
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=attention_head_size,
        value_dim=attention_head_size,
        dropout=dropout,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(inputs, inputs, inputs)

    x = layers.Dropout(rate=dropout)(x)

    x = layers.LayerNormalization(
        epsilon=1e-6,
    )(inputs + x)

    x = layers.Dense(
        ff_dim,
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(x)
    x = layers.Dense(
        feature_size,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(x)

    x = layers.LayerNormalization(
        epsilon=1e-6,
    )(inputs + x)

    x = keras.layers.Dropout(rate=dropout)(x)
    res = x + inputs

    # Feed Forward Part, with added convolutions layers
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
        input_shape,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, num_heads, ff_dim, dropout)

    # Global pooling to reduce dimensions down to that of the number of classes
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
             keras.callbacks.ModelCheckpoint('transformer_model.h5',
                                             monitor='val_loss',
                                             save_best_only=True,
                                             mode='min',
                                             verbose=1),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                               patience=8, min_lr=1e-6)
             ]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)
