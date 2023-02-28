from tensorflow import keras
from tensorflow.keras import layers
from data_cleaning.prepare_data import prepare_data
from data_presentation.plot_model_history import plot_model_history

x_train, y_train, x_test, y_test, _, num_classes = prepare_data('transformer')


# from keras nlp transformer encoder definition, based upon 'Attention is all you need' architecture:
# https://github.com/keras-team/keras-nlp/blob/v0.3.1/keras_nlp/layers/transformer_encoder.py#L24
def transformer_encoder(inputs, num_heads, dropout=0):
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

    x = layers.LayerNormalization(
        epsilon=1e-6,
    )(inputs + x)

    x = layers.Dense(
        4,
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(x)
    x = layers.Dropout(rate=dropout)(x)
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

    return res


def build_model(
        input_shape,
        num_heads,
        num_transformer_blocks,
        dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        res = transformer_encoder(x, num_heads, dropout)
        # Added convolutions layers as per Keras example code here:
        # https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_classification_transformer.py
        x = layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6,)(x)
        x = x + res

    # Global pooling to reduce dimensions down to that of the number of classes
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def run_model():
    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        num_heads=4,
        num_transformer_blocks=4,
        dropout=0.25
    )

    # Using legacy Adam optimizer because of macOS compatibility issues
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    # Reduce learning rate on plateau allows for better approximation of local minimum
    callbacks = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                 keras.callbacks.ModelCheckpoint('transformer_model_3.h5',
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 mode='min',
                                                 verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                   patience=8, min_lr=1e-6)
                 ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    plot_model_history(history)

    model.evaluate(x_test, y_test, verbose=1)


if __name__ == '__main__':
    run_model()
