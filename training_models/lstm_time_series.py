from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

from data_presentation.plot_model_history import plot_model_history
from data_cleaning.prepare_data import prepare_data


x_train, y_train, x_test, y_test, _, _ = prepare_data('lstm')
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:][0]


def build_model(input_dims, output):
    # Define the model
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dims, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
#     we added another LSTM layer with 32 units and set return_sequences=False
# to return the output of the LSTM layer only at the final time step.
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output, activation='linear'))

    return model


def run_model():
    lstm_model = build_model(input_shape, output_shape)
    # Compile the model
    lstm_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-4))

    callbacks = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                 keras.callbacks.ModelCheckpoint('../model_weights/lstm_model.h5',
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 mode='min',
                                                 verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                   patience=8, min_lr=1e-6)
                 ]
    # Train the model
    history = lstm_model.fit(x_train, y_train, epochs=250, batch_size=64, validation_split=0.2, callbacks=callbacks)

    plot_model_history(history)

    # Save the model
    lstm_model.save("lstm_model.h5")

    # Evaluate the model
    scores = lstm_model.evaluate(x_test, y_test)
    print("Validation loss: ", scores)


if __name__ == '__main__':
    run_model()
