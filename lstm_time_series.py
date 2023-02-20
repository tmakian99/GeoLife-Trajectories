from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

from prepare_data import prepare_data

x_train, y_train, x_test, y_test, _, _ = prepare_data('lstm')
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:][0]


def build_model(input_dims, output):
    # Define the model
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dims, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(output, activation='linear'))

    return model


def run_model():
    lstm_model = build_model(input_shape, output_shape)
    # Compile the model
    lstm_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-4))

    # Train the model
    history = lstm_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2,)

    # Save the model
    lstm_model.save("lstm_model.h5")

    # Evaluate the model
    scores = lstm_model.evaluate(x_test, y_test)
    print("Validation loss: ", scores)


if __name__ == '__main__':
    run_model()
