import matplotlib.pyplot as plt
import pandas as pd


def plot_model_history(history):
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    if len(history.history) <= 3:
        plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    if len(history.history) > 3:
        fig_acc.savefig('results/transformer_loss.png')
        plt.close()
        fig_acc = plt.figure(figsize=(10, 10))
        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig_acc.savefig('transformer_accuracy')
        plt.close()
    else:
        fig_acc.savefig('results/lstm_loss.png')
        plt.close()


if __name__ == '__main__':
    print('No model history loaded')
