import matplotlib.pyplot as plt
import pandas as pd


def plot_model_history(history):
    df = pd.read_csv(history)
    if history == 'lstm_history.txt':
        df = df.iloc[::2]
        df = df.iloc[:, 0].str.split(' ', expand=True)
        df.drop([0, 1, 2, 3, 4, 5, 6, 8, 9], axis=1, inplace=True)
        df.columns = ['loss', 'val_loss']
        df.reset_index(inplace=True, drop=True)
        label_names = ['loss',
                       'val_loss',
                       'lstm_loss.png',
                       ]
    else:
        df = df.iloc[::3]
        df = df.iloc[:, 0].str.split(' ', expand=True)
        df.drop([0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 19], axis=1, inplace=True)
        df.columns = ['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy']
        df.reset_index(inplace=True, drop=True)
        label_names = ['loss',
                       'val_loss',
                       'transformer_loss.png',
                       'sparse_categorical_accuracy',
                       'val_sparse_categorical_accuracy',
                       'transformer_accuracy.png'
                       ]
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(df[label_names[0]])
    plt.plot(df[label_names[1]])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig_acc.savefig(label_names[2])
    if history == 'transformer_history.txt':
        fig_acc = plt.figure(figsize=(10, 10))
        plt.plot(df[label_names[3]])
        plt.plot(df[label_names[4]])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig_acc.savefig(label_names[5])


if __name__ == '__main__':
    print('No model history loaded')
