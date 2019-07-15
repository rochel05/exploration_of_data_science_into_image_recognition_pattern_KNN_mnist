import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

def load_data_from_csv():
    #test data
    df_test = pd.read_csv('data/mnist_test.csv')
    print("df test : ", df_test.head())
    print("df test shape : ", df_test.shape)
    df_test_x = df_test.iloc[:, 1:1784]
    df_test_y = df_test['label']
    #standardize data
    sc = StandardScaler()
    df_test_x = sc.fit_transform(df_test_x)
    print('df_test rescaled : ', df_test_x)
    print('df_test_y : ', df_test_y)

    #train data
    df_train = pd.read_csv('data/mnist_train.csv')
    print("df_train : ", df_train.head())
    print("df_train shape: ", df_train.shape)
    df_train_x = df_train.iloc[:, 1:]
    df_train_y = df_train['label']
    first_image = df_train_x.loc[0, :]
    first_label = df_train_y[0]
    plt.imshow(np.reshape(first_image.values, (28, 28)), cmap='gray_r')
    plt.title('Digit Label: {}'.format(first_label))
    plt.show()
    df_train_x = sc.fit_transform(df_train_x)
    print('df_train rescaled : ', df_train_x)
    print('df_train_y : ', df_train_y)

    #df_test_y = to_categorical(df_test_y,10)
    #df_train_y = to_categorical(df_train_y,10)

    return df_test_x, df_test_y,df_train_x, df_train_y

if __name__=="__main__":
    df_test_x, df_test_y,df_train_x, df_train_y =load_data_from_csv()
    print('trainX shape :', df_train_x.shape)
    print('trainy shape :', df_train_y.shape)
    print('testX shape :', df_test_x.shape)
    print('testY shape :', df_test_y.shape)
