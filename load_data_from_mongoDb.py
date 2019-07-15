from pymongo import MongoClient
from pandas.io.json import json_normalize
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

def load_data_from_mongoDb():
    #load data
    client = MongoClient('localhost', 27017)
    db = client.image_recognition_within_ml_and_big_data  # image_classification_ann_keras_v1

    #train data
    collectionTrain = db.train_mnist_datasets
    dataTrain = collectionTrain.find()
    #agregate data
    dataTrain = list(dataTrain)
    print(dataTrain[0])
    #show data
    df_train = json_normalize(dataTrain)
    print(df_train.head())
    df_train_x = df_train.iloc[:, 0:784].values
    print("df_train head: ", df_train_x.head())
    df_train_y = df_train['label'].values
    #show image
    first_image = df_train_x[0, :]
    first_label = df_train_y[0]
    #plt.imshow(np.reshape(first_image, (28, 28)), cmap='gray_r')
    #plt.title('Digit Label: {}'.format(first_label))
    #plt.show()
    #standardize data
    sc = StandardScaler()
    df_train_x = sc.fit_transform(df_train_x)
    print('df_train rescaled : ', df_train_x)
    print('df_train_y : ', df_train_y)

    #df_train_y = to_categorical(df_train_y,10)

    return df_train_x, df_train_y

if __name__=="__main__":
    df_train_x, df_train_y = load_data_from_mongoDb()
    print('trainX shape :', df_train_x.shape)
    print('trainy shape :', df_train_y.shape)