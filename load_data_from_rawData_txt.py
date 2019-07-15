from keras.utils import to_categorical
import glob
from pickle import dump
import numpy as np

#define variables
n_classes = 10

#define path directory
train_labels_path = 'data/mnist_train'
eval_labels_path = 'data/mnist_test'
train_img_path =  'data/mnist_train'
eval_img_path = 'data/mnist_test'
data_generated_path = 'data_generated_mnist/'

#define function to save files
def save_files(path, filename, objet):
    fichier = open(path + filename, 'w')
    fichier.write(objet.__str__())
    fichier.close()

#define function which verify if file is exist
def is_files_exist(path, name):
    if glob.glob(path + name): return True
    else: return False

#load train sentence
def load_doc(filename):
    fichier = open(filename, 'r')
    data = fichier.read()
    fichier.close()
    return data

#define cleaned_train_label function
def clean_train_labels():
    train_Y = list()
    filename = train_labels_path + '/mnist_train.txt'
    data_train = load_doc(filename)
    for item in data_train.split('\n'):
        if item.__eq__(''): pass
        else:
            train_Y.append(item)
    return train_Y

#define cleaned test label function
def cleaned_test_labels():
    test_Y = list()
    filename = eval_labels_path + '/mnist_test.txt'
    data_test = load_doc(filename)
    for item in data_test.split('\n'):
        if item.__eq__(''): pass
        else:
            test_Y.append(item)
    return test_Y

#define sentence loader function
def sentenceLoader():
    trainY = clean_train_labels()
    testY = cleaned_test_labels()

    print ('trainY[0] : {}'.format(trainY[0]))
    trainY = to_categorical(trainY, num_classes=n_classes)
    print ('trainY[0] onehot encoded : {}'.format(trainY[0]))
    print ('trainY shape : {}'.format(trainY.shape))

    print ('testY[0] : {}'.format(testY[0]))
    testY = to_categorical(testY, num_classes=n_classes)
    print ('testY[0] onehot encoded : {}'.format(testY[0]))
    print ('testY shape : {}'.format(testY.shape))

    print ('\ntrainY shape : {}'.format(trainY.shape))
    print ('testY shape : {}'.format(testY.shape))
    if is_files_exist(data_generated_path, 'descriptionTrain.txt'): print ('le fichier descriptionTrain.txt est deja existe!!!...')
    else: save_files(data_generated_path, 'descriptionTrain.txt', trainY); print ('le fichier descriptionTrain.txt est enregistre avec succes!!!...')
    if is_files_exist(data_generated_path, 'descriptionTest.txt'): print ('le fichier descriptionTest.pkl est deja existe!!!...')
    else: save_files(data_generated_path, 'descriptionTest.txt', testY); print ('le fichier descriptionTest.txt est enregistre avec succes!!!...')
    dump(trainY, open('data_generated_mnist/featuresTrainY.pkl', 'wb'))
    dump(testY, open('data_generated_mnist/featuresTestY.pkl', 'wb'))

    trainY = np.array(trainY)
    testY = np.array(testY)
    trainY = np.resize(trainY, (-5400,))
    testY = np.resize(testY, (-3141,))

    return trainY, testY


if __name__=="__main__":
    trainY, testY = sentenceLoader()
    print('trainY shape :', trainY.shape)
    print('testY shape :', testY.shape)