from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from pickle import dump
import os
import glob
import cv2

#define path directory
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

#define image features extractor function
def train_image_features_extracor(directory):
    train_X = list()
    for dossier in os.listdir(directory):
        if(dossier.title()=="0"):
            for img in os.listdir(directory + "/0/"):
                # print img
                filename = directory + "/0/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                #image = cv2.imread(filename, 0)
                print(image.shape)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="1"):
            for img in os.listdir(directory + "/1/"):
                # print img
                filename = directory + "/1/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="2"):
            for img in os.listdir(directory + "/2/"):
                # print img
                filename = directory + "/2/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="3"):
            for img in os.listdir(directory + "/3/"):
                # print img
                filename = directory + "/3/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="4"):
            for img in os.listdir(directory + "/4/"):
                # print img
                filename = directory + "/4/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="5"):
            for img in os.listdir(directory + "/5/"):
                # print img
                filename = directory + "/5/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="6"):
            for img in os.listdir(directory + "/6/"):
                # print img
                filename = directory + "/6/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="7"):
            for img in os.listdir(directory + "/7/"):
                # print img
                filename = directory + "/7/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="8"):
            for img in os.listdir(directory + "/8/"):
                # print img
                filename = directory + "/8/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
        elif(dossier.title()=="9"):
            for img in os.listdir(directory + "/9/"):
                # print img
                filename = directory + "/9/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                train_X.append(image)
    save_files(data_generated_path, 'train_images_features.pkl', train_X)
    dump(train_X, open('data_generated_mnist/featuresTrainX.pkl', 'wb'))
    return train_X

def eval_image_features_extractor(directory):
    test_X = list()
    for dossier in os.listdir(directory):
        if(dossier.title()=="0"):
            for img in os.listdir(directory + "/0/"):
                # print img
                filename = directory + "/0/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="1"):
            for img in os.listdir(directory + "/1/"):
                # print img
                filename = directory + "/1/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="2"):
            for img in os.listdir(directory + "/2/"):
                # print img
                filename = directory + "/2/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="3"):
            for img in os.listdir(directory + "/3/"):
                # print img
                filename = directory + "/3/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="4"):
            for img in os.listdir(directory + "/4/"):
                # print img
                filename = directory + "/4/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="5"):
            for img in os.listdir(directory + "/5/"):
                # print img
                filename = directory + "/5/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="6"):
            for img in os.listdir(directory + "/6/"):
                # print img
                filename = directory + "/6/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="7"):
            for img in os.listdir(directory + "/7/"):
                # print img
                filename = directory + "/7/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="8"):
            for img in os.listdir(directory + "/8/"):
                # print img
                filename = directory + "/8/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
        elif(dossier.title()=="9"):
            for img in os.listdir(directory + "/9/"):
                # print img
                filename = directory + "/9/" + img
                image = load_img(filename, target_size=(28, 28))
                image = np.mean(image, axis=2)
                # image = cv2.imread(filename, 0)
                image = img_to_array(image)
                test_X.append(image)
    save_files(data_generated_path, 'test_images_features.pkl', test_X)
    dump(test_X, open('data_generated_mnist/featuresTestX.pkl', 'wb'))
    return test_X


def imageLoader():
    #extract image from zip file
    trainX = train_image_features_extracor(train_img_path)
    testX = eval_image_features_extractor(eval_img_path)
    trainX = np.reshape(trainX, (-1, 28, 28))
    testX = np.reshape(testX, (-1, 28, 28))
    #reshape data
    #convert image data type to float32
    trainX = np.array(trainX, dtype='float32')
    testX = np.array(testX, dtype='float32')
    #rescale array image between 0 and 1
    trainX = trainX/255
    testX = testX/255
    trainX = np.reshape(trainX, (-1, 784))
    testX = np.reshape(testX, (-1, 784))
    dump(trainX, open('data_generated_mnist/featuresTrainX.pkl', 'wb'))
    dump(testX, open('data_generated_mnist/featuresTestX.pkl', 'wb'))
    if is_files_exist(data_generated_path, 'featuresTrainX.pkl'): print ('le fichier featuresTrain.pkl est deja existe!!!...')
    else: save_files(data_generated_path, 'featuresTrainX.txt', trainX); print ('le fichier featuresTrain.pkl est enregistre avec succes!!!...')
    if is_files_exist(data_generated_path, 'featuresTestX.pkl'): print ('le fichier featuresTest.pkl est deja existe!!!...')
    else: save_files(data_generated_path, 'featuresTestX.txt', testX); print ('le fichier featuresTest.txt est enregistre avec succes!!!...')
    return trainX, testX

def plot_image_from_dataset():
    # plot and view image in dataset
    trainX, testX = imageLoader()
    plt.figure(figsize=[5, 5])
    plt.suptitle('image recognition model [show samples in dataset]')
    plt.subplot(121)
    print('trainX', trainX[0])
    print('testX', testX[0])
    plt.imshow(np.reshape(trainX[0], (28, 28)), interpolation='none')
    plt.title('category : 0')

    plt.subplot(122)
    plt.imshow(np.reshape(testX[60], (28, 28)), interpolation='none')
    plt.title('category: 1')
    plt.show()

if __name__=='__main__':
    trainX, testX = imageLoader()
    print('trainX shape :', trainX.shape)
    print('trainy shape :', testX.shape)
    #plot_image_from_dataset()
