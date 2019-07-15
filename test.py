from load_data_from_csv import load_data_from_csv
from load_data_from_mongoDb import load_data_from_mongoDb
from load_data_from_rawData_Img import imageLoader
from load_data_from_rawData_txt import sentenceLoader
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from pickle import dump, load
from sklearn.metrics import accuracy_score
from model import KNNClassifierModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from .model import agregation_of_heterogenous_datas, reduction_of_dimension_with_PCA, reduction_of_dimension_with_LDA

def test():
    #load heterogenous data
    Xtrain1, Xtest1 = imageLoader()
    Ytrain1, Ytest1 = sentenceLoader()
    Xtest2, Ytest2, Xtrain2, Ytrain2 = load_data_from_csv()
    df_train_x, df_train_y = load_data_from_mongoDb()
    Xtrain3, Xtest3, Ytrain3, Ytest3 = train_test_split(df_train_x, df_train_y, random_state=0, test_size=0.2)

    #agregate data with numpy
    Xtrain, Ytrain, Xtest, Ytest = agregation_of_heterogenous_datas(Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, Xtest1, Ytest1, Xtest2, Ytest2, Xtest3, Ytest3)
    #reduce dimension of agregated data with PCA
    Xtrain, Xtest = reduction_of_dimension_with_PCA(Xtrain, Xtest)
    # reduce dimension of agregated data with LDA
    #Xtrain, Xtest = reduction_of_dimension_with_LDA(Xtrain, Xtest, Ytrain)

    #call model
    KNNClassifier = KNNClassifierModel()
    #fit data into KNNClassifier
    KNNClassifier.fit(Xtrain, Ytrain)
    #evaluate
    predicted = KNNClassifier.predict(Xtest)
    #Ytest11 = list()
    #for elt in Ytest1:
    #    elt = int(elt)
    #    Ytest11.append(elt)
    print('accuracy : {}%'.format(str(accuracy_score(Ytest, predicted)*100)))
    print(classification_report(Ytest, predicted))
    print(confusion_matrix(Ytest, predicted))
    print('predicted shape : ', predicted.shape)
    print('Ytest1 shape : ', Ytest.shape)

if __name__=='__main__':
    test()