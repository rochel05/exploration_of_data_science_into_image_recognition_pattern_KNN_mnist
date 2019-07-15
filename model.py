from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

def KNNClassifierModel():
    model = KNeighborsClassifier(n_neighbors=10)
    return model

def agregation_of_heterogenous_datas(Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, Xtest1, Ytest1, Xtest2, Ytest2, Xtest3, Ytest3):
    #just comment and uncomment the lines if you choose to train only mongo data with csv data or mongo with raw
    Xtrain = np.append(Xtrain1, Xtrain2, axis=0)
    Xtest = np.append(Xtest1, Xtest2, axis=0)
    Ytrain = np.append(Ytrain1, Ytrain2, axis=0)
    Ytest = np.append(Ytest1, Ytest2, axis=0)

    #Xtrain = np.append(Xtrain, Xtrain3, axis=0)
    #Xtest = np.append(Xtest, Xtest3, axis=0)
    #Ytrain = np.append(Ytrain, Ytrain3, axis=0)
    #Ytest = np.append(Ytest, Ytest3, axis=0)

    print(' Xtrain1 shape : {} - Ytrain1 shape : {}'.format(Xtrain1.shape, Ytrain1.shape))
    print(' Xtest1 shape : {} - Ytest1 shape : {}'.format(Xtest1.shape, Ytest1.shape))
    print(' Xtrain2 shape : {} - Ytrain2 shape : {}'.format(Xtrain2.shape, Ytrain2.shape))
    print(' Xtest2 shape : {} - Ytest2 shape : {}'.format(Xtest2.shape, Ytest2.shape))
    #print(' Xtrain3 shape : {} - Ytrain3 shape : {}'.format(Xtrain3.shape, Ytrain3.shape))
    #print(' Xtest3 shape : {} - Ytest3 shape : {}'.format(Xtest3.shape, Ytest3.shape))
    print(' Xtrain shape : {} - Ytrain shape : {}'.format(Xtrain.shape, Ytrain.shape))
    print(' Xtest shape : {} - Ytest shape : {}'.format(Xtest.shape, Ytest.shape))
    return Xtrain, Ytrain, Xtest, Ytest

def reduction_of_dimension_with_PCA(Xtrain, Xtest):
    #standardize data with StandardScaler
    sc = StandardScaler()
    Xtrain = sc.fit_transform(Xtrain)
    Xtest = sc.transform(Xtest)
    #call pca function of sklearn
    pca = PCA()
    Xtrain = pca.fit_transform(Xtrain)
    Xtest = pca.transform(Xtest)
    return Xtrain, Xtest

def reduction_of_dimension_with_LDA(Xtrain, Xtest, Ytrain):
    #standardize data with StandardScaler
    sc = StandardScaler()
    Xtrain = sc.fit_transform(Xtrain)
    Xtest = sc.transform(Xtest)
    #call lda function of sklearn
    lda = LDA(n_components=10)
    Xtrain = lda.fit_transform(Xtrain, Ytrain)
    Xtest = lda.transform(Xtest)
    return Xtrain, Xtest
