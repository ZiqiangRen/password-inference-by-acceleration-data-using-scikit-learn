#coding=utf-8
import sys
from sklearn import metrics
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.tree import export_graphviz
from matplotlib.pyplot import *
from sklearn.externals.joblib import Parallel, delayed
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import BernoulliRBM
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets, metrics
import numpy as np
from numpy import random

csvname='total_features.csv'
my_matrixa = np.loadtxt(open(csvname,"rb"),delimiter=",",skiprows=1) 

csvname='total_features_1.csv'
my_matrix = np.loadtxt(open(csvname,"rb"),delimiter=",",skiprows=0) 
csvname2='total_features_2.csv'
my_matrix2 = np.loadtxt(open(csvname2,"rb"),delimiter=",",skiprows=0)

X=my_matrix[0:240,0:6]
y=my_matrix[0:240,7]
testX=my_matrix2[0:60,0:6]
testY=my_matrix2[0:60,7]
Xa=my_matrixa[0:1500,0:6]
ya=my_matrixa[0:1500,7]


def _parallel_helper(obj, methodname, *args, **kwargs):
    return getattr(obj, methodname)(*args, **kwargs)

def mymulticlassSVM():
    print("\nSVM")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xa, ya, test_size=NUM,random_state=0)
    model = OneVsRestClassifier(SVC())
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    dif=y_test-predicted
    #print(predicted)
    #print(dif)
    print(1-sum(dif!=0)/len(y_test))
    
def mymulticlassKNN():
    print("\nKNN")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xa, ya, test_size=NUM,random_state=0)
    model = neighbors.KNeighborsClassifier()   
    model.fit(X_train, y_train)  
    predicted = model.predict(X_test)
    dif=y_test-predicted
    print(1-sum(dif!=0)/len(y_test))  
    
def mymulticlassTree():
    print("\nRandomForest")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xa, ya, test_size=NUM,random_state=0)
    model = RandomForestClassifier(n_estimators = 10)  
    model.fit(X_train, y_train)  
    predicted = model.predict(X_test)
    dif=y_test-predicted
    print(1-sum(dif!=0)/len(y_test))
#     print("feature importances:",model.feature_importances_)
#     print('all trees:',model.estimators_)
#     all_proba = Parallel(n_jobs=10, verbose=model.verbose, backend="threading")(
#                 delayed(_parallel_helper)(e, 'predict_proba', X_test[0]) for e in model.estimators_)
#     print('所有树的判定结果：%s' % all_proba)

#     proba = all_proba[0]
#     for j in range(1, len(all_proba)):
#         proba += all_proba[j]
#     proba /= len(model.estimators_)
#     print('数的棵树：%s ， 判不作弊的树比例：%s' % (model.n_estimators , proba[0,0]))
#     print('数的棵树：%s ， 判作弊的树比例：%s' % (model.n_estimators , proba[0,1]))

#     #当判作弊的树多余不判作弊的树时，最终结果是判作弊
#     print('判断结果：%s' % model.classes_.take(np.argmax(proba, axis=1), axis=0))

#     #把所有的树都保存到word
#     for i in range(len(model.estimators_)):
#         export_graphviz(model.estimators_[i] , '%d.png'%i)

# def mymulticlass_neural_network():
#     print("\nneural_network")
#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xa, ya, test_size=NUM,random_state=0)
#     logistic = linear_model.LogisticRegression()  
#     rbm = BernoulliRBM(random_state=0, verbose=True)   
#     classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]) 
#     rbm.learning_rate = 0.06  
#     rbm.n_iter = 20  
#     # More components tend to give better prediction performance, but larger fitting time
#     rbm.n_components = 100  
#     logistic.C = 6000.0
#     # Training RBM-Logistic Pipeline  
#     classifier.fit(X_train, y_train)  

#     # Training Logistic regression  
#     logistic_classifier = linear_model.LogisticRegression(C=100.0)  
#     logistic_classifier.fit(X_train, y_train)   
#     print()  
#     print("Logistic regression using RBM features:\n%s\n" % (  
#         metrics.classification_report(  
#             y_test,  
#             classifier.predict(X_test))))  

#     print("Logistic regression using raw pixel features:\n%s\n" % (  
#         metrics.classification_report(  
#             y_test,  
#             logistic_classifier.predict(X_test))))
def mymulticlassBayes():
    print("\nBayes")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xa, ya, test_size=NUM,random_state=0)
    #model = MultinomialNB()  
    model=BernoulliNB()
    model.fit(X_train, y_train)  
    predicted = model.predict(X_test)
    dif=y_test-predicted
    print(1-sum(dif!=0)/len(y_test))    
    
def mymulticlassDecision():
    print("\nDecisionTree")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xa, ya, test_size=NUM,random_state=0)
    #model = MultinomialNB()  
    model=DecisionTreeClassifier(max_depth=9)   #depth=9,acc=73%
    model.fit(X_train, y_train)  
    predicted = model.predict(X_test)
    dif=y_test-predicted
    print(1-sum(dif!=0)/len(y_test))  
    
    
    
    
if __name__ == '__main__':
    print("normal:\n")
    try:
        NUM=float(sys.argv[1])
    except:
        NUM=0.2
    print(NUM)
    mymulticlassSVM()
    mymulticlassKNN()
    mymulticlassTree()
    mymulticlassBayes()
    mymulticlassDecision()
    #mymulticlass_neural_network()
