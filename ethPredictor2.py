
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.pipeline import Pipeline
import re
from sklearn.feature_selection import RFECV

class ethPredictor2():

    ## colName: the name of the column for names (Lastname, Firstname)

    ## ethCol: the name of the column that contains ethnicity information.

    ## ethnicityCol: the name of dummy variable column that indicates the interested etnnicity.

    ## Read in data with specified variables.
    def __init__(self, fileName, colName, ethnicityCol):
        print ("Reading data from file: " + fileName)
        self.df = pd.read_csv(fileName, encoding='latin-1')
        self.colName = colName
        self.ethnicityCol = ethnicityCol
        self.ratio = self.df.groupby(ethnicityCol).count()
        print(self.ratio)

    # def getFeatures(self):
        
    #     featureset = []
    #     feature_names=["last_two", "last_three", "first_three"]
    #     for Name, Japanese in zip(self.df[self.colName], self.df[self.ethnicityCol]):
    #         features = self.nameFeatures(Name)
    #         featureset.append(features,)
       
    #     return featureset


    # def nameFeatures(self,name):
    #     name = str(name).upper()
    #     return {
    #         'last_two': name[-2:],
    #         'last_three' : name[-3:],
    #         'first_three' : name[:3]
    #     }

    def trainAndTest(self,trainingPercent=0.80):
        featureset=[]
        for Lastname, Firstname in zip(self.df[self.colName], self.df['FName']):
            featureset.append(Lastname + ' ' + Firstname)
        
        def words_and_char_bigrams(text):
            words = re.findall(r'\w{3,}', text)
            for w in words:
                yield w
                for i in range(len(w) - 2):
                    yield w[i:i+3]

        self.vec = CountVectorizer(analyzer=words_and_char_bigrams)
       
        X = self.vec.fit_transform(featureset).toarray()
        
        Y = self.df[self.ethnicityCol].values
        
        names = self.vec.get_feature_names()
        print('Created Features: ', random.sample(names, 7))
        print("length of x: " ,len(X) , "; length of y: " , len(Y))
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, train_size=0.80)

        print('Split training set and testing set 0.80 and 0.20.')
        self.train(x_train,y_train)
        return self.test(x_test, y_test)

    def train(self,x_train, y_train):
        
        self.clf_RF = RandomForestClassifier(class_weight="balanced_subsample")
        self.clf_RF.fit(x_train, y_train)
        
        save_classifier2 = open("randomForest.pickle","wb")
        pickle.dump(self.clf_RF, save_classifier2)
        save_classifier2.close()
        print("Saved Random Forest classifier as randomForest.pickle")


    def test(self,x_test, y_test):
        
        RF_preds = self.clf_RF.predict(x_test)

        topfeatures= []
        feature_names = self.vec.get_feature_names()
        top10 = np.argsort(self.clf_RF.feature_importances_)[-10:]
        for i in top10:
            topfeatures.append(feature_names[i])
       
        print("Accuracy: ",metrics.accuracy_score(y_test, RF_preds))
        print("Confusion Matrix: ")
        cm = metrics.confusion_matrix(y_test, RF_preds)
        print(cm)
        sensitivity = float(cm[1][1]) / float(cm[1][0]+cm[1][1])
        specificity = float(cm[0][0]) / float(cm[0][0]+cm[0][1])
        weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
        print('Sensitivity ',sensitivity, '; Specificity: ', specificity)
        print('Weighted Accuracy: ', weightedAccuracy)

        print("Most informative features: ")
        print(topfeatures)






