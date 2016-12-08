
import numpy as np
import pandas as pd
import random
import nltk
from nltk import NaiveBayesClassifier,classify
from nltk import ConfusionMatrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import pickle

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

    def getFeatures(self):

        featureset = list()

        for Name, Japanese in zip(self.df[self.colName], self.df[self.ethnicityCol]):
            features = self.nameFeatures(Name)
            featureset.append((features, Japanese))
        return featureset


    def nameFeatures(self,name):
        name = str(name).upper()
        return {
            'last_two': name[-2:],
            'last_three' : name[-3:],
            'last_is_vowel' : (name[-1] in 'aeiou')
        }

    def trainAndTest(self,trainingPercent=0.80):
        featureset = self.getFeatures()

        print ("Created a set of features.")
        # random.shuffle(featureset)
        # name_count = len(featureset)
        # cut_point=int(name_count*trainingPercent)
        # self.train_set = featureset[:cut_point]
        # self.test_set  = featureset[cut_point:]
        # print ("Created training set and testing set.")
        self.vec = DictVectorizer()
        X = self.vec.fit_transform([item[0] for item in featureset]).toarray()
        Y = [item[1] for item in featureset]

        print('Vectorized features.')
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, train_size=0.80)

        print('Split training set and testing set 0.80 and 0.20.')
        self.train(x_train,y_train)
        return self.test(x_test, y_test)

    def train(self,x_train, y_train):
        self.clf_NB = MultinomialNB()
        #self.clf_RF = RandomForestClassifier(n_estimators=50, min_samples_split=6)

        self.clf_NB.fit(x_train, y_train)
        #self.clf_RF.fit(x_train, y_train)
        save_classifier = open("naiveBayes.pickle","wb")
        pickle.dump(self.clf_NB, save_classifier)
        save_classifier.close()
        print("Saved Naive Bayes classifier as naiveBayes.pickle")

        # save_classifier2 = open("randomForest.pickle","wb")
        # pickle.dump(self.clf_RF, save_classifier2)
        # save_classifier2.close()
        # print("Saved Random Forest classifier as randomForest.pickle")


    def getMostInformativeFeatures(self,clf,vectorizer):

        print("Prints features with the highest coefficient values:")
        feature_names = vectorizer.get_feature_names()
        top10 = np.argsort(clf.coef_)[-10:]
        print("%s: %s" % ("Japanese", " ".join(feature_names[j] for j in top10)))


    def test(self,x_test, y_test):
        NB_preds = self.clf_NB.predict(x_test)
        #RF_preds = self.clf_RF.predict(x_test)

        print("Accuracy Score: ")
        print(metrics.accuracy_score(y_test, NB_preds))
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(y_test, NB_preds))
        print("Auc Score:")
        probs = nb.predict_proba(x_test)[:, 1]
        print metrics.roc_auc_score(y_test, probs)
        print(self.getMostInformativeFeatures(self.clf_NB, self.vec ))

        # print("Random Forest accuracy and confusion matrix: ")
        # print(metrics.accuracy_score(test_set, RF_preds))
        # print(metrics.confusion_matrix(y_test, RF_preds))
